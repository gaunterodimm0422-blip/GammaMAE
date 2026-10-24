# -*- coding: utf-8 -*-
"""模型：CNN1D / ResNet1D / Transformer(从零训练) / GammaMAE v2 (MSM + DCL 预训练)。"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def _pos_head(c_in, n_pos, n_cls, c_red=32, hid=128):
    """保留能量位置信息的分类头：1x1 卷积降维后展平（不用全局平均池化，否则峰位信息丢失）。"""
    return nn.Sequential(nn.Conv1d(c_in, c_red, 1), nn.GELU(), nn.Flatten(), nn.Dropout(0.2),
                         nn.Linear(c_red * n_pos, hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, n_cls))


class CNN1D(nn.Module):
    def __init__(self, n_bins, n_cls, w=32):
        super().__init__()
        ch = [1, w, 2 * w, 4 * w, 4 * w, 8 * w]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv1d(a, b, 7, padding=3), nn.BatchNorm1d(b), nn.GELU(), nn.MaxPool1d(2)]
        self.f = nn.Sequential(*layers)
        self.head = _pos_head(ch[-1], n_bins // 2 ** (len(ch) - 1), n_cls)

    def forward(self, x, det=None):
        return self.head(self.f(x.unsqueeze(1)))


class _Res(nn.Module):
    def __init__(self, a, b, s):
        super().__init__()
        self.c1 = nn.Conv1d(a, b, 7, s, 3); self.b1 = nn.BatchNorm1d(b)
        self.c2 = nn.Conv1d(b, b, 7, 1, 3); self.b2 = nn.BatchNorm1d(b)
        self.sk = nn.Sequential() if (a == b and s == 1) else nn.Sequential(nn.Conv1d(a, b, 1, s), nn.BatchNorm1d(b))

    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.sk(x))


class ResNet1D(nn.Module):
    def __init__(self, n_bins, n_cls, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, w, 15, 2, 7), nn.BatchNorm1d(w), nn.GELU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(_Res(w, w, 1), _Res(w, 2 * w, 2), _Res(2 * w, 4 * w, 2), _Res(4 * w, 8 * w, 2),
                               _Res(8 * w, 8 * w, 2))
        self.head = _pos_head(8 * w, n_bins // 64, n_cls)

    def forward(self, x, det=None):
        return self.head(self.f(self.stem(x.unsqueeze(1))))


def _encoder(d, L, h, drop):
    layer = nn.TransformerEncoderLayer(d, h, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, L, enable_nested_tensor=False)


class SpectrumEncoder(nn.Module):
    """分块嵌入 + 位置嵌入 + 探测器嵌入 + Transformer 编码器 (GammaMAE 与 Transformer 基线共用)。"""
    def __init__(self, n_bins=2048, patch=32, d=256, L=6, heads=8, n_det=3, drop=0.1, use_det=True):
        super().__init__()
        self.patch, self.n_p, self.d, self.use_det = patch, n_bins // patch, d, use_det
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_p, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.det = nn.Embedding(n_det, d); nn.init.trunc_normal_(self.det.weight, std=0.02)
        self.blocks = _encoder(d, L, heads, drop)
        self.norm = nn.LayerNorm(d)

    def tokens(self, x, det):
        z = self.embed(x.view(x.shape[0], self.n_p, self.patch)) + self.pos
        if self.use_det:
            z = z + self.det(det).unsqueeze(1)
        return z

    def forward(self, x, det, keep=None):
        z = self.tokens(x, det)
        if keep is not None:
            z = torch.gather(z, 1, keep.unsqueeze(-1).expand(-1, -1, self.d))
        return self.norm(self.blocks(z))

    def pooled(self, x, det):
        return self(x, det).mean(1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_bins, n_cls, enc_kw=None, encoder=None):
        super().__init__()
        self.enc = encoder if encoder is not None else SpectrumEncoder(n_bins, **(enc_kw or {}))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.d, n_cls))

    def forward(self, x, det):
        return self.head(self.enc.pooled(x, det))


class GammaMAE(nn.Module):
    """掩码谱建模 (MSM) + 探测器对比学习 (DCL)。"""
    def __init__(self, n_bins=2048, patch=32, d=256, L=6, heads=8, dec_d=128, dec_L=2, mask_ratio=0.5,
                 n_det=3, drop=0.1, proj=128):
        super().__init__()
        self.enc = SpectrumEncoder(n_bins, patch, d, L, heads, n_det, drop)
        self.mask_ratio, self.patch, self.n_p = mask_ratio, patch, n_bins // patch
        self.dec_in = nn.Linear(d, dec_d)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dec_d)); nn.init.trunc_normal_(self.mask_tok, std=0.02)
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n_p, dec_d)); nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.dec_det = nn.Embedding(n_det, dec_d)
        self.dec = _encoder(dec_d, dec_L, 4, drop)
        self.dec_out = nn.Linear(dec_d, patch)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Linear(d, proj))   # BN 防止表示坍缩 (SimCLR 式投影头)

    def forward(self, x, det):
        B = x.shape[0]
        nv = max(1, int(round(self.n_p * (1 - self.mask_ratio))))
        ids = torch.rand(B, self.n_p, device=x.device).argsort(1)
        keep = ids[:, :nv]
        h = self.enc(x, det, keep)                                     # (B, nv, d)
        hd = self.dec_in(h)
        full = self.mask_tok.expand(B, self.n_p, -1).to(hd.dtype).clone()   # AMP 下 hd 为 fp16, 需同类型
        full.scatter_(1, keep.unsqueeze(-1).expand(-1, -1, hd.shape[-1]), hd)
        full = full + self.dec_pos + self.dec_det(det).unsqueeze(1)
        pred = self.dec_out(self.dec(full))                            # (B, n_p, patch)
        mask = torch.ones(B, self.n_p, device=x.device, dtype=torch.bool); mask.scatter_(1, keep, False)
        return pred, mask

    def embed_full(self, x, det):
        """DCL 用的嵌入：完整（不掩码）谱的池化表示 -> 投影头 -> 单位球。与下游微调/探针看到的输入一致。"""
        return F.normalize(self.proj(self.enc.pooled(x, det)), dim=1)


def msm_loss(pred, x, mask):
    tgt = x.view(pred.shape)
    l = ((pred - tgt) ** 2).mean(-1)
    return (l * mask).sum() / mask.sum().clamp(min=1)


def dcl_loss(z, y, det, tau=0.2):
    """探测器混合批次上的监督对比损失 (SupCon; z 只含有标注样本)。
    正样本 = 同核素 (含不同探测器的同核素谱)，负样本 = 不同核素。批次中三个探测器混合，
    因此损失同时要求同一核素在不同探测器上的表示靠近。"""
    if len(z) < 2:
        return z.sum() * 0
    with torch.autocast(z.device.type, enabled=False):      # 相似度与 logsumexp 用 fp32 计算 (AMP 下防溢出)
        return _supcon(z.float(), y, tau)


def _supcon(z, y, tau):
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
