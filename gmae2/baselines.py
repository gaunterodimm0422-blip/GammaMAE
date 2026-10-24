# -*- coding: utf-8 -*-
"""经典基线：逻辑回归 (验证集选 C)、余弦 KNN (验证集选 k)、泊松模板拟合 (源模板 + 本底模板, ML-EM)。"""
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from . import config as C


def logreg(Xtr, ytr, Xva, yva, Cs=(0.01, 0.1, 1.0), seed=0, log=print):
    best = None
    for c in Cs:
        m = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(Xtr, ytr)
        a = (m.predict(Xva) == yva).mean()
        log(f"    logreg C={c}: val acc {a:.4f}")
        if best is None or a > best[0]:
            best = (a, c, m)
    return best[2], dict(C=best[1], val_acc=float(best[0]))


def _norm(x, dev):
    x = torch.as_tensor(x, device=dev)
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def knn_predict(Xtr, ytr, Xq, k, dev="cpu", bs=1024):
    A = _norm(Xtr, dev); yt = torch.as_tensor(ytr, device=dev); out = []
    nc = int(ytr.max()) + 1
    for i in range(0, len(Xq), bs):
        s = _norm(Xq[i:i + bs], dev) @ A.T
        v, j = s.topk(k, dim=1)
        votes = torch.zeros(len(j), nc, device=dev).scatter_add_(1, yt[j], v)
        out.append(votes.argmax(1).cpu().numpy())
    return np.concatenate(out)


def knn(Xtr, ytr, Xva, yva, ks=(1, 5, 15), dev="cpu", log=print):
    best = None
    for k in ks:
        a = (knn_predict(Xtr, ytr, Xva, k, dev) == yva).mean()
        log(f"    knn k={k}: val acc {a:.4f}")
        if best is None or a > best[0]:
            best = (a, k)
    return best[1], dict(k=best[1], val_acc=float(best[0]))


class TemplateFit:
    """每个 (探测器, 核素, 距离) 一个源模板 = 该组 train 谱的平均 (已含名义本底)，
    另加探测器本底模板；对每条谱做 c ~ Poisson(a*T_k + b*B) 的 ML-EM 拟合，取对数似然最大的核素。"""
    def __init__(self, counts, y, det, dist, bkg_templates, eps=1e-6):
        counts = counts() if callable(counts) else counts
        self.T, self.lab, self.tdet = [], [], []
        for di in np.unique(det):
            for c in np.unique(y):
                for r in np.unique(dist):
                    m = (det == di) & (y == c) & (dist == r)
                    if m.any():
                        s = counts[m].sum(0).astype(np.float64) + eps
                        self.T.append(s / s.sum()); self.lab.append(c); self.tdet.append(di)
        self.T, self.lab, self.tdet = np.array(self.T), np.array(self.lab), np.array(self.tdet)
        self.B = {di: (bkg_templates[di] + eps) / (bkg_templates[di] + eps).sum() for di in np.unique(det)}

    def predict(self, counts, det, dev="cpu", iters=60, bs=256):
        """counts: (N, n) 数组或可调用对象 f(idx)->计数"""
        pred = np.zeros(len(det), np.int64)
        for di in np.unique(det):
            sel = np.where(det == di)[0]
            T = torch.as_tensor(self.T[self.tdet == di], device=dev, dtype=torch.float32)       # (K, n)
            B = torch.as_tensor(self.B[di], device=dev, dtype=torch.float32)                    # (n,)
            lab = self.lab[self.tdet == di]
            for i in range(0, sel.size, bs):
                cc = counts(sel[i:i + bs]) if callable(counts) else counts[sel[i:i + bs]]
                c = torch.as_tensor(cc, device=dev, dtype=torch.float32)                         # (b, n)
                N = c.sum(1, keepdim=True).clamp(min=1)
                a = 0.5 * N.expand(-1, T.shape[0]).clone(); b = 0.5 * N.expand(-1, T.shape[0]).clone()
                for _ in range(iters):
                    lam = a[..., None] * T[None] + b[..., None] * B[None, None]                 # (b, K, n)
                    r = c[:, None, :] / lam.clamp(min=1e-12)
                    a = a * (r * T[None]).sum(-1); b = b * (r * B[None, None]).sum(-1)
                lam = a[..., None] * T[None] + b[..., None] * B[None, None]
                ll = (c[:, None, :] * torch.log(lam.clamp(min=1e-12))).sum(-1) - lam.sum(-1)
                pred[sel[i:i + bs]] = lab[ll.argmax(1).cpu().numpy()]
        return pred
