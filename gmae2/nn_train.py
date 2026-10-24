# -*- coding: utf-8 -*-
"""神经网络训练：有监督分类 (早停看验证集) 与 GammaMAE 预训练；全程记录训练曲线。"""
import copy, math, time
import numpy as np, torch, torch.nn.functional as F
from .models import msm_loss, dcl_loss


def _t(a, dev, dt=None):
    return torch.as_tensor(a, device=dev, dtype=dt)


def _sched(opt, total, warm):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))


@torch.no_grad()
def predict(model, X, det, dev, bs=1024):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = _t(X[i:i + bs], dev, torch.float32); d = _t(det[i:i + bs], dev, torch.long)
        with torch.autocast(dev.split(":")[0], enabled=dev.startswith("cuda")):
            out.append(model(x, d).float().argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_classifier(model, tr, va, dev, epochs=60, lr=1e-3, wd=0.05, bs=256, patience=10, seed=0, log=print,
                     lr_encoder_scale=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(dev)
    X, y, d = _t(tr["X"], dev, torch.float32), _t(tr["y"], dev, torch.long), _t(tr["det"], dev, torch.long)
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    groups = [{"params": rest, "lr": lr}] + ([{"params": enc, "lr": lr * lr_encoder_scale}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    steps = epochs * math.ceil(len(X) / bs); sch = _sched(opt, steps, max(1, steps // 20))
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist, best, best_state, bad = [], -1, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0.0
        perm = torch.randperm(len(X), generator=g).to(dev)
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            with torch.autocast(dev.split(":")[0], enabled=dev.startswith("cuda")):
                loss = F.cross_entropy(model(X[j], d[j]), y[j], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); tl += loss.item() * len(j)
        va_acc = float((predict(model, va["X"], va["det"], dev) == va["y"]).mean())
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(X), val_acc=va_acc, sec=time.time() - t0))
        log(f"      ep {ep + 1:3d} loss {tl / len(X):.4f} val {va_acc:.4f}")
        if va_acc > best:
            best, best_state, bad = va_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, dict(best_val_acc=best, epochs_run=len(hist), history=hist)


def pretrain_gmae(model, tr, va, dev, epochs=40, lr=1e-3, wd=0.05, bs=256, lam_dcl=0.5, seed=0, log=print):
    """tr: X, det, y, labeled (bool, DCL 只用 labeled 样本)。lam_dcl=0 即纯 MSM。"""
    torch.manual_seed(seed)
    model.to(dev)
    X, d = _t(tr["X"], dev, torch.float32), _t(tr["det"], dev, torch.long)
    y, lab = _t(tr["y"], dev, torch.long), _t(tr["labeled"], dev, torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    steps = epochs * math.ceil(len(X) / bs); sch = _sched(opt, steps, max(1, steps // 20))
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); sm = sd = 0.0
        perm = torch.randperm(len(X), generator=g).to(dev)
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            with torch.autocast(dev.split(":")[0], enabled=dev.startswith("cuda")):
                pred, mk = model(X[j], d[j])
                lm = msm_loss(pred.float(), X[j], mk)
                jl = j[lab[j]] if lam_dcl > 0 else j[:0]
                ld = dcl_loss(model.embed_full(X[jl], d[jl]).float(), y[jl], d[jl]) if jl.numel() > 1 else lm * 0
                loss = lm + lam_dcl * ld
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            sm += lm.item() * len(j); sd += ld.item() * len(j)
        model.eval()
        with torch.no_grad():   # 验证集 MSM 损失 (固定掩码种子)
            torch.manual_seed(1234); vl = 0.0
            for i in range(0, len(va["X"]), 1024):
                x = _t(va["X"][i:i + 1024], dev, torch.float32); dd = _t(va["det"][i:i + 1024], dev, torch.long)
                pred, mk = model(x, dd); vl += msm_loss(pred.float(), x, mk).item() * len(x)
        hist.append(dict(epoch=ep + 1, msm=sm / len(X), dcl=sd / len(X), val_msm=vl / len(va["X"]), sec=time.time() - t0))
        log(f"      pre ep {ep + 1:3d} msm {sm / len(X):.4f} dcl {sd / len(X):.4f} val_msm {vl / len(va['X']):.4f}")
    return model, dict(history=hist)


@torch.no_grad()
def embed(enc, X, det, dev, bs=1024):
    enc.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(enc.pooled(_t(X[i:i + bs], dev, torch.float32), _t(det[i:i + bs], dev, torch.long)).float().cpu().numpy())
    return np.concatenate(out)
