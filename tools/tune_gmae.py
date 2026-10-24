# -*- coding: utf-8 -*-
"""GammaMAE 预训练/微调设置的验证集消融 (只用 train 种子0 / val 种子1，不碰 test)。
E1 设置：三探测器全部 train 谱预训练 (MSM 无标注；DCL 用标注)，然后在全部 train 标注上微调，
另做"少标注"微调：每 (探测器, 核素, 距离, 时间) 只给少量标注，检验预训练的价值。
用法: python -u -m tools.tune_gmae --data D:\\gammamae_v2_data_1kBq *> D:\\gammamae_v2_tune.log"""
import argparse, copy, time, numpy as np, torch
from gmae2 import config as C
from gmae2.data import load, subset, Standardizer
from gmae2.models import CNN1D, TransformerClassifier, GammaMAE
from gmae2.nn_train import train_classifier, predict, pretrain_gmae

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pre_epochs", type=int, default=30); ap.add_argument("--ft_epochs", type=int, default=20)
ap.add_argument("--epochs", type=int, default=40, help="从零训练的 CNN/Transformer")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--low", type=int, default=5, help="少标注设置: 每个 (det,nuc,dist,t) 单元的标注条数")
ap.add_argument("--quick", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.device
tr, va = load(a.data, "train"), load(a.data, "val")
if a.quick:
    r = np.random.default_rng(0)
    tr = subset(tr, r.choice(len(tr["y"]), a.quick, replace=False)); va = subset(va, r.choice(len(va["y"]), a.quick // 2, replace=False))
std = Standardizer().fit(tr["x"])
D = lambda d: dict(X=std(d["x"]), y=d["y"], det=d["det"])
TR, VA = D(tr), D(va)
# 少标注子集
r = np.random.default_rng(7); low = np.zeros(len(tr["y"]), bool)
key = np.stack([tr["det"], tr["y"], tr["dist"], tr["t"]], 1)
for k in np.unique(key, axis=0):
    idx = np.where((key == k).all(1))[0]; low[r.choice(idx, min(a.low, idx.size), replace=False)] = True
TRL = {k: v[low] for k, v in TR.items()}
print(f"device {dev}  train {len(tr['y'])} (少标注 {low.sum()})  val {len(va['y'])}", flush=True)
res = []


def report(name, model, secs):
    p = predict(model, VA["X"], VA["det"], dev); ok = p == VA["y"]
    by_t = " ".join(f"{t}s:{100 * ok[va['t'] == t].mean():.1f}" for t in C.TIMES)
    by_d = " ".join(f"{C.DETECTORS[i]}:{100 * ok[va['det'] == i].mean():.1f}" for i in range(3))
    line = f"{name:34s} val {100 * ok.mean():5.1f} | {by_t} | {by_d} | {secs / 60:.1f} min"
    print(">>> " + line, flush=True); res.append(line)


quiet = lambda s: None
for lab_name, TRx in (("全标注", TR), (f"少标注{a.low}", TRL)):
    for name, M, lr in (("cnn", CNN1D, 1e-3), ("transformer", TransformerClassifier, 1e-3)):
        t0 = time.time(); m = M(C.N_BINS, len(C.NUCLIDES))
        m, _ = train_classifier(m, TRx, VA, dev, epochs=a.epochs, lr=lr, bs=a.bs, patience=10, log=quiet)
        report(f"{lab_name} {name} (从零)", m, time.time() - t0)

FULL, LOW = "全标注", f"少标注{a.low}"
# (名称, lambda_DCL, 掩码率, DCL 可用的标注, 用于哪些微调设置)；DCL 在少标注设置下只能看少标注子集
PRE = [("msm", 0.0, 0.75, None, (FULL, LOW)), ("msm_m50", 0.0, 0.5, None, (FULL, LOW)),
       ("msm+dcl", 0.5, 0.75, np.ones(len(tr["y"]), bool), (FULL,)), ("msm+dcl", 0.5, 0.75, low, (LOW,))]
SETS = {FULL: TR, LOW: TRL}
for pname, lam, mask, lab, uses in PRE:
    t0 = time.time(); g = GammaMAE(C.N_BINS, mask_ratio=mask)
    lab = np.ones(len(tr["y"]), bool) if lab is None else lab
    g, info = pretrain_gmae(g, dict(X=TR["X"], det=TR["det"], y=TR["y"], labeled=lab), dict(X=VA["X"], det=VA["det"]), dev,
                            epochs=a.pre_epochs, bs=a.bs, lam_dcl=lam, log=quiet)
    h = info["history"][-1]; tp = time.time() - t0
    print(f"    预训练 {pname} (DCL 标注 {int(lab.sum()) if lam > 0 else 0}): msm {h['msm']:.4f} dcl {h['dcl']:.4f} "
          f"val_msm {h['val_msm']:.4f} ({tp / 60:.1f} min)", flush=True)
    for lab_name in uses:
        for ft_lr, scale in ((3e-4, 0.5), (1e-3, 0.3)):
            t0 = time.time()
            m = TransformerClassifier(C.N_BINS, len(C.NUCLIDES), encoder=copy.deepcopy(g.enc))
            m, _ = train_classifier(m, SETS[lab_name], VA, dev, epochs=a.ft_epochs, lr=ft_lr, bs=a.bs, patience=8, log=quiet,
                                    lr_encoder_scale=scale)
            report(f"{lab_name} gmae[{pname}] ft lr{ft_lr:g}", m, time.time() - t0)
print("\n汇总 (val 准确率 %, 按测量时间 / 探测器):"); [print("  " + l) for l in res]
open("tune_gmae.txt", "w", encoding="utf-8").write("\n".join(res))
