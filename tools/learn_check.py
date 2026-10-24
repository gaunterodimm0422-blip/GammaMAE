# -*- coding: utf-8 -*-
"""学习性检查: 小样本上训练若干 epoch, 确认 NN 的 train/val 准确率能明显超过随机并逼近 logreg。
用法: python -m tools.learn_check --data D:\\gammamae_v2_data --n 6000 --epochs 15"""
import argparse, time, numpy as np, torch
from gmae2 import config as C
from gmae2.data import load, subset, Standardizer
from gmae2 import baselines as BL
from gmae2.models import CNN1D, TransformerClassifier, GammaMAE
from gmae2.nn_train import train_classifier, predict, pretrain_gmae

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--methods", default="logreg,cnn,transformer,gmae")
a = ap.parse_args()
rng = np.random.default_rng(0)
tr, va = load(a.data, "train"), load(a.data, "val")
tr = subset(tr, rng.choice(len(tr["y"]), min(a.n, len(tr["y"])), replace=False))
va = subset(va, rng.choice(len(va["y"]), min(a.n // 3, len(va["y"])), replace=False))
std = Standardizer().fit(tr["x"])
D = lambda d: dict(X=std(d["x"]), y=d["y"], det=d["det"])
print(f"train {len(tr['y'])} val {len(va['y'])} chance {1/len(C.NUCLIDES):.3f} device {a.device}")
res = {}
for m in a.methods.split(","):
    t0 = time.time()
    if m == "logreg":
        oh = lambda d: np.eye(len(C.DETECTORS), dtype=np.float32)[d["det"]]
        f = lambda d: np.hstack([std(d["x"]), oh(d)])
        clf, hp = BL.logreg(f(tr), tr["y"], f(va), va["y"], seed=0, log=lambda s: None)
        res[m] = ((clf.predict(f(tr)) == tr["y"]).mean(), (clf.predict(f(va)) == va["y"]).mean())
    else:
        if m == "gmae":
            g = GammaMAE(C.N_BINS)
            g, _ = pretrain_gmae(g, dict(X=std(tr["x"]), det=tr["det"], y=tr["y"], labeled=np.ones(len(tr["y"]), bool)),
                                 dict(X=std(va["x"]), det=va["det"]), a.device, epochs=a.epochs, bs=a.bs, log=print)
            model = TransformerClassifier(C.N_BINS, len(C.NUCLIDES), encoder=g.enc)
            model, h = train_classifier(model, D(tr), D(va), a.device, epochs=a.epochs, lr=3e-4, bs=a.bs, patience=100,
                                        log=print, lr_encoder_scale=0.5)
        else:
            model = {"cnn": CNN1D, "transformer": TransformerClassifier}[m](C.N_BINS, len(C.NUCLIDES))
            model, h = train_classifier(model, D(tr), D(va), a.device, epochs=a.epochs, bs=a.bs, patience=100, log=print)
        res[m] = ((predict(model, std(tr["x"]), tr["det"], a.device) == tr["y"]).mean(),
                  (predict(model, std(va["x"]), va["det"], a.device) == va["y"]).mean())
    print(f">>> {m}: train acc {res[m][0]:.3f}  val acc {res[m][1]:.3f}  ({time.time()-t0:.0f} s)", flush=True)
print("\n汇总:"); [print(f"  {k:12s} train {v[0]:.3f} val {v[1]:.3f}") for k, v in res.items()]
