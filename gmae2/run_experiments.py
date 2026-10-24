# -*- coding: utf-8 -*-
"""GammaMAE v2 实验。

  e1   同探测器分类: 三个探测器的 train(种子0) 联合训练, val(种子1) 选超参/早停,
       test(种子2) 的 clean 与 8 种事件层面扰动 (增益/分辨率/本底) 上评估。
  lodo 留一探测器迁移: 用另外两个探测器的标注训练, 在目标探测器 test 上评估;
       k=0 (零样本) 与 k-shot (目标 train 中每类 k 条标注)。GammaMAE 预训练时 MSM 可见目标探测器
       的无标注 train 谱, DCL 只用源探测器标注; 验证集只用源探测器 (不碰目标标注)。
  dist 距离迁移: 只用 10 cm 标注训练, 在 5/10/25 cm test 上评估。
每个 (实验, 方法, 种子, 测试集) 的逐谱预测存入 preds/, 汇总计数写入 results.csv; 中断后重跑会跳过已完成项。

用法: python -m gmae2.run_experiments --data D:\\gammamae_v2_data --out D:\\gammamae_v2_results --device cuda
"""
import argparse, csv, json, os, time
import numpy as np, torch
from . import config as C
from .data import load, subset, mask, concat, few_shot, Standardizer, counts_of
from . import baselines as BL
from .models import CNN1D, ResNet1D, TransformerClassifier, GammaMAE, n_params
from .nn_train import train_classifier, predict, pretrain_gmae, embed

NN = {"cnn": CNN1D, "resnet": ResNet1D, "transformer": TransformerClassifier}


class Runner:
    def __init__(self, a):
        self.a, self.dev = a, a.device
        os.makedirs(os.path.join(a.out, "preds"), exist_ok=True); os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
        os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
        self.csv = os.path.join(a.out, "results.csv")
        self.logf = open(os.path.join(a.out, "run.log"), "a", encoding="utf-8")
        self.E = dict(epochs=a.epochs, pre_epochs=a.pre_epochs, ft_epochs=a.ft_epochs)

    def log(self, s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"; print(line, flush=True); self.logf.write(line + "\n"); self.logf.flush()

    def done(self, key):
        return os.path.exists(os.path.join(self.a.out, "preds", key + ".npz"))

    def record(self, key, exp, method, seed, testset, te, pred, info):
        np.savez_compressed(os.path.join(self.a.out, "preds", key + ".npz"), pred=pred.astype(np.int8), y=te["y"].astype(np.int8))
        json.dump(info, open(os.path.join(self.a.out, "logs", key + ".json"), "w"), indent=1, default=float)
        new = not os.path.exists(self.csv)
        with open(self.csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["exp", "method", "seed", "testset", "det", "t", "dist", "n", "correct"])
            ok = pred == te["y"]
            for di in np.unique(te["det"]):
                for t in np.unique(te["t"]):
                    for r in np.unique(te["dist"]):
                        m = (te["det"] == di) & (te["t"] == t) & (te["dist"] == r)
                        if m.any():
                            w.writerow([exp, method, seed, testset, C.DETECTORS[di], int(t), int(r), int(m.sum()), int(ok[m].sum())])
        self.log(f"  {key}: acc {ok.mean():.4f}")

    # ---------------- 通用：训练一个方法, 在若干测试集上预测 ----------------
    def run_method(self, exp, method, seed, tr, va, tests, counts_tr=None, pre=None):
        keys = {name: f"{exp}__{method}__s{seed}__{name}" for name in tests}
        if all(self.done(k) for k in keys.values()):
            return
        self.log(f"== {exp} {method} seed {seed}  train {len(tr['y'])} val {len(va['y'])}")
        std = self.std   # 统一在全部 train 谱 (无标注信息) 上拟合, 所有方法相同
        info = dict(method=method, seed=seed, n_train=int(len(tr["y"])))
        oh = lambda d: np.eye(len(C.DETECTORS), dtype=np.float32)[d["det"]]
        if method == "logreg":
            f = lambda d: np.hstack([std(d["x"]), oh(d)])
            m, hp = BL.logreg(f(tr), tr["y"], f(va), va["y"], seed=seed, log=self.log); info.update(hp)
            P = {n: m.predict(f(t)) for n, t in tests.items()}
        elif method == "knn":
            k, hp = BL.knn(tr["x"], tr["y"], va["x"], va["y"], dev=self.dev, log=self.log); info.update(hp)
            P = {n: BL.knn_predict(tr["x"], tr["y"], t["x"], k, self.dev) for n, t in tests.items()}
        elif method == "template":
            tf = BL.TemplateFit(lambda: counts_of(tr), tr["y"], tr["det"], tr["dist"], np.load(os.path.join(self.a.data, "bkg_templates.npy")))
            P = {n: tf.predict(lambda idx, t=t: counts_of(t, idx), t["det"], self.dev) for n, t in tests.items()}
        elif method in NN:
            model = NN[method](C.N_BINS, len(C.NUCLIDES))
            info["params"] = n_params(model)
            model, h = train_classifier(model, dict(X=std(tr["x"]), y=tr["y"], det=tr["det"]),
                                        dict(X=std(va["x"]), y=va["y"], det=va["det"]), self.dev,
                                        epochs=self.E["epochs"], lr=self.a.lr, bs=self.a.bs, seed=seed, log=self.log)
            info.update(h)
            P = {n: predict(model, std(t["x"]), t["det"], self.dev) for n, t in tests.items()}
        elif method.startswith("gmae"):
            enc, pinfo = pre
            info["pretrain"] = pinfo
            if method.endswith("_probe"):          # 冻结编码器 + 逻辑回归 (线性探针)
                Z = lambda d: embed(enc, std(d["x"]), d["det"], self.dev)
                zs = Standardizer().fit(Z(tr)); f = lambda d: zs(Z(d))
                m, hp = BL.logreg(f(tr), tr["y"], f(va), va["y"], seed=seed, log=self.log); info.update(hp)
                P = {n: m.predict(f(t)) for n, t in tests.items()}
            else:                                   # 微调
                import copy
                model = TransformerClassifier(C.N_BINS, len(C.NUCLIDES), encoder=copy.deepcopy(enc))
                model, h = train_classifier(model, dict(X=std(tr["x"]), y=tr["y"], det=tr["det"]),
                                            dict(X=std(va["x"]), y=va["y"], det=va["det"]), self.dev,
                                            epochs=self.E["ft_epochs"], lr=self.a.ft_lr, bs=self.a.bs, patience=8, seed=seed,
                                            log=self.log, lr_encoder_scale=self.a.ft_enc_scale)
                info.update(h)
                P = {n: predict(model, std(t["x"]), t["det"], self.dev) for n, t in tests.items()}
        for n, t in tests.items():
            self.record(keys[n], exp, method, seed, n, t, P[n], info)

    def pretrain(self, tag, seed, unl, lab_mask, va, lam):
        """unl: 预训练用的全部 train 谱; lab_mask: 哪些可用于 DCL。结果缓存到 models/。"""
        path = os.path.join(self.a.out, "models", f"{tag}_s{seed}.pt")
        std = self.std
        g = GammaMAE(C.N_BINS, mask_ratio=self.a.mask)
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.dev, weights_only=False)
            g.load_state_dict(ck["state"]); g.to(self.dev); return g.enc, ck["info"]
        self.log(f"== 预训练 {tag} seed {seed}: {len(unl['y'])} 条谱, DCL 标注 {int(lab_mask.sum())}, lambda={lam}")
        g, info = pretrain_gmae(g, dict(X=std(unl["x"]), det=unl["det"], y=unl["y"], labeled=lab_mask),
                                dict(X=std(va["x"]), det=va["det"]), self.dev, epochs=self.E["pre_epochs"],
                                lr=self.a.pre_lr, bs=self.a.bs, lam_dcl=lam, seed=seed, log=self.log)
        info["params"] = n_params(g); info["encoder_params"] = n_params(g.enc)
        torch.save(dict(state=g.state_dict(), info=info), path)
        return g.enc, info


def methods_list(a):
    return a.methods.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="e1,lodo,dist,low"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="logreg,knn,template,cnn,resnet,transformer,gmae_msm,gmae_msm_probe,gmae_dcl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pre_epochs", type=int, default=40); ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--ft_epochs", type=int, default=30); ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--ft_enc_scale", type=float, default=0.3, help="微调时编码器学习率 = ft_lr x 该值")
    ap.add_argument("--mask", type=float, default=0.5, help="MSM 掩码率 (验证集消融: 0.5 优于 0.75)")
    ap.add_argument("--low", type=int, default=5, help="low 实验: 每个 (探测器,核素,距离,时间) 单元的标注条数")
    ap.add_argument("--lam_dcl", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--kshot", default="0,10")
    ap.add_argument("--quick", type=int, default=0, help="冒烟测试: 每个集合随机取这么多条")
    a = ap.parse_args()
    R = Runner(a); M = methods_list(a); seeds = [int(s) for s in a.seeds.split(",")]
    R.log(f"device={a.device} exp={a.exp} methods={M} seeds={seeds}")

    def q(d, s=0):
        if not a.quick or len(d["y"]) <= a.quick:
            return d
        return subset(d, np.isin(np.arange(len(d["y"])), np.random.default_rng(s).choice(len(d["y"]), a.quick, replace=False)))

    tr_all, va_all = q(load(a.data, "train")), q(load(a.data, "val"))
    R.std = Standardizer().fit(tr_all["x"])
    te_clean = q(load(a.data, "test"))
    det_seed = lambda m: [0] if m in ("logreg", "knn", "template") else seeds   # 确定性方法只跑一次

    ones = lambda d: np.ones(len(d["y"]), bool)

    def get_pre(m, s, dcl_tag, dcl_mask, va):
        """gmae_msm*: 一个共享的 MSM 编码器 (全部 train 谱, 无标注, 各实验共用);
        gmae_dcl*: MSM+DCL, DCL 只用该实验允许的标注 (dcl_mask)。"""
        if not m.startswith("gmae"):
            return None
        if m.startswith("gmae_msm"):
            return R.pretrain("msm", s, tr_all, ones(tr_all) & False, va_all, 0.0)
        return R.pretrain(dcl_tag, s, tr_all, dcl_mask, va, a.lam_dcl)

    if "e1" in a.exp:
        tests = {"clean": te_clean}
        for v in C.PERTURBATIONS:
            tests[v] = q(load(a.data, "test", v))
        for m in M:
            for s in det_seed(m):
                R.run_method("e1", m, s, tr_all, va_all, tests, pre=get_pre(m, s, "e1_dcl", ones(tr_all), va_all))
        del tests

    if "lodo" in a.exp:
        for ti, tgt in enumerate(C.DETECTORS):
            src = mask(tr_all, det=[i for i in range(len(C.DETECTORS)) if i != ti])
            va = subset(va_all, mask(va_all, det=[i for i in range(len(C.DETECTORS)) if i != ti]))
            te = {"target": subset(te_clean, mask(te_clean, det=ti))}
            for k in [int(x) for x in a.kshot.split(",")]:
                for m in [m for m in M if m != "template"]:
                    for s in det_seed(m) if k == 0 else seeds:
                        fs = few_shot(tr_all, ti, k, seed=100 + s) if k > 0 else np.zeros(len(tr_all["y"]), bool)
                        tr = subset(tr_all, src | fs)
                        R.run_method(f"lodo_{tgt}_k{k}", m, s, tr, va, te, pre=get_pre(m, s, f"lodo_{tgt}_dcl", src, va))

    if "dist" in a.exp:
        d10 = mask(tr_all, dist=10)
        tr = subset(tr_all, d10); va = subset(va_all, mask(va_all, dist=10))
        for m in M:
            for s in det_seed(m):
                R.run_method("dist10", m, s, tr, va, {"clean": te_clean}, pre=get_pre(m, s, "dist10_dcl", d10, va))

    if "low" in a.exp:   # 少标注: 每单元 a.low 条标注 (train 与 val 同样预算); MSM 编码器仍用全部无标注 train 谱
        def budget(d, seed):
            r = np.random.default_rng(seed); b = np.zeros(len(d["y"]), bool)
            key = np.stack([d["det"], d["y"], d["dist"], d["t"]], 1)
            for kk in np.unique(key, axis=0):
                idx = np.where((key == kk).all(1))[0]; b[r.choice(idx, min(a.low, idx.size), replace=False)] = True
            return b
        lt, lv = budget(tr_all, 7), budget(va_all, 8)
        tr, va = subset(tr_all, lt), subset(va_all, lv)
        R.log(f"low: train 标注 {lt.sum()}, val 标注 {lv.sum()}")
        for m in [m for m in M if not m.startswith("gmae_dcl")]:   # DCL 在少标注下坍缩 (验证集消融), 不跑
            for s in det_seed(m):
                R.run_method(f"low{a.low}", m, s, tr, va, {"clean": te_clean}, pre=get_pre(m, s, None, None, va))
    R.log("全部完成")


if __name__ == "__main__":
    main()
