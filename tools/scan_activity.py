# -*- coding: utf-8 -*-
"""源活度扫描：对若干活度 (kBq) 只建 clean train/test，用泊松模板拟合与逻辑回归看 探测器 x 测量时间 的准确率，
用来选一个使 1 s 明显困难、100 s 接近饱和的源活度。不写任何数据集文件。
用法: python -m tools.scan_activity --events D:\\gammamae_v2_npz --acts 37,10,3,1,0.3"""
import argparse, time, numpy as np, pandas as pd, torch
from gmae2 import config as C
from gmae2.build_dataset import build_split, BKG_TPL
from gmae2.data import Standardizer
from gmae2 import baselines as BL
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--events", required=True); ap.add_argument("--acts", default="37,10,3,1,0.3")
ap.add_argument("--n_lr", type=int, default=10000)
ap.add_argument("--n_test", type=int, default=0, help="只评估这么多条 test 谱 (0 = 全部)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()


def arr(o):
    d = {k: (np.stack(v) if k == "counts" else np.asarray(v)) for k, v in o.items()}
    tot = d["counts"].sum(1, keepdims=True).astype(np.float64)
    d["x"] = np.log1p(1000.0 * d["counts"] / np.maximum(tot, 1)).astype(np.float32)
    return d

rows, cnt = [], []
for act in [float(s) for s in a.acts.split(",")]:
    t0 = time.time(); f = act / C.SIM_ACTIVITY_KBQ
    tr = arr(build_split(a.events, "train", {"clean": {}}, f, quiet=True)["clean"])
    te = arr(build_split(a.events, "test", {"clean": {}}, f, quiet=True)["clean"])
    if a.n_test:
        j = np.sort(np.random.default_rng(1).choice(len(te["nuc"]), a.n_test, replace=False))
        te = {k: v[j] for k, v in te.items()}
    tf = BL.TemplateFit(tr["counts"], tr["nuc"], tr["det"], tr["dist"], BKG_TPL.copy())
    p_tf = tf.predict(te["counts"], te["det"], a.device)
    std = Standardizer().fit(tr["x"]); oh = lambda d: np.eye(3, dtype=np.float32)[d["det"]]
    i = np.random.default_rng(0).choice(len(tr["nuc"]), min(a.n_lr, len(tr["nuc"])), replace=False)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(np.hstack([std(tr["x"][i]), oh(tr)[i]]), tr["nuc"][i])
    p_lr = lr.predict(np.hstack([std(te["x"]), oh(te)]))
    for m, p in (("template", p_tf), ("logreg", p_lr)):
        df = pd.DataFrame(dict(det=te["det"], t=te["t"], ok=p == te["nuc"]))
        for (di, t), g in df.groupby(["det", "t"]):
            rows.append(dict(act=act, method=m, det=C.DETECTORS[di], t=t, acc=100 * g.ok.mean()))
    for di in range(3):
        for dist in C.DISTANCES:
            m = (te["det"] == di) & (te["t"] == 1) & (te["dist"] == dist)
            cnt.append(dict(act=act, det=C.DETECTORS[di], dist=dist, src_cps=te["n_src"][m].mean(), bkg_cps=te["n_bkg"][m].mean()))
    print(f"活度 {act} kBq 完成 ({time.time() - t0:.0f} s)", flush=True)

R = pd.DataFrame(rows).pivot_table(index=["act", "method"], columns=["det", "t"], values="acc").round(1)
N = pd.DataFrame(cnt).pivot_table(index=["act"], columns=["det", "dist"], values=["src_cps"]).round(1)
B = pd.DataFrame(cnt).groupby("det").bkg_cps.mean().round(1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
out = ["== 准确率 (%), 行 = 活度 kBq / 方法, 列 = 探测器 / 测量时间 s", R.to_string(),
       "", "== 平均源计数率 (cps, 10 核素平均), 列 = 探测器 / 距离 cm", N.to_string(), "", "== 本底计数率 (cps)", B.to_string()]
print("\n".join(out)); open("scan_activity.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n已写入 scan_activity.txt")
