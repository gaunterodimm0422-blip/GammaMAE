# -*- coding: utf-8 -*-
"""results.csv -> 汇总表 (Markdown + CSV)。
准确率 = 各 (探测器 x 测量时间) 单元准确率的平均 (扰动集与 clean 集窗口数不同, 单元平均可直接比较);
± 为训练种子间标准差 (确定性方法只有一次运行, 不给 ±)。另给 95% 二项置信区间的半宽 (按总样本数)。
用法: python -m gmae2.summarize --out D:\\gammamae_v2_results
"""
import argparse, os
import numpy as np, pandas as pd

ORDER = ["template", "logreg", "knn", "cnn", "resnet", "transformer", "gmae_msm", "gmae_msm_probe", "gmae_dcl", "gmae_dcl_probe"]


def cell_acc(df, by):
    keys = list(dict.fromkeys(by + ["seed", "det", "t"]))     # 去重 (by 里可能已有 det/t)
    g = df.groupby(keys, as_index=False)[["n", "correct"]].sum()
    g["acc"] = g.correct / g.n
    s = g.groupby(by + ["seed"], as_index=False).agg(acc=("acc", "mean"), n=("n", "sum"))
    return s.groupby(by, as_index=False).agg(mean=("acc", "mean"), std=("acc", "std"), runs=("acc", "size"), n=("n", "mean"))


def fmt(r):
    s = f"{100 * r['mean']:.1f}"
    if r["runs"] > 1:
        s += f" ± {100 * r['std']:.1f}"
    return s


def pivot(t, index, cols):
    t = t.copy(); t["v"] = t.apply(fmt, axis=1)
    p = t.pivot_table(index=index, columns=cols, values="v", aggfunc="first")
    if index == "method":
        p = p.reindex([m for m in ORDER if m in p.index])
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.out, "results.csv")).drop_duplicates(["exp", "method", "seed", "testset", "det", "t", "dist"], keep="last")
    md = []
    e1 = df[df.exp == "e1"]
    if len(e1):
        c = e1[e1.testset == "clean"]
        md += ["## E1 同探测器分类 (clean test, %)", pivot(cell_acc(c, ["method", "det"]), "method", "det").to_markdown(), ""]
        g = c.groupby(["method", "seed", "t"], as_index=False)[["n", "correct"]].sum(); g["acc"] = g.correct / g.n
        t = g.groupby(["method", "t"], as_index=False).agg(mean=("acc", "mean"), std=("acc", "std"), runs=("acc", "size"))
        md += ["## E1 按测量时间 (三探测器合并, %)", pivot(t, "method", "t").to_markdown(), ""]
        g = c.groupby(["method", "seed", "det", "t"], as_index=False)[["n", "correct"]].sum(); g["acc"] = g.correct / g.n
        t = g.groupby(["method", "det", "t"], as_index=False).agg(mean=("acc", "mean"), std=("acc", "std"), runs=("acc", "size"))
        md += ["## E1 按探测器 x 测量时间 (%)", pivot(t, "method", ["det", "t"]).to_markdown(), ""]
        md += ["## E1 按源距离 (单元平均, %)", pivot(cell_acc(c, ["method", "dist"]), "method", "dist").to_markdown(), ""]
        md += ["## E3 扰动鲁棒性 (单元平均, %)", pivot(cell_acc(e1, ["method", "testset"]), "method", "testset").to_markdown(), ""]
    lo = df[df.exp.str.startswith("lodo")]
    if len(lo):
        lo = lo.assign(target=lo.exp.str.split("_").str[1], k=lo.exp.str.split("_").str[2])
        md += ["## E2 留一探测器迁移 (目标探测器 test, %)", pivot(cell_acc(lo, ["method", "target", "k"]), "method", ["target", "k"]).to_markdown(), ""]
    di = df[df.exp == "dist10"]
    if len(di):
        g = di.groupby(["method", "seed", "dist", "det", "t"], as_index=False)[["n", "correct"]].sum(); g["acc"] = g.correct / g.n
        g = g.groupby(["method", "seed", "dist"], as_index=False).acc.mean()
        t = g.groupby(["method", "dist"], as_index=False).agg(mean=("acc", "mean"), std=("acc", "std"), runs=("acc", "size"))
        md += ["## E4 距离迁移: 只用 10 cm 训练 (%)", pivot(t, "method", "dist").to_markdown(), ""]
    lw = df[df.exp.str.startswith("low")]
    if len(lw):
        g = lw.groupby(["method", "seed", "det", "t"], as_index=False)[["n", "correct"]].sum(); g["acc"] = g.correct / g.n
        g = g.groupby(["method", "seed", "t"], as_index=False).acc.mean()      # 先在探测器间平均, ± 只反映训练种子
        t = g.groupby(["method", "t"], as_index=False).agg(mean=("acc", "mean"), std=("acc", "std"), runs=("acc", "size"))
        md += [f"## E5 少标注 ({lw.exp.iloc[0]}: 每单元标注数见名称; 三探测器平均, %)", pivot(t, "method", "t").to_markdown(), ""]
        md += ["## E5 少标注 (单元平均, %)", pivot(cell_acc(lw, ["method"]).assign(k="all"), "method", "k").to_markdown(), ""]
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
