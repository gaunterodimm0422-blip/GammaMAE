# -*- coding: utf-8 -*-
"""数据读取与特征：x = log1p(1000 p), p = counts / sum(counts)；对所有谱统一一个尺度。"""
import os
import numpy as np
from . import config as C


def tagname(v):
    return v.replace(" ", "").replace("%", "pct").replace("+", "p").replace("-", "m")


def load(data_dir, split, variant="clean"):
    z = np.load(os.path.join(data_dir, f"{split}_{tagname(variant)}.npz"))
    d = {k: z[k] for k in z.files if k != "counts"}
    c = z["counts"]
    tot = c.sum(1, keepdims=True).astype(np.float64)
    d["x"] = np.empty(c.shape, np.float32)
    for i in range(0, len(c), 4096):   # 分块, 避免整块 float64 临时数组
        d["x"][i:i + 4096] = np.log1p(1000.0 * c[i:i + 4096] / np.maximum(tot[i:i + 4096], 1))
    del c
    d["total"] = tot[:, 0]
    d["y"] = d["nuc"].astype(np.int64); d["det"] = d["det"].astype(np.int64)
    return d


def subset(d, m):
    m = np.asarray(m)
    if m.dtype != bool:                      # 整数索引 -> 布尔掩码
        b = np.zeros(len(d["y"]), bool); b[m] = True; m = b
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape[:1] else v) for k, v in d.items()}


def mask(d, det=None, dist=None, t=None):
    m = np.ones(len(d["y"]), bool)
    if det is not None: m &= np.isin(d["det"], np.atleast_1d(det))
    if dist is not None: m &= np.isin(d["dist"], np.atleast_1d(dist))
    if t is not None: m &= np.isin(d["t"], np.atleast_1d(t))
    return m


def concat(*ds):
    keys = [k for k in ds[0] if isinstance(ds[0][k], np.ndarray)]
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


def few_shot(d, det, k, seed):
    """从目标探测器 train 中每类随机取 k 条 (跨测量时间/距离随机)。"""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(C.NUCLIDES)):
        cand = np.where((d["det"] == det) & (d["y"] == c))[0]
        idx += list(rng.choice(cand, min(k, cand.size), replace=False))
    m = np.zeros(len(d["y"]), bool); m[idx] = True
    return m


class Standardizer:
    """逐道 z-score；std 下限避免高能空道放大噪声。"""
    def fit(self, x):
        self.mu = x.mean(0); s = x.std(0)
        self.sd = np.maximum(s, 0.1 * s.mean() + 1e-6)
        return self

    def __call__(self, x):
        return ((x - self.mu) / self.sd).astype(np.float32)


def counts_of(d, idx=slice(None)):
    """由 x 与总计数精确还原整数计数 (只存一份特征以省内存)。"""
    return np.rint(np.expm1(d["x"][idx].astype(np.float64)) * d["total"][idx, None] / 1000.0).astype(np.float32)
