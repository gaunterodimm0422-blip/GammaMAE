# -*- coding: utf-8 -*-
"""事件列表 (convert_to_npz.py 输出) -> 能谱数据集。

每个源 job (探测器 x 核素 x 距离 x 种子) 的 600 s 事件按真实时间切成不重叠窗口
(1/3/10/30/100 s)，逐事件高斯展宽，按物理注量率从同一种子的本底事件池泊松抽样混入。
种子 0/1/2 -> train/val/test，三者不共享任何事件。测试集另生成事件层面的扰动版本
(增益、分辨率、本底水平)，窗口与 clean 测试集一一对应 (配对比较)。

用法:
  python -m gmae2.build_dataset --events D:\\gammamae_v2_npz --out D:\\gammamae_v2_data
"""
import argparse, glob, hashlib, json, os, time
import numpy as np
from . import config as C


def _rng(*keys):
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def load_job(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z["edep_keV"].astype(np.float64), z["time_s"], z["sourceID"], meta


def measure(det, e, rng, res=1.0, gain=0.0):
    """沉积能量 -> 测量能量 (高斯展宽 + 增益)；返回道址 (超出范围的丢弃)"""
    sig = C.fwhm(det, np.maximum(e, 0.0), res) / 2.3548
    em = (e + rng.normal(0.0, 1.0, e.size) * sig) * (1.0 + gain)
    em = em[em >= C.E_THRESHOLD]
    ch = (em / C.BIN_W).astype(np.int64)
    return ch[ch < C.N_BINS]


class BackgroundPool:
    def __init__(self, path, det):
        e, _, sid, meta = load_job(path)
        cont = max(l["sourceID"] for l in meta["lines"])        # 连续谱 sourceID
        ks = meta["kappa_sim"]; T = meta["t_meas_s"]
        sim_l = meta["entering_per_s"] / (1 + ks); sim_c = meta["entering_per_s"] * ks / (1 + ks)
        phys = C.PHI_LINES * C.surface_cm2(det) / 4.0
        self.e_l, self.e_c = e[sid != cont], e[sid == cont]
        self.rate_l = self.e_l.size / T * phys / sim_l               # 探测到的本底主线事件 /s
        self.rate_c = self.e_c.size / T * C.KAPPA * phys / sim_c     # 探测到的散射连续谱事件 /s
        self.det = det

    def draw(self, t, rng, mult=1.0):
        nl = rng.poisson(self.rate_l * t * mult); nc = rng.poisson(self.rate_c * t * mult)
        return np.concatenate([self.e_l[rng.integers(0, self.e_l.size, nl)],
                               self.e_c[rng.integers(0, self.e_c.size, nc)]])


BKG_TPL = np.zeros((len(C.DETECTORS), C.N_BINS))


def build_split(events_dir, split, variants, act_scale=1.0, quiet=False):
    """act_scale = 源活度 / 模拟活度 (37 kBq)。对源事件做独立伯努利稀疏 (泊松过程稀疏后仍是泊松过程,
    等价于活度按比例降低)；稀疏掩码只依赖 (split, 探测器, 核素, 距离)，clean 与各扰动版本共用，保持配对。"""
    rep = C.SPLIT_SEED[split]
    out = {v: dict(counts=[], det=[], nuc=[], dist=[], t=[], win=[], n_src=[], n_bkg=[]) for v in variants}
    for di, det in enumerate(C.DETECTORS):
        bp = os.path.join(events_dir, f"{det}_BKG_s{rep}.npz")
        if not os.path.exists(bp):
            print(f"  !! 缺少本底 {bp}, 跳过 {det}"); continue
        pool = BackgroundPool(bp, det)
        if not quiet: print(f"  {det}: 本底 {pool.rate_l + pool.rate_c:.1f} cps (主线 {pool.rate_l:.1f}, 散射 {pool.rate_c:.1f})")
        if split == "train":   # 名义本底模板 (供模板拟合基线), 来自 train 种子
            r = _rng("bkgtpl", det)
            BKG_TPL[di] = np.bincount(measure(det, pool.draw(3000.0, r), r), minlength=C.N_BINS)
        for ni, nuc in enumerate(C.NUCLIDES):
            for dist in C.DISTANCES:
                jp = os.path.join(events_dir, f"{det}_{nuc}_d{dist:02d}_s{rep}.npz")
                if not os.path.exists(jp):
                    print(f"  !! 缺少 {os.path.basename(jp)}"); continue
                e, tt, sid, meta = load_job(jp)
                if act_scale < 1.0:
                    keep = _rng("thin", split, det, nuc, dist).random(e.size) < act_scale
                    e, tt = e[keep], tt[keep]
                for v, pv in variants.items():
                    res, gain, bm = pv.get("res", 1.0), pv.get("gain", 0.0), pv.get("bkg", 1.0)
                    maxw = C.MAX_WINDOWS if v == "clean" else C.MAX_WINDOWS_PERTURBED
                    rng = _rng(split, det, nuc, dist, v)
                    for t in C.TIMES:
                        nw = int(C.T_MEAS // t)
                        wins = np.sort(_rng("win", split, det, nuc, dist, t).permutation(nw)[:maxw[t]])
                        lo = np.searchsorted(tt, wins * t); hi = np.searchsorted(tt, (wins + 1) * t)
                        for w, a, b in zip(wins, lo, hi):
                            chs = measure(det, e[a:b], rng, res, gain)
                            chb = measure(det, pool.draw(t, rng, bm), rng, res, gain)
                            o = out[v]
                            o["counts"].append(np.bincount(np.concatenate([chs, chb]), minlength=C.N_BINS).astype(np.uint32))
                            o["det"].append(di); o["nuc"].append(ni); o["dist"].append(dist); o["t"].append(t)
                            o["win"].append(w); o["n_src"].append(chs.size); o["n_bkg"].append(chb.size)
            if not quiet: print(f"    {det} {nuc} 完成")
    return out


def save(out, path):
    arr = {k: (np.stack(v) if k == "counts" else np.asarray(v)) for k, v in out.items()}
    if arr["counts"].size == 0:
        print("  (空)"); return
    np.savez_compressed(path, **arr)
    print(f"  -> {path}  {arr['counts'].shape[0]} 条谱")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="convert_to_npz.py 的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--activity_kbq", type=float, default=C.ACTIVITY_KBQ,
                    help=f"源活度 (kBq); GATE 按 {C.SIM_ACTIVITY_KBQ} kBq 模拟, 更低的活度通过事件稀疏实现")
    a = ap.parse_args()
    act = a.activity_kbq / C.SIM_ACTIVITY_KBQ
    assert 0 < act <= 1.0, "活度不能高于模拟活度"
    print(f"源活度 {a.activity_kbq} kBq (稀疏因子 {act:.4f})")
    os.makedirs(a.out, exist_ok=True)
    for split in a.splits.split(","):
        t0 = time.time()
        variants = {"clean": {}}
        if split == "test":
            variants.update(C.PERTURBATIONS)
        print(f"== {split} (GATE seed {C.SPLIT_SEED[split]}), 版本: {list(variants)}")
        out = build_split(a.events, split, variants, act)
        for v, o in out.items():
            tag = v.replace(" ", "").replace("%", "pct").replace("+", "p").replace("-", "m")
            save(o, os.path.join(a.out, f"{split}_{tag}.npz"))
        if split == "train":
            np.save(os.path.join(a.out, "bkg_templates.npy"), BKG_TPL)
        print(f"   用时 {time.time() - t0:.0f} s")
    cfg = {k: getattr(C, k) for k in ["DETECTORS", "NUCLIDES", "DISTANCES", "N_BINS", "E_MAX", "E_THRESHOLD",
                                          "RESOLUTION", "PHI_LINES", "KAPPA", "TIMES", "MAX_WINDOWS", "PERTURBATIONS"]}
    cfg["ACTIVITY_KBQ"] = a.activity_kbq
    json.dump(cfg, open(os.path.join(a.out, "dataset_config.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
