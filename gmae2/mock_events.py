# -*- coding: utf-8 -*-
"""仅供流程测试: 生成与 convert_to_npz.py 输出同格式的假事件列表 (物理很粗糙)。"""
import argparse, glob, json, os
import numpy as np

def fake_response(E, det, rng):
    n = E.size
    pint = 1 - np.exp(-{"NaI": 3.0, "GAGG": 1.5, "HPGe": 2.0}[det] * (E / 300.0) ** -0.8)
    keep = rng.random(n) < pint
    E = E[keep]
    pt = np.clip({"NaI": 0.6, "GAGG": 0.35, "HPGe": 0.3}[det] * (E / 662.0) ** -0.5, 0, 1)
    fep = rng.random(E.size) < pt
    ec = E * (1 - 1 / (1 + 2 * E / 511.0))
    dep = np.where(fep, E, rng.random(E.size) * ec)
    return dep, keep

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.05); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for mf in sorted(glob.glob(os.path.join(a.pkg, "jobs", "*", "meta.json"))):
        m = json.load(open(mf)); rng = np.random.default_rng(abs(hash(m["job"])) % 2**32)
        T = m["t_meas_s"]
        if m["kind"] == "source":
            L = m["lines"]; w = np.array([l["intensity"] for l in L]); en = np.array([l["energy_keV"] for l in L])
            n = rng.poisson(m["entering_per_s"] * T * a.scale)
            sid = rng.choice(len(L), n, p=w / w.sum()); E = en[sid]
        else:
            L = [l for l in m["lines"] if l["energy_keV"] is not None]
            w = np.array([l["flux_weight"] for l in L]); en = np.array([l["energy_keV"] for l in L])
            n = rng.poisson(m["entering_per_s"] * T * a.scale); ks = m["kappa_sim"]
            cont = rng.random(n) < ks / (1 + ks)
            sid = np.where(cont, len(L), rng.choice(len(L), n, p=w / w.sum()))
            E = np.where(cont, 30 + rng.exponential(250, n), en[np.minimum(sid, len(L) - 1)])
        dep, keep = fake_response(E, m["detector"], rng); sid = sid[keep]
        t = np.sort(rng.uniform(0, T, dep.size))
        m["entering_per_s"] *= a.scale
        np.savez_compressed(os.path.join(a.out, m["job"] + ".npz"), edep_keV=dep.astype(np.float32), time_s=t,
                            sourceID=sid.astype(np.int16), meta=json.dumps(m))
    print("mock done")

if __name__ == "__main__":
    main()
