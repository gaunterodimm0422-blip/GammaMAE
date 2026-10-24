# -*- coding: utf-8 -*-
"""Windows 侧: 共享盘 GATE 结果 (events_c*.root) -> 每个 job 一个 .npz + 汇总表
用法 (远控机 PowerShell):
  python convert_to_npz.py `
      --src D:\\vgate_shared\\output\\gammamae_v2 --dst D:\\gammamae_v2_npz
只处理含 DONE 的 job; 已转换过的跳过 (加 --force 重转)。
输出 npz 字段: edep_keV(float32, 未展宽的沉积能量), time_s(float64), sourceID(int16), meta(json 字符串)
汇总 summary.csv: 每个 job 的事件数、计数率、探测/入射比、每条主要谱线 ±1 keV 全能峰计数
"""
import argparse, glob, json, os, sys
import numpy as np
try:
    import uproot
except ImportError:
    sys.exit("缺少 uproot: python.exe -m pip install uproot awkward numpy")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
rows = []
jobs = sorted(d for d in glob.glob(os.path.join(a.src, "*")) if os.path.isfile(os.path.join(d, "DONE")))
print(f"{len(jobs)} 个已完成 job")
for jd in jobs:
    job = os.path.basename(jd)
    out = os.path.join(a.dst, job + ".npz")
    meta = json.load(open(os.path.join(jd, "meta.json"), encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(jd, "events_c*.root")), key=lambda f: int(f.split("_c")[-1].split(".")[0]))
    if len(files) != meta["n_chunks"]:
        print(f"!! {job}: chunk 文件 {len(files)} != {meta['n_chunks']}, 跳过"); continue
    if os.path.exists(out) and not a.force:
        z = np.load(out, allow_pickle=False); e = z["edep_keV"]; sid = z["sourceID"]
    else:
        E, T, S = [], [], []
        for f in files:
            t = uproot.open(f)["events"].arrays(["edep_keV", "time_s", "sourceID"], library="np")
            E.append(t["edep_keV"]); T.append(t["time_s"]); S.append(t["sourceID"])
        e = np.concatenate(E).astype(np.float32); tt = np.concatenate(T); sid = np.concatenate(S).astype(np.int16)
        o = np.argsort(tt, kind="stable")
        e, tt, sid = e[o], tt[o], sid[o]
        np.savez_compressed(out, edep_keV=e, time_s=tt, sourceID=sid, meta=json.dumps(meta))
    T_ = meta["t_meas_s"]
    r = dict(job=job, kind=meta["kind"], detector=meta["detector"], nuclide=meta.get("nuclide", "BKG"),
             dist_cm=meta.get("distance_front_cm", ""), seed=meta["seed_rep"], events=len(e), cps=len(e) / T_,
             det_over_enter=len(e) / T_ / meta["entering_per_s"])
    if meta["kind"] == "source":
        main = sorted([l for l in meta["lines"] if l["type"] == "g"], key=lambda l: -l["intensity"])[:3]
        for k, l in enumerate(main):
            m = np.abs(e - l["energy_keV"]) <= 1.0
            r[f"line{k}_keV"] = l["energy_keV"]
            r[f"line{k}_FEP"] = int(m.sum())
            r[f"line{k}_P/T_src"] = float((m & (sid == l["sourceID"])).sum() / max(1, (sid == l["sourceID"]).sum()))
    else:
        for l in (1460.82, 2614.511, 609.321):
            r[f"FEP_{l:.0f}"] = int((np.abs(e - l) <= 1.0).sum())
    rows.append(r)
    print(f"{job:24s} events={len(e):>10d} cps={r['cps']:9.1f} det/enter={r['det_over_enter']:.3f}")
if rows:
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.dst, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("汇总 ->", os.path.join(a.dst, "summary.csv"))
