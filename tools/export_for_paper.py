# -*- coding: utf-8 -*-
"""Export the small files used for the paper figures/tables (mean spectra, example spectra,
background templates, count statistics, GATE summary) into one zip.
Usage: python -m tools.export_for_paper --data D:\\gammamae_v2_data_1kBq --npz D:\\gammamae_v2_npz --out D:\\gmae_paper_export.zip
The file paper_spectra.npz inside the zip is the one expected in paper/inputs/."""
import argparse, io, os, zipfile, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
z = np.load(os.path.join(a.data, "test_clean.npz")); c, det, nuc, dist, t = z["counts"], z["det"], z["nuc"], z["dist"], z["t"]
mean = np.zeros((3, 10, 3, 2048), np.float32)          # (det, nuc, dist): mean 100 s spectrum
for di in range(3):
    for ni in range(10):
        for ri, r in enumerate([5, 10, 25]):
            m = (det == di) & (nuc == ni) & (dist == r) & (t == 100)
            if m.any(): mean[di, ni, ri] = c[m].mean(0)
ex = {}                                                 # Am241 / Co60 / Cs137 at 10 cm: first window of each live time
for ni in (0, 3, 4):
    for di in range(3):
        for tt in (1, 10, 100):
            i = np.where((det == di) & (nuc == ni) & (dist == 10) & (t == tt))[0]
            if i.size: ex[f"ex_d{di}_n{ni}_t{tt}"] = c[i[0]]
stats = []                                              # det, dist, t, mean source counts, mean background counts, n spectra
for di in range(3):
    for r in (5, 10, 25):
        for tt in (1, 3, 10, 30, 100):
            m = (det == di) & (dist == r) & (t == tt)
            stats.append([di, r, tt, z["n_src"][m].mean(), z["n_bkg"][m].mean(), int(m.sum())])
buf = io.BytesIO()
np.savez_compressed(buf, mean100=mean, stats=np.array(stats), bkg=np.load(os.path.join(a.data, "bkg_templates.npy")), **ex)
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("paper_spectra.npz", buf.getvalue())
    zf.write(os.path.join(a.data, "dataset_config.json"), "dataset_config.json")
    zf.write(os.path.join(a.npz, "summary.csv"), "gate_summary.csv")
print("->", a.out, os.path.getsize(a.out) // 1024, "KB")
