import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
INP = os.path.join(HERE, 'inputs'); OUT = os.environ.get('PAPER_OUT', os.path.join(HERE, 'out')); os.makedirs(OUT, exist_ok=True)
from agg import *
import pandas as pd, numpy as np
d = np.load(os.path.join(INP, 'paper_spectra.npz')); DET = ['NaI', 'GAGG', 'HPGe']
NUC = 'Am241 Ba133 Co57 Co60 Cs137 Eu152 I131 K40 Mn54 Na22'.split()
def nl(n):
    i = 2 if n[1].islower() else 1
    return f"$^{{{n[i:]}}}${n[:i]}"
LAB = {'template': 'Template fit (Poisson ML)', 'logreg': 'Logistic regression', 'knn': '$k$NN (cosine)', 'cnn': 'CNN',
       'resnet': 'ResNet', 'transformer': 'Transformer (scratch)', 'gmae_msm': 'GammaMAE (fine-tuned)',
       'gmae_msm_probe': 'GammaMAE (linear probe)', 'gmae_dcl': 'GammaMAE + DCL'}
def f(v, bold=False):
    m, s = v; t = f"{m:.1f}" if np.isnan(s) else f"{m:.1f}$\\pm${s:.1f}"
    return f"\\textbf{{{t}}}" if bold else t
def best(vals):
    ms_ = [v[0] for v in vals]; b = max(ms_); return [abs(x - b) < 0.05 for x in ms_]
out = {}
# --- nuclide table: rates at 10 cm, 1 kBq
sl = pd.read_csv(os.path.join(HERE, '..', 'gate', 'source_lines.csv'))
rows = []
for ni, n in enumerate(NUC):
    g = sl[sl.nuclide == n]
    main = ', '.join(f"{e:.1f}" for e, i, t in zip(g.energy_keV, g.intensity, g.type) if (i >= 0.05 and t == 'g') or (n == 'Na22' and t == 'g'))
    rates = []
    for di in range(3):
        rates.append((d['mean100'][di, ni, 1].sum() - d['bkg'][di].sum() / 30) / 100)
    rows.append(f"{nl(n)} & {main} & {len(g)} & " + ' & '.join(f"{r:.1f}" for r in rates) + r" \\")
out['tab_nuclides'] = '\n'.join(rows)
# --- detector stats: src cps at 5/10/25 and bkg cps
st = d['stats']  # di, dist, t, n_src, n_bkg, n
rows = []
for di, dn in enumerate(DET):
    s = st[(st[:, 0] == di) & (st[:, 2] == 100)]
    src = [s[s[:, 1] == r][0, 3] / 100 for r in (5, 10, 25)]; bkg = s[:, 4].mean() / 100
    rows.append((dn, src, bkg))
out['det_rates'] = rows
# --- E1 main
cols = []
rows = []
M = ['template', 'logreg', 'knn', 'cnn', 'resnet', 'transformer', 'gmae_msm_probe', 'gmae_msm']
vals = {m: [ms(sel('e1', m, 'clean'))] + [ms(sel('e1', m, 'clean').query('det==@dn')) for dn in DET]
        + [ms(sel('e1', m, 'clean').query('t==@t')) for t in (1, 100)] + [ms(sel('e1', m, 'clean').query('dist==@r')) for r in (5, 25)] for m in M}
B = [best([vals[m][j] for m in M]) for j in range(8)]
for i, m in enumerate(M):
    rows.append(LAB[m] + ' & ' + ' & '.join(f(vals[m][j], B[j][i]) for j in range(8)) + r' \\' + (r' \midrule' if m in ('knn',) else ''))
out['tab_e1'] = '\n'.join(rows)
# --- LODO
M2 = ['logreg', 'knn', 'cnn', 'resnet', 'transformer', 'gmae_msm_probe', 'gmae_msm']
keys = [(dn, k) for dn in DET for k in ('k0', 'k10')]
V = {m: [ms(sel(f'lodo_{dn}_{k}', m)) for dn, k in keys] for m in M2}
B = [best([V[m][j] for m in M2]) for j in range(6)]
rows = [LAB[m] + ' & ' + ' & '.join(f(V[m][j], B[j][i]) for j in range(6)) + r' \\' + (r' \midrule' if m == 'knn' else '') for i, m in enumerate(M2)]
e1row = 'GammaMAE, all target labels (E1) & ' + ' & '.join(f"\\multicolumn{{2}}{{c}}{{{ms(sel('e1','gmae_msm','clean').query('det==@dn'))[0]:.1f}}}" for dn in DET) + r' \\'
out['tab_lodo'] = '\n'.join(rows) + '\n\\midrule\n' + e1row
# --- perturbations
conds = ['clean', 'gain-3%', 'gain-1%', 'gain+1%', 'gain+3%', 'res x1.25', 'res x1.5', 'bkg x2', 'bkg x4']
M3 = ['template', 'logreg', 'knn', 'cnn', 'resnet', 'transformer', 'gmae_msm_probe', 'gmae_msm']
V = {m: [ms(sel('e1', m, c)) for c in conds] for m in M3}
B = [best([V[m][j] for m in M3]) for j in range(9)]
def g(v, b):
    t = f"{v[0]:.1f}"; return f"\\textbf{{{t}}}" if b else t
rows = [LAB[m] + ' & ' + ' & '.join(g(V[m][j], B[j][i]) for j in range(9)) + r' \\' + (r' \midrule' if m == 'knn' else '') for i, m in enumerate(M3)]
out['tab_pert'] = '\n'.join(rows)
out['pert_sd_max'] = max(V[m][j][1] for m in ['cnn', 'resnet', 'transformer', 'gmae_msm', 'gmae_msm_probe'] for j in range(9))
# --- E4 + E5
V4 = {m: [ms(sel('dist10', m).query('dist==@r')) for r in (5, 10, 25)] for m in M3}
V5 = {m: [ms(sel('low5', m))] + [ms(sel('low5', m).query('t==@t')) for t in (1, 10, 100)] for m in M3}
B4 = [best([V4[m][j] for m in M3]) for j in range(3)]; B5 = [best([V5[m][j] for m in M3]) for j in range(4)]
rows = [LAB[m] + ' & ' + ' & '.join(f(V4[m][j], B4[j][i]) for j in range(3)) + ' & ' + ' & '.join(f(V5[m][j], B5[j][i]) for j in range(4))
        + r' \\' + (r' \midrule' if m == 'knn' else '') for i, m in enumerate(M3)]
t2 = R2[R2.method == 'transformer']
v200 = [ms(t2)] + [ms(t2.query('t==@t')) for t in (1, 10, 100)]
rows.append(r'Transformer (scratch), 200 epochs & -- & -- & -- & ' + ' & '.join(f(v) for v in v200) + r' \\')
out['tab_e4e5'] = '\n'.join(rows)
# --- appendix: DCL rows
dcl = dict(e1=ms(sel('e1', 'gmae_dcl', 'clean')), lodo=[ms(sel(f'lodo_{dn}_{k}', 'gmae_dcl')) for dn, k in keys],
           dist25=ms(sel('dist10', 'gmae_dcl').query('dist==25')), gain3=ms(sel('e1', 'gmae_dcl', 'gain+3%')))
out['dcl'] = dcl
# --- E1 per detector x time (appendix)
rows = []
for m in ['template', 'logreg', 'cnn', 'transformer', 'gmae_msm']:
    cells = []
    for dn in DET:
        r = ms(sel('e1', m, 'clean').query('det==@dn'), 't'); cells += [f"{r[t][0]:.1f}" for t in (1, 3, 10, 30, 100)]
    rows.append(LAB[m] + ' & ' + ' & '.join(cells) + r' \\')
out['tab_dett'] = '\n'.join(rows)
rows = []
for m in ['template', 'logreg', 'cnn', 'transformer', 'gmae_msm']:
    cells = []
    for dn in DET:
        r = ms(sel('e1', m, 'clean').query('det==@dn'), 'dist'); cells += [f"{r[k][0]:.1f}" for k in (5, 10, 25)]
    rows.append(LAB[m] + ' & ' + ' & '.join(cells) + r' \\')
out['tab_detd'] = '\n'.join(rows)
for k, v in out.items():
    if k.startswith('tab_'):
        open(os.path.join(OUT, k + '.tex'), 'w').write(v + '\n')
import json; json.dump({k: v for k, v in out.items() if not k.startswith('tab_')}, open(os.path.join(OUT, 'numbers.json'), 'w'), default=float, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.startswith('tab_')}, default=lambda x: round(float(x), 2)))
