import os, pandas as pd, numpy as np
INP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')
R = pd.read_csv(os.path.join(INP, 'results.csv')); R['acc'] = 100 * R.correct / R.n
R2 = pd.read_csv(os.path.join(INP, 'results_low_tr200.csv')); R2['acc'] = 100 * R2.correct / R2.n
def ms(df, by=None):
    """cell-average per seed, then mean/sd over seeds"""
    keys = ['seed'] + ([by] if isinstance(by, str) else (by or []))
    g = df.groupby(keys).acc.mean().reset_index()
    if by is None:
        v = g.acc.values; return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)
    out = {}
    for k, gg in g.groupby(by):
        v = gg.acc.values; out[k] = (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan)
    return out
def sel(exp, method, testset=None, df=R):
    d = df[(df.exp == exp) & (df.method == method)]
    return d if testset is None else d[d.testset == testset]
METHODS = ['template', 'logreg', 'knn', 'cnn', 'resnet', 'transformer', 'gmae_msm', 'gmae_msm_probe', 'gmae_dcl']
def fmt(m, s, d=1):
    return f"{m:.{d}f}" if (s is None or np.isnan(s)) else f"{m:.{d}f}$\\pm${s:.{d}f}"
