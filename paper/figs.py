import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
INP = os.path.join(HERE, 'inputs'); OUT = os.environ.get('PAPER_OUT', os.path.join(HERE, 'out')); os.makedirs(OUT, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
from agg import *
plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.2})
W = 6.85
DET = ['NaI', 'GAGG', 'HPGe']; DLAB = {'NaI': 'NaI(Tl)', 'GAGG': 'GAGG:Ce', 'HPGe': 'HPGe'}
NUC = 'Am241 Ba133 Co57 Co60 Cs137 Eu152 I131 K40 Mn54 Na22'.split()
NL = {n: f"$^{{{n[2:] if n[1].islower() else n[1:]}}}${n[:2] if n[1].islower() else n[:1]}" for n in NUC}
COL = {'template': '#222222', 'logreg': '#9a9a9a', 'knn': '#c8c8c8', 'cnn': '#1f77b4', 'resnet': '#7fb3e0',
       'transformer': '#ff7f0e', 'gmae_msm': '#c0172d', 'gmae_msm_probe': '#e78ac3', 'gmae_dcl': '#7b3fa0'}
LAB = {'template': 'Template fit', 'logreg': 'LogReg', 'knn': 'kNN', 'cnn': 'CNN', 'resnet': 'ResNet',
       'transformer': 'Transformer (scratch)', 'gmae_msm': 'GammaMAE (fine-tuned)', 'gmae_msm_probe': 'GammaMAE (linear probe)',
       'gmae_dcl': 'GammaMAE + DCL'}
SHORT = dict(LAB, transformer='Transformer', gmae_msm='GammaMAE', gmae_msm_probe='GammaMAE-probe')
def tag(ax, s, x=-0.13, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')
def box(ax, x, y, w, h, txt, fc='#eef3fb', ec='#4a6fa5', fs=7, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015', fc=fc, ec=ec, lw=0.8))
    lines = txt.split('\n')
    if bold_first:
        ax.text(x + w / 2, y + h - 0.012, lines[0], ha='center', va='top', fontsize=fs, fontweight='bold')
        if len(lines) > 1:
            ax.text(x + w / 2, y + (h - 0.05) / 2, '\n'.join(lines[1:]), ha='center', va='center', fontsize=fs - 0.9, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)
def arrow(ax, p, q, c='#555555'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=0.8, color=c))
E = (np.arange(2048) + 0.5) * 3000 / 2048

# ---------------- Fig 1: geometry + pipeline ----------------
def fig1():
    fig = plt.figure(figsize=(W, 2.9))
    ax = fig.add_axes([0.0, 0.0, 0.34, 1.0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1.25); ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.02, 1.17, 'a', fontweight='bold', fontsize=9)
    c0 = np.array([0.27, 0.62]); R0 = 0.19
    ax.add_patch(Circle(tuple(c0), R0, fc='#fbf4e6', ec='#b98b2c', lw=0.8, ls='--'))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        p = c0 + R0 * np.array([np.cos(ang), np.sin(ang)]); q = p + 0.06 * (c0 - p) / R0
        arrow(ax, tuple(p), tuple(q), c='#b98b2c')
    ax.add_patch(Rectangle((c0[0] - 0.07, c0[1] - 0.07), 0.14, 0.14, fc='#cfe0f3', ec='#1f4e8c', lw=0.9))
    ax.text(c0[0], c0[1], 'crystal', ha='center', va='center', fontsize=6)
    xs_ = 0.90
    ax.plot([c0[0] + 0.07, xs_], [c0[1]] * 2, color='#777', lw=0.5, ls=':')
    ax.add_patch(Polygon([[xs_, c0[1]], [c0[0] + 0.07, c0[1] + 0.095], [c0[0] + 0.07, c0[1] - 0.095]], closed=True, fc='#c0172d', alpha=0.10, ec='none'))
    ax.plot([xs_, c0[0] + 0.07], [c0[1], c0[1] + 0.095], color='#c0172d', lw=0.5); ax.plot([xs_, c0[0] + 0.07], [c0[1], c0[1] - 0.095], color='#c0172d', lw=0.5)
    ax.plot([xs_], [c0[1]], marker='*', color='#c0172d', ms=9)
    ax.annotate('', xy=(c0[0] + 0.07, 0.36), xytext=(xs_, 0.36), arrowprops=dict(arrowstyle='<->', lw=0.6))
    ax.text((c0[0] + 0.07 + xs_) / 2, 0.345, 'd = 5, 10, 25 cm', ha='center', va='top', fontsize=6)
    ax.text(xs_, c0[1] + 0.06, 'point source\n(one per line)', ha='center', va='bottom', fontsize=5.8, color='#c0172d')
    ax.text(0.5, 0.25, 'background sphere: K/U/Th lines +\nscattered continuum, inward cosine law', ha='center', va='top', fontsize=5.8, color='#8a6516')
    ax.text(0.5, 1.02, 'bare crystal in air\nGATE 9.4.2 / Geant4 11.4.0\nemstandard_opt3', ha='center', va='center', fontsize=6)
    ax2 = fig.add_axes([0.35, 0.0, 0.65, 1.0]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
    ax2.text(0.0, 0.94, 'b', fontweight='bold', fontsize=9)
    xs, w, h = [0.02, 0.355, 0.69], 0.30, 0.34
    row1 = ['GATE event lists\n10 nuclides × 3 detectors\n× 3 distances, 37 kBq, 600 s\nseeds 0/1/2 = train/val/test',
            'Activity scaling\nBernoulli thinning of\nsource events to 1 kBq\n(Poisson process preserved)',
            'Ambient background\nresampled per window\nΦ$_{\\rm lines}$ = 1 γ cm$^{-2}$ s$^{-1}$\ncontinuum κ = 2']
    row2 = ['Energy response\nGaussian FWHM(E)\ngain (1+g), 15 keV threshold\n2048 ch, 0–3000 keV',
            'Time windows\nnon-overlapping 1–100 s\n≤200 windows / configuration\n43 740 spectra per split',
            'Model input\nx = ln(1 + 1000 c/N)\nper-channel z-score\n+ detector index']
    for i, t in enumerate(row1):
        box(ax2, xs[i], 0.56, w, h, t, fc='#fbf4e6' if i == 2 else '#eef3fb', ec='#b98b2c' if i == 2 else '#4a6fa5', fs=6.6)
    for i, t in enumerate(row2):
        box(ax2, xs[i], 0.13, w, h, t, fc='#f3f3f3' if i < 2 else '#fde8ea', ec='#666' if i < 2 else '#c0172d', fs=6.6)
    arrow(ax2, (xs[0] + w, 0.73), (xs[1], 0.73)); arrow(ax2, (xs[1] + w, 0.73), (xs[2], 0.73))
    arrow(ax2, (xs[2] + w / 2, 0.56), (xs[0] + w / 2 + 0.05, 0.47))
    arrow(ax2, (xs[0] + w, 0.30), (xs[1], 0.30)); arrow(ax2, (xs[1] + w, 0.30), (xs[2], 0.30))
    ax2.text(0.5, 0.06, 'perturbed test sets: the same test events re-processed with gain ±1 %, ±3 %,\nFWHM ×1.25, ×1.5 or background rate ×2, ×4',
             ha='center', va='center', fontsize=6, style='italic')
    fig.savefig(os.path.join(OUT, 'fig1_setup.pdf')); plt.close(fig)

# ---------------- Fig 2: spectra ----------------
def fig2():
    d = np.load(os.path.join(INP, 'paper_spectra.npz'))
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.0))
    show = [('Am241', '#2ca02c'), ('Cs137', '#c0172d'), ('Co60', '#1f77b4'), ('Eu152', '#9467bd')]
    for j, dn in enumerate(DET):
        di = DET.index(dn); ax = axs[0, j]
        reb = lambda v, k=4: v.reshape(-1, k).sum(1)
        Eb = E.reshape(-1, 4).mean(1)
        b = d['bkg'][di] / 30.0
        for n, c in show:
            ax.step(Eb, reb(d['mean100'][di, NUC.index(n), 1]), where='mid', color=c, lw=0.7, label=NL[n])
        ax.step(Eb, reb(b), where='mid', color='k', lw=0.8, ls='--', label='background only')
        ax.set_yscale('log'); ax.set_ylim(0.3, 3e3 if dn != 'GAGG' else 6e2); ax.set_xlim(0, 3000); ax.set_title(DLAB[dn], fontsize=8)
        if j == 0: ax.set_ylabel('counts / 5.9 keV (100 s)')
        tag(ax, 'abc'[j], x=-0.2)
        ax = axs[1, j]
        k = 8; Ek = E.reshape(-1, k).mean(1); m = Ek < 1500
        for o, (tt, c) in enumerate([(1, '#555555'), (10, '#ff7f0e'), (100, '#c0172d')]):
            v = d[f'ex_d{di}_n4_t{tt}'].astype(float); vb = v.reshape(-1, k).sum(1)[m]
            ax.step(Ek[m], o * 1.15 + vb / max(vb.max(), 1), where='mid', color=c, lw=0.7)
            ax.text(1480, o * 1.15 + 0.75, f'{tt} s, N = {int(v.sum())}', ha='right', fontsize=6, color=c)
        ax.axvline(661.7, color='#c0172d', lw=0.4, ls=':'); ax.set_yticks([]); ax.set_xlim(0, 1500); ax.set_xlabel('Energy (keV)')
        if j == 0: ax.set_ylabel('counts / 11.7 keV (scaled, offset)')
        tag(ax, 'def'[j], x=-0.2)
    axs[0, 0].legend(fontsize=6, loc='upper right', ncol=1)
    fig.tight_layout(h_pad=0.8, w_pad=0.6); fig.savefig(os.path.join(OUT, 'fig2_spectra.pdf')); plt.close(fig)

# ---------------- Fig 3: architecture ----------------
def fig3():
    fig = plt.figure(figsize=(W, 2.35)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    d = np.load(os.path.join(INP, 'paper_spectra.npz')); v = np.log1p(1000 * d['mean100'][0, 4, 1] / d['mean100'][0, 4, 1].sum())
    sa = fig.add_axes([0.015, 0.50, 0.17, 0.36]); sa.plot(E, v, color='k', lw=0.6); sa.set_xticks([0, 1500, 3000]); sa.set_yticks([])
    sa.tick_params(labelsize=5.5); sa.set_xlabel('keV', fontsize=5.5, labelpad=1)
    rng = np.random.default_rng(3); masked = rng.permutation(64)[:32]
    for m in masked: sa.axvspan(m * 3000 / 64, (m + 1) * 3000 / 64, color='#c0172d', alpha=0.15, lw=0)
    ax.text(0.10, 0.90, '2048-ch spectrum,\n64 patches × 32 ch', ha='center', fontsize=6.3)
    ax.text(0.10, 0.36, 'shaded: masked\npatches (50 %)', ha='center', fontsize=6, color='#c0172d', va='top')
    box(ax, 0.21, 0.50, 0.14, 0.36, 'Tokens\nlinear patch embedding\n+ position embedding\n+ detector embedding\n(NaI / GAGG / HPGe)')
    box(ax, 0.39, 0.50, 0.16, 0.36, 'Encoder\n6 pre-LN Transformer\nlayers, d = 256,\n8 heads (4.8 M param.)\nvisible tokens in stage 1', fc='#fde8ea', ec='#c0172d')
    box(ax, 0.59, 0.62, 0.17, 0.30, 'Decoder\nmask tokens + position\n+ detector embedding\n2 layers, d = 128', fc='#f3f3f3', ec='#666')
    box(ax, 0.80, 0.62, 0.18, 0.30, 'MSM loss\nMSE on masked patches\nof the standardised\nlog spectrum', fc='#f3f3f3', ec='#666')
    box(ax, 0.59, 0.12, 0.17, 0.30, 'Fine-tuning\nall 64 tokens, mean pool\nlinear head (10 classes)\nencoder lr × 0.3', fc='#eef7ee', ec='#2c7a2c')
    box(ax, 0.80, 0.12, 0.18, 0.30, 'Linear probe\nfrozen encoder\nmean-pooled embedding\n+ logistic regression', fc='#eef7ee', ec='#2c7a2c')
    arrow(ax, (0.185, 0.68), (0.21, 0.68)); arrow(ax, (0.35, 0.68), (0.39, 0.68)); arrow(ax, (0.55, 0.74), (0.59, 0.77))
    arrow(ax, (0.76, 0.77), (0.80, 0.77)); arrow(ax, (0.47, 0.50), (0.64, 0.42)); arrow(ax, (0.53, 0.50), (0.87, 0.42))
    ax.text(0.785, 0.035, 'stage 2: supervised, labelled spectra', ha='center', fontsize=6.3, color='#2c7a2c')
    ax.text(0.67, 0.955, 'stage 1: self-supervised, unlabelled spectra', ha='center', fontsize=6.3, color='#555')
    ax.text(0.30, 0.10, 'Transformer (scratch): identical encoder and head,\nrandom initialisation, no pre-training', ha='center',
            fontsize=6.3, style='italic')
    fig.savefig(os.path.join(OUT, 'fig3_model.pdf')); plt.close(fig)

# ---------------- Fig 4: E1 vs time per detector + distance ----------------
def fig4():
    fig, axs = plt.subplots(1, 4, figsize=(W, 2.2))
    ms_ = ['template', 'logreg', 'cnn', 'transformer', 'gmae_msm']
    T = [1, 3, 10, 30, 100]
    for j, dn in enumerate(['NaI', 'GAGG', 'HPGe']):
        ax = axs[j]
        for m in ms_:
            r = ms(sel('e1', m, 'clean').query('det==@dn'), 't')
            y = [r[t][0] for t in T]; e = [0 if np.isnan(r[t][1]) else r[t][1] for t in T]
            ax.errorbar(T, y, yerr=e, color=COL[m], marker='o', ms=2.5, lw=1.0, capsize=1.5, label=SHORT[m],
                        ls='--' if m in ('template', 'logreg') else '-')
        ax.set_xscale('log'); ax.set_xticks(T); ax.set_xticklabels(T); ax.set_ylim(40, 101); ax.set_title(DLAB[dn], fontsize=8)
        ax.set_xlabel('measurement time (s)'); ax.grid(alpha=0.25, lw=0.4)
        if j == 0: ax.set_ylabel('accuracy (%)')
        tag(ax, 'abcd'[j], x=-0.22)
    ax = axs[3]; D = [5, 10, 25]; xx = np.arange(3); wbar = 0.16
    for i, m in enumerate(ms_):
        r = ms(sel('e1', m, 'clean'), 'dist')
        ax.bar(xx + (i - 2) * wbar, [r[k][0] for k in D], wbar, color=COL[m], yerr=[0 if np.isnan(r[k][1]) else r[k][1] for k in D],
               error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(xx); ax.set_xticklabels(['5', '10', '25']); ax.set_xlabel('source distance (cm)'); ax.set_ylim(0, 100)
    ax.set_title('all detectors', fontsize=8); ax.grid(axis='y', alpha=0.25, lw=0.4); tag(ax, 'd', x=-0.22)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.5); fig.savefig(os.path.join(OUT, 'fig4_e1.pdf')); plt.close(fig)

# ---------------- Fig 5: LODO ----------------
def fig5():
    ms_ = ['logreg', 'knn', 'cnn', 'resnet', 'transformer', 'gmae_dcl', 'gmae_msm_probe', 'gmae_msm']
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True)
    for j, dn in enumerate(['NaI', 'GAGG', 'HPGe']):
        ax = axs[j]; x = np.arange(len(ms_))
        for k, (kk, al, hat) in enumerate([('k0', 1.0, None), ('k10', 0.45, None)]):
            vals = [ms(sel(f'lodo_{dn}_{kk}', m)) for m in ms_]
            ax.bar(x + (k - 0.5) * 0.38, [v[0] for v in vals], 0.38, color=[COL[m] for m in ms_], alpha=al,
                   yerr=[0 if np.isnan(v[1]) else v[1] for v in vals], error_kw=dict(lw=0.5, capsize=1), edgecolor='k', linewidth=0.3)
        e1 = ms(sel('e1', 'gmae_msm', 'clean').query('det==@dn'))[0]
        ax.axhline(e1, color='#c0172d', lw=0.6, ls=':')
        ax.text(-0.5, e1 + 0.8, 'GammaMAE with all target labels (E1)', fontsize=5.5, ha='left', va='bottom', color='#c0172d')
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m].replace('GammaMAE', 'GMAE').replace(' + ', '+') for m in ms_], rotation=55, ha='right', fontsize=6.3)
        ax.set_title(f'target detector: {DLAB[dn]}', fontsize=7.5); ax.set_ylim(40, 100)
        ax.grid(axis='y', alpha=0.25, lw=0.4); tag(ax, 'abc'[j], x=-0.15)
    axs[0].set_ylabel('target-detector accuracy (%)')
    from matplotlib.patches import Patch
    axs[1].legend([Patch(fc='#888', ec='k', lw=0.3), Patch(fc='#888', alpha=0.45, ec='k', lw=0.3)],
                  ['zero-shot (k = 0)', 'k = 10 target labels / class'], fontsize=6, loc='upper right')
    fig.tight_layout(w_pad=0.4); fig.savefig(os.path.join(OUT, 'fig5_lodo.pdf')); plt.close(fig)

# ---------------- Fig 6: perturbations, distance extrapolation, few labels ----------------
def fig6():
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1.45, 0.85, 1.0]))
    ms_ = ['template', 'cnn', 'transformer', 'gmae_msm']
    conds = ['clean', 'gain-3%', 'gain-1%', 'gain+1%', 'gain+3%', 'res x1.25', 'res x1.5', 'bkg x2', 'bkg x4']
    cl = ['clean', 'gain −3%', 'gain −1%', 'gain +1%', 'gain +3%', 'FWHM ×1.25', 'FWHM ×1.5', 'bkg ×2', 'bkg ×4']
    ax = axs[0]; x = np.arange(len(conds))
    for i, m in enumerate(ms_):
        v = [ms(sel('e1', m, c)) for c in conds]
        ax.bar(x + (i - 1.5) * 0.2, [a[0] for a in v], 0.2, color=COL[m], label=SHORT[m],
               yerr=[0 if np.isnan(a[1]) else a[1] for a in v], error_kw=dict(lw=0.5, capsize=0.8))
    ax.set_xticks(x); ax.set_xticklabels(cl, fontsize=6.0, rotation=40, ha='right'); ax.set_ylim(50, 90); ax.set_ylabel('accuracy (%)')
    ax.set_title('perturbed test spectra (trained on clean)', fontsize=7.5); ax.grid(axis='y', alpha=0.25, lw=0.4)
    ax.legend(fontsize=6, ncol=2, loc='upper right'); tag(ax, 'a', x=-0.1)
    ax = axs[1]; ms2 = ['template', 'logreg', 'cnn', 'transformer', 'gmae_msm']; D = [5, 10, 25]; xx = np.arange(3)
    for i, m in enumerate(ms2):
        r = ms(sel('dist10', m), 'dist')
        ax.bar(xx + (i - 2) * 0.16, [r[k][0] for k in D], 0.16, color=COL[m], label=SHORT[m],
               yerr=[0 if np.isnan(r[k][1]) else r[k][1] for k in D], error_kw=dict(lw=0.5, capsize=0.8))
    ax.set_xticks(xx); ax.set_xticklabels(['5', '10*', '25']); ax.set_xlabel('test distance (cm); *training'); ax.set_ylim(0, 100)
    ax.set_title('train at 10 cm only', fontsize=7.5); ax.grid(axis='y', alpha=0.25, lw=0.4); tag(ax, 'b', x=-0.2)
    ax = axs[2]; T = [1, 3, 10, 30, 100]
    for m in ['template', 'cnn', 'resnet', 'transformer', 'gmae_msm', 'gmae_msm_probe']:
        r = ms(sel('low5', m), 't')
        ax.errorbar(T, [r[t][0] for t in T], yerr=[0 if np.isnan(r[t][1]) else r[t][1] for t in T], color=COL[m], marker='o', ms=2.3,
                    lw=1.0, capsize=1.2, label=SHORT[m], ls='--' if m in ('template', 'gmae_msm_probe') else '-')
    ax.set_xscale('log'); ax.set_xticks(T); ax.set_xticklabels(T); ax.set_xlabel('measurement time (s)'); ax.set_ylim(30, 100)
    ax.set_title('5 labels per cell (2250 total)', fontsize=7.5); ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=5.6, loc='lower right')
    tag(ax, 'c', x=-0.2)
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(OUT, 'fig6_stress.pdf')); plt.close(fig)

if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']:
        globals()[f](); print('ok', f)
