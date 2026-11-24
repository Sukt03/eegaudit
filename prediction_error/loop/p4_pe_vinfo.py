"""P4 (PE) + P3b: V-information ladder for the prediction-error target with per-trial PVI, LOPO conditional classes.
T = normal-scores PE (h(T) = 1/2 log2(2 pi e) exactly). For each read-out class (raw ERP ridge CV prediction, EEGNet LOPO, REVE LOPO ridge, CBraMod-ft; y z-scored
within participant) and conditional family q(t|y): linear N(a+by, s^2) [P4], quadratic N(a+by+cy^2, s^2) [P3b-i], heteroscedastic quadratic
N(a+by+cy^2, (s0+s1|y|)^2) [P3b-ii]; parameters fitted on the pooled (y, t) of the OTHER 22 participants (least squares; s1 by least squares of
|residual| on |y|). H_V(T|X)_i = mean_t[-log2 q(t|y)]; H_V(T)_i = the same with y circularly shifted (>= 40 trials, 20 shifts); I_V = H_V(T) - H_V(T|X);
PVI_t = log2 q(t|y_t) - log2 q_shift(t|y_t) (mean = I_V) with median and IQR. Ladder increments per participant: EEGNet - raw (CNN - linear),
REVE/CBraMod - EEGNet (transformer - CNN). Output results/loop/p4/p4_pe_vinfo.json."""
import sys, json, numpy as np
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); rng = np.random.default_rng(0)
import deep_decoders as DD
src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); ns = {}
exec('import torch, numpy as np\nimport sys; sys.path.insert(0, "/path/to/eegaudit")\n' + src[src.index('from belief_bottleneck import LAM, folds'):src.index('SMOKE =')], ns); cvp = ns['cv_predict_fast']
data, _ = DD.load_all(); nS = len(data); nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y)); z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
T = [nscore(yt) for _, yt, _ in data]
Y = {}
for dec in ['raw', 'eegnet', 'reve_lopo', 'cbramod_ft']:
    Y[dec] = []
    for i, (X, yt, tri) in enumerate(data):
        zf = np.load(ROOT / f'results/features/{dec}/sub-{i + 1:03d}.npz'); F = zf['F'].astype(float)
        Y[dec].append(z(F[:, 0]) if F.shape[1] == 1 else z(cvp(F, yt[:, None].astype(float), 64, colw=zf['colw'].astype(float))[:, 0]))
def design(y, fam): return np.c_[np.ones(len(y)), y] if fam == 'linear' else np.c_[np.ones(len(y)), y, y ** 2]
def fit(yo, to, fam):
    A = design(yo, fam); w, *_ = np.linalg.lstsq(A, to, rcond=None); res = to - A @ w
    if fam == 'hetero':
        B = np.c_[np.ones(len(yo)), np.abs(yo)]; sw, *_ = np.linalg.lstsq(B, np.abs(res) * np.sqrt(np.pi / 2), rcond=None); return w, sw
    return w, np.array([res.std(), 0.0])
def nll_bits(y, t, w, sw, fam):
    mu = design(y, fam) @ w; sd = np.maximum(sw[0] + sw[1] * np.abs(y), 1e-3) if fam == 'hetero' else np.full(len(y), max(sw[0], 1e-3))
    return -stats.norm.logpdf(t, mu, sd) / np.log(2)
out = {'n_participants': nS, 'classes': {}, 'ladder': {}}
for dec, Ys in Y.items():
    out['classes'][dec] = {}
    for fam in ['linear', 'quadratic', 'hetero']:
        rows = []
        for i in range(nS):
            yo = np.concatenate([Ys[j] for j in range(nS) if j != i]); to = np.concatenate([T[j] for j in range(nS) if j != i]); w, sw = fit(yo, to, fam)
            y, t = Ys[i], T[i]; n = len(y); ce = nll_bits(y, t, w, sw, fam)
            shifts = rng.integers(40, n - 40, 20); ce_sh = np.mean([nll_bits(np.roll(y, s), t, w, sw, fam) for s in shifts], 0)
            pvi = ce_sh - ce; rows.append(dict(subject=i + 1, H_cond=float(ce.mean()), H_floor=float(ce_sh.mean()), I_V=float(pvi.mean()), pvi_median=float(np.median(pvi)), pvi_iqr=[float(np.percentile(pvi, 25)), float(np.percentile(pvi, 75))], frac_pvi_positive=float((pvi > 0).mean())))
        IV = np.array([r['I_V'] for r in rows]); bs = [IV[rng.integers(0, nS, nS)].mean() for _ in range(4000)]
        out['classes'][dec][fam] = dict(rows=rows, I_V_mean=float(IV.mean()), I_V_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], I_V_median=float(np.median(IV)), n_nonneg=int((IV >= 0).sum()), H_floor_mean=float(np.mean([r['H_floor'] for r in rows])))
        print(f"{dec:10s} {fam:9s} I_V {IV.mean():.4f} [{out['classes'][dec][fam]['I_V_ci95'][0]:.4f},{out['classes'][dec][fam]['I_V_ci95'][1]:.4f}] median {np.median(IV):.4f} nonneg {int((IV>=0).sum())}/23 H_floor {out['classes'][dec][fam]['H_floor_mean']:.3f}", flush=True)
L = lambda dec, fam: np.array([r['I_V'] for r in out['classes'][dec][fam]['rows']])
for fam in ['linear', 'quadratic', 'hetero']:
    inc = {'eegnet_minus_raw': L('eegnet', fam) - L('raw', fam), 'reve_minus_eegnet': L('reve_lopo', fam) - L('eegnet', fam), 'cbramod_minus_eegnet': L('cbramod_ft', fam) - L('eegnet', fam)}
    out['ladder'][fam] = {k: dict(mean=float(v.mean()), ci95=[float(np.percentile([v[rng.integers(0, nS, nS)].mean() for _ in range(4000)], q)) for q in (2.5, 97.5)], n_positive=int((v > 0).sum())) for k, v in inc.items()}
    out['ladder'][fam]['median_order'] = sorted(['raw', 'eegnet', 'reve_lopo', 'cbramod_ft'], key=lambda d: -np.median(L(d, fam)))
for dec in Y:
    for fam in ['quadratic', 'hetero']:
        d = L(dec, fam) - L(dec, 'linear'); out['classes'][dec][fam]['increment_over_linear'] = dict(mean=float(d.mean()), ci95=[float(np.percentile([d[rng.integers(0, nS, nS)].mean() for _ in range(4000)], q)) for q in (2.5, 97.5)])
json.dump(out, open(ROOT / 'results/loop/p4/p4_pe_vinfo.json', 'w'), indent=1)
print('ladder', json.dumps(out['ladder'])[:1500]); print('P3b increments', {d: {f: out['classes'][d][f]['increment_over_linear'] for f in ('quadratic', 'hetero')} for d in Y})
