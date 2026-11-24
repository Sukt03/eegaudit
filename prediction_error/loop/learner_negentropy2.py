"""S-L (a), second pass. (1) Negentropy J of the standardised residualised PE target with three estimators, each debiased against N(0,1)
samples of the same n: Kozachenko-Leonenko kNN (k=5), Vasicek m-spacing (plain), Gaussian KDE plug-in; plus the entropy of a skew-normal MLE fit.
(2) Repair: the Gaussian-channel bound is exact-valid for a Gaussian target, and I(X;Y) is invariant to monotone transforms of X, so
I(PE; readout) >= -1/2 log2(1 - rho_g^2) with rho_g = corr(normal-scores(PE), readout) holds without correction. rho_g per participant for
every decoder whose LOPO single-column read-out is cached (results/features/<dec>/sub-XXX.npz, F (n,1)) on the same trials as the target."""
import sys, json, numpy as np
from pathlib import Path
from scipy import stats, special
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT))
import deep_decoders as DD
data, _ = DD.load_all(); hg = 0.5 * np.log(2 * np.pi * np.e); rng = np.random.default_rng(0)
def h_knn(x, k=5):
    x = np.sort(np.asarray(x, float)); n = len(x); d = np.empty(n)
    for i in range(n):
        dd = np.abs(x - x[i]); dd[i] = np.inf; d[i] = np.partition(dd, k - 1)[k - 1]
    return special.digamma(n) - special.digamma(k) + np.log(2) + np.mean(np.log(d + 1e-12))
def h_vas(x):
    x = np.sort(np.asarray(x, float)); n = len(x); m = int(round(np.sqrt(n))); xp = np.concatenate([np.full(m, x[0]), x, np.full(m, x[-1])])
    return float(np.mean(np.log(n / (2 * m) * (xp[2 * m:] - xp[:-2 * m]) + 1e-12)))
def h_kde(x): k = stats.gaussian_kde(x); return float(-np.mean(np.log(k(x))))
def h_skewnorm(x):
    a, loc, sc = stats.skewnorm.fit(x); return float(stats.skewnorm.entropy(a, loc, sc))
ests = dict(knn=h_knn, vasicek=h_vas, kde=h_kde, skewnorm=h_skewnorm)
rows = []; bias = {k: [] for k in ests}
for i, (X, y, tri) in enumerate(data):
    y = (y - y.mean()) / y.std(); g = rng.standard_normal(len(y)); r = dict(subject=i + 1, n=len(y))
    for k, f in ests.items():
        r[f'J_{k}_bits'] = (hg - f(y)) / np.log(2); bias[k].append((hg - f(g)) / np.log(2))
    rows.append(r)
J = {k: dict(raw_mean=float(np.mean([r[f'J_{k}_bits'] for r in rows])), gauss_ref_mean=float(np.mean(bias[k])),
             debiased_mean=float(np.mean([r[f'J_{k}_bits'] for r in rows]) - np.mean(bias[k])),
             debiased_per_participant=[float(r[f'J_{k}_bits'] - np.mean(bias[k])) for r in rows]) for k in ests}
# (2) Gaussianised-target coupling per decoder
b = lambda r: float(-0.5 * np.log2(1 - min(abs(r), 0.999) ** 2))
def nscore(y): return stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))
dec = {}
for d in ['eegnet', 'eegnet_post', 'reve_lopo', 'p3_lopo', 'rewp_lopo', 'cbramod_ft']:
    rr = []
    for i, (X, y, tri) in enumerate(data):
        p = ROOT / f'results/features/{d}/sub-{i + 1:03d}.npz'
        if not p.exists(): break
        z = np.load(p); F = z['F']
        if F.shape[1] != 1 or len(F) != len(y) or not np.array_equal(z['tri'], tri): rr = None; break
        f = F[:, 0]; rr.append(dict(subject=i + 1, r_raw=float(np.corrcoef(f, y)[0, 1]), r_gauss=float(np.corrcoef(f, nscore(y))[0, 1]), r_spearman=float(stats.spearmanr(f, y)[0])))
    if rr and len(rr) == len(data):
        R = {k: np.array([x[k] for x in rr]) for k in ('r_raw', 'r_gauss', 'r_spearman')}
        dec[d] = dict(rows=rr, **{k: dict(mean=float(v.mean()), fisher_mean=float(np.tanh(np.arctanh(v).mean())), bits_at_fisher_mean=b(np.tanh(np.arctanh(v).mean())),
                                           mean_per_participant_bits=float(np.mean([b(x) for x in v]))) for k, v in R.items()},
                      delta_bits_gauss_minus_raw_mean_per_participant=float(np.mean([b(x) for x in R['r_gauss']]) - np.mean([b(x) for x in R['r_raw']])))
out = dict(n_participants=len(data), negentropy_bits=J, skew_mean=float(np.mean([stats.skew((y - y.mean()) / y.std()) for _, y, _ in data])),
           excess_kurtosis_mean=float(np.mean([stats.kurtosis(y) for _, y, _ in data])), gaussianised_coupling=dec, rows=rows)
json.dump(out, open(ROOT / 'results/loop/learner/negentropy_pe_target_v2.json', 'w'), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'debiased_per_participant'} for k, v in J.items()}, indent=1))
for d, v in dec.items(): print(d, {k: {kk: round(vv, 4) for kk, vv in v[k].items()} for k in ('r_raw', 'r_gauss', 'r_spearman')}, 'delta bits', round(v['delta_bits_gauss_minus_raw_mean_per_participant'], 4))
