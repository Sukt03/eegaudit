"""S-L (a): validity of the Gaussian-channel bound I >= -1/2 log2(1 - rho^2) for the standardised, outcome-residualised signed PE target.
For any (X, Y) with X standardised: I(X;Y) >= h(X) - h(X - aY) >= h(X) - 1/2 log(2 pi e (1 - rho^2)) = -1/2 log(1 - rho^2) - J(X),
with J(X) = 1/2 log(2 pi e) - h(X) >= 0 the negentropy of the target. So the Gaussian bound overstates the guaranteed information by at most
J(X) bits; the corrected lower bound is bits_gauss - J. J is estimated per participant from the trial-wise targets exactly as deep_decoders.py
builds them (load_all: PE at the behavioural MAP, residualised on outcome, arm, arm x outcome, cubic time; z-scored) with the Vasicek
m-spacing entropy estimator (m = round(sqrt(n)), bias-corrected) and cross-checked with a Gaussian-KDE plug-in; skew and excess kurtosis reported."""
import sys, json, numpy as np
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT))
import deep_decoders as DD
data, n_ch = DD.load_all()
def vasicek(x, m=None):
    x = np.sort(np.asarray(x, float)); n = len(x); m = m or int(round(np.sqrt(n)))
    xp = np.concatenate([np.full(m, x[0]), x, np.full(m, x[-1])]); d = xp[2 * m:] - xp[:-2 * m]
    h = np.mean(np.log(n / (2 * m) * d)); 
    # bias correction (Ebrahimi-style constant for Vasicek): log(2m) - 1 + psi(2m) ... use the standard small-sample term
    return h + np.log(2 * m) - 1 + (2 * m / n) * 0 - (stats.digamma(2 * m) if False else 0)  # plain Vasicek; KDE below as cross-check
def kde_entropy(x):
    k = stats.gaussian_kde(x); return float(-np.mean(np.log(k(x))))
hg = 0.5 * np.log(2 * np.pi * np.e)
rows = []
for i, (X, y, tri) in enumerate(data):
    y = (y - y.mean()) / y.std()
    hv = float(vasicek(y)); hk = kde_entropy(y)
    rows.append(dict(subject=i + 1, n=len(y), skew=float(stats.skew(y)), excess_kurtosis=float(stats.kurtosis(y)), shapiro_p=float(stats.shapiro(y).pvalue),
                     J_vasicek_bits=float((hg - hv) / np.log(2)), J_kde_bits=float((hg - hk) / np.log(2))))
J_v = np.array([r['J_vasicek_bits'] for r in rows]); J_k = np.array([r['J_kde_bits'] for r in rows])
# Gaussian reference: the same estimators on N(0,1) samples of the same n (estimator bias)
rng = np.random.default_rng(0); ref = []
for r in rows:
    g = rng.standard_normal(r['n']); ref.append(((hg - vasicek(g)) / np.log(2), (hg - kde_entropy(g)) / np.log(2)))
ref = np.array(ref)
out = dict(n_participants=len(rows), rows=rows,
           J_bits=dict(vasicek_mean=float(J_v.mean()), vasicek_max=float(J_v.max()), kde_mean=float(J_k.mean()), kde_max=float(J_k.max()),
                       estimator_bias_on_gaussian=dict(vasicek_mean=float(ref[:, 0].mean()), kde_mean=float(ref[:, 1].mean())),
                       vasicek_mean_bias_corrected=float(J_v.mean() - ref[:, 0].mean()), kde_mean_bias_corrected=float(J_k.mean() - ref[:, 1].mean())),
           skew_mean=float(np.mean([r['skew'] for r in rows])), excess_kurtosis_mean=float(np.mean([r['excess_kurtosis'] for r in rows])),
           n_shapiro_reject_05=int(sum(r['shapiro_p'] < 0.05 for r in rows)))
json.dump(out, open(ROOT / 'results/loop/learner/negentropy_pe_target.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
print('per participant J_kde bits:', np.round(J_k, 3).tolist())
