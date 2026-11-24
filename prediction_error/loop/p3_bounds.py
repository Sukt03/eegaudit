"""P3: three information estimates for the prediction-error target, per participant (ds003458, 23), per decoder.
X_g = normal-scores of the residualised PE (exactly N(0,1) marginally); Y = read-out: cached LOPO single column (eegnet, reve_lopo, cbramod_ft)
or, for raw ERP, the exact per-participant CV PCA-64 ridge prediction (cv_predict_fast of s13b, trained on the raw target). Y is z-scored within participant.
(a) Gaussian channel: -1/2 log2(1 - corr(Y, X_g)^2) (per participant, uncorrected here; the null-corrected version is in S13b).
(b) Barber-Agakov: q(x|y) = N(a + b y, s^2) with (a, b, s) fitted by least squares on the pooled (y, x_g) of the OTHER 22 participants;
    I_BA = 1/2 log2(2 pi e) - mean_t[-log2 q(x_t | y_t)] (h(X_g) is exactly that of N(0,1)), clipped at 0; floor = same with y circularly shifted
    by >= 40 trials (20 shifts), reported as the mean floor and as I_BA minus floor.
(c) KSG (Kraskov et al. 2004, estimator 1, k = 5, Chebyshev) on (y, x_g) per participant, minus the mean of 20 trial permutations, in bits.
Outputs results/loop/learner/p3_bounds.json: per decoder mean, participant-bootstrap CI, CI width / mean, per-participant values; decoder orders."""
import sys, json, numpy as np, torch
from pathlib import Path
from scipy import stats, special
from scipy.spatial import cKDTree
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT))
import deep_decoders as DD
src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); ns = {}
exec('import torch, numpy as np\nimport sys; sys.path.insert(0, "/path/to/eegaudit")\n' + src[src.index('from belief_bottleneck import LAM, folds'):src.index('SMOKE =')], ns); cv_predict_fast = ns['cv_predict_fast']
data, _ = DD.load_all(); rng = np.random.default_rng(0); nS = len(data)
nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y)); z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
hX = 0.5 * np.log2(2 * np.pi * np.e)
def ksg(x, y, k=5):
    n = len(x); pts = np.c_[x, y]; d = cKDTree(pts).query(pts, k + 1, p=np.inf)[0][:, -1]
    nx = np.array([len(cKDTree(x[:, None]).query_ball_point([xi], r - 1e-12, p=np.inf)) - 1 for xi, r in zip(x, d)])
    ny = np.array([len(cKDTree(y[:, None]).query_ball_point([yi], r - 1e-12, p=np.inf)) - 1 for yi, r in zip(y, d)])
    return (special.digamma(k) + special.digamma(n) - np.mean(special.digamma(nx + 1) + special.digamma(ny + 1))) / np.log(2)
def ba(xg, y, a, b_, s): return float(max(0.0, hX - np.mean(-stats.norm.logpdf(xg, a + b_ * y, s) / np.log(2))))
readouts = {}
for dec in ['eegnet', 'reve_lopo', 'cbramod_ft', 'raw']:
    Ys = []
    for i, (X, yt, tri) in enumerate(data):
        zf = np.load(ROOT / f'results/features/{dec}/sub-{i + 1:03d}.npz'); F = zf['F'].astype(float)
        assert np.array_equal(zf['tri'], tri)
        Ys.append(z(F[:, 0]) if F.shape[1] == 1 else z(cv_predict_fast(F, yt[:, None].astype(float), 64, colw=zf['colw'].astype(float))[:, 0]))
    readouts[dec] = Ys
Xg = [nscore(yt) for _, yt, _ in data]
out = {'n_participants': nS, 'decoders': {}}
def ci(v): b = [np.mean(np.asarray(v)[rng.integers(0, len(v), len(v))]) for _ in range(4000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for dec, Ys in readouts.items():
    A, B, Bfl, C = [], [], [], []
    for i in range(nS):
        y, xg = Ys[i], Xg[i]; A.append(float(-0.5 * np.log2(1 - min(abs(np.corrcoef(y, xg)[0, 1]), 0.999) ** 2)))
        yo = np.concatenate([Ys[j] for j in range(nS) if j != i]); xo = np.concatenate([Xg[j] for j in range(nS) if j != i])
        b_, a = np.polyfit(yo, xo, 1); s = np.std(xo - (a + b_ * yo))
        B.append(ba(xg, y, a, b_, s)); n = len(y)
        Bfl.append(float(np.mean([ba(xg, np.roll(y, sh), a, b_, s) for sh in rng.integers(40, n - 40, 20)])))
        C.append(float(ksg(y, xg) - np.mean([ksg(y[rng.permutation(n)], xg) for _ in range(20)])))
    A, B, Bfl, C = map(np.array, (A, B, Bfl, C)); Bc = B - Bfl
    out['decoders'][dec] = dict(gauss=dict(mean=float(A.mean()), ci95=ci(A), per_participant=A.round(4).tolist()),
                                ba=dict(mean=float(B.mean()), ci95=ci(B), floor_mean=float(Bfl.mean()), minus_floor_mean=float(Bc.mean()), minus_floor_ci95=ci(Bc), per_participant=B.round(4).tolist(), floor_per_participant=Bfl.round(4).tolist()),
                                ksg=dict(mean=float(C.mean()), ci95=ci(C), per_participant=C.round(4).tolist()))
    for k in ('gauss', 'ba', 'ksg'):
        d = out['decoders'][dec][k]; d['ci_width_over_mean'] = float((d['ci95'][1] - d['ci95'][0]) / max(d['mean'], 1e-9))
    print(dec, {k: (round(v['mean'], 4), [round(x, 4) for x in v['ci95']], round(v['ci_width_over_mean'], 2)) for k, v in out['decoders'][dec].items()}, 'BA floor', round(out['decoders'][dec]['ba']['floor_mean'], 4), flush=True)
order = lambda k: sorted(out['decoders'], key=lambda d: -out['decoders'][d][k]['mean'])
out['order'] = {k: order(k) for k in ('gauss', 'ba', 'ksg')}; out['order_agree_gauss_ba'] = out['order']['gauss'] == out['order']['ba']
e = out['decoders']['eegnet']; out['headline'] = 'ba' if e['ba']['ci_width_over_mean'] < e['gauss']['ci_width_over_mean'] else 'gauss'
out['ksg_sanity_eegnet'] = bool(e['ksg']['mean'] >= e[out['headline']]['mean'] - 0.01)
json.dump(out, open(ROOT / 'results/loop/learner/p3_bounds.json', 'w'), indent=1)
print('orders', out['order'], 'agree', out['order_agree_gauss_ba'], 'headline', out['headline'], 'ksg sanity', out['ksg_sanity_eegnet'])
