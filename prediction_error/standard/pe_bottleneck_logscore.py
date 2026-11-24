"""Standard test, PE bottleneck under the common log-score (ds003458, 2026-11-20).
Read-out = raw-ERP within-participant ridge (q1_pe_yield_bottleneck.py: 64ch x 23 samples = 1472 cols, in-fold z-score * colw, ridge LAM)
trained on the participant's OWN residualised PE. Bottleneck: in-fold PCA-L or fixed random orthonormal projection (seed 0), L in LS, 'full'.
Score = common_log_yield.density_gain: held-out log2 Gaussian density of the PE given a linear calibration of the read-out minus
intercept-only density, both fit on training rows only. Nested: in outer fold b the calibration is fit on inner-CV (9 contiguous folds)
read-out predictions of the training rows; the test rows get the read-out of the ridge trained on all training rows. No test target touches
calibration or ridge. Floors = Gaussian-noise input (seed 1000+s, as q1); the same pipeline refit with the target circularly shifted by n/4, n/2, 3n/4 trials (time-structure-preserving).
Output: results/standard/pe_bottleneck/sub-XXX.json and summary.json (L90 = smallest L whose participant-mean gain >= 90% of the max over L;
participant bootstrap CI, 10000 resamples, seed 20261120)."""
import sys, json, time, numpy as np, torch
from pathlib import Path
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/agent_loop_20261120'))
from belief_bottleneck import LAM, folds
from i2_complementarity import targets
from common_log_yield import density_gain
from scipy import stats
nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))   # as s13b_gauss_bits.py / q1
OUT = ROOT / 'results/standard/pe_bottleneck'; OUT.mkdir(parents=True, exist_ok=True)
LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 'full']; DEV = 'cuda'; LN2 = np.log(2)


def ridge_preds(F, Y, tr, te, colw, R):
    """Returns {(mode,L): (n_te x T) predictions} for ridge trained on rows tr."""
    A, B = F[tr], F[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * colw; B = (B - mu) / sd * colw
    Yc = Y[tr] - Y[tr].mean(0); _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False); out = {}
    for mode in ('pca', 'random'):
        for L in LS:
            if L == 'full':
                if mode == 'random': continue
                Atr, Ate = A, B
            else:
                Vk = Vt[:min(L, Vt.shape[0])].T if mode == 'pca' else R[:, :L]; Atr, Ate = A @ Vk, B @ Vk
            k = Atr.shape[1]
            W = torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ Yc)
            out[(mode, L)] = Ate @ W
    return out


def gain_rows(y_tr, x_tr, y_te, x_te):
    """Per-test-row log2 density gain, calibration fit on (x_tr, y_tr) only (as common_log_yield.density_gain)."""
    D = np.column_stack([np.ones(len(x_tr)), x_tr]); c = np.linalg.lstsq(D, y_tr, rcond=None)[0]
    pt, pe = D @ c, c[0] + c[1] * x_te; rv = np.mean((y_tr - pt) ** 2); m = y_tr.mean(); vv = np.mean((y_tr - m) ** 2)
    return (np.log(vv / rv) + (y_te - m) ** 2 / vv - (y_te - pe) ** 2 / rv) / (2 * LN2)


def run_subject(s, evaluate, fpath):
    fz = np.load(fpath('raw', s)); tri = fz['tri']; n = len(tri)
    y = evaluate(s, tri).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * k / 4)) for k in (1, 2, 3)])
    F = torch.tensor(fz['F'].astype(np.float64), device=DEV); cw = torch.tensor(fz['colw'].astype(np.float64), device=DEV)
    Y = torch.tensor(Ys, device=DEV); R = torch.tensor(np.linalg.qr(np.random.default_rng(0).standard_normal((F.shape[1], F.shape[1])))[0], dtype=torch.float64, device=DEV)
    Fn = torch.tensor(np.random.default_rng(1000 + s).standard_normal(tuple(F.shape)), device=DEV)   # noise-input control (as q1)
    fo = folds(n); scores = {}; Pout = {}
    for inp, X in (('eeg', F), ('noise', Fn)):
        for b in range(10):
            tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; assert not np.intersect1d(tr, te).size
            outer = ridge_preds(X, Y, torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV), cw, R)
            inner = {key: np.zeros((len(tr), 4)) for key in outer}; fi = folds(len(tr), 9)
            for c in range(9):
                itr, ite = tr[fi != c], tr[fi == c]; assert np.isin(ite, tr).all() and not np.isin(ite, te).any()
                p = ridge_preds(X, Y, torch.tensor(itr, device=DEV), torch.tensor(ite, device=DEV), cw, R)
                for key in p: inner[key][fi == c] = p[key].cpu().numpy()
            for key, P in outer.items():
                P = P.cpu().numpy(); Pout.setdefault((inp,) + key, np.full(n, np.nan))[te] = P[:, 0]
                for j in range(4):
                    scores.setdefault((inp,) + key, np.full((n, 4), np.nan))[te, j] = gain_rows(Ys[tr, j], inner[key][:, j], Ys[te, j], P[:, j])
    rows = []
    for (inp, mode, L), sc in scores.items():
        assert np.isfinite(sc).all()
        if inp == 'noise' and mode == 'random': continue
        rows.append(dict(input=inp, mode=mode, L=L, gain=float(sc[:, 0].mean()), shifted=[float(v) for v in sc[:, 1:].mean(0)]))
    # check A: per-fold gain_rows == common_log_yield.density_gain on the cached EEGNet read-out (same folds, non-nested)
    ez = np.load(fpath('eegnet', s)); xe, te_ = ez['F'].astype(float), ez['tri']; ye = evaluate(s, te_); fe = folds(len(ye)); ge = np.full(len(ye), np.nan)
    for b in range(10): ge[fe == b] = gain_rows(ye[fe != b], xe[fe != b, 0], ye[fe == b], xe[fe == b, 0])
    common = {r['subject']: r['gain'] for r in json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/common_log_yield.json').read_text())['behavioral']['rows']['eegnet']}
    # check B: outer-fold ridge read-out at PCA-64 reproduces q1/X7 r_g = corr(read-out, normal-scored PE)
    q1 = json.loads((ROOT / f'results/yield/pe_bottleneck_ds003458/raw_sub-{s:03d}.json').read_text())
    q1r = next(r['r_g'] for r in q1['rows'] if r['L'] == 64 and r['condition'] == 'eeg')
    checks = dict(eegnet_density_gain=float(ge.mean()), eegnet_common_log_yield=common[s], eegnet_abs_diff=abs(float(ge.mean()) - common[s]),
                  pca64_r_g=float(np.corrcoef(Pout[('eeg', 'pca', 64)], nscore(y))[0, 1]), q1_pca64_r_g=q1r,
                  pca64_r_g_abs_diff=abs(float(np.corrcoef(Pout[('eeg', 'pca', 64)], nscore(y))[0, 1]) - q1r),
                  fold_rule='standardization, PCA SVD, ridge and calibration fit on training rows only; calibration on inner-CV read-outs of training rows; asserted disjoint')
    return dict(dataset='ds003458', participant=f'sub-{s:03d}', n_trials=int(n), full_dim=int(F.shape[1]), unit='bits per feedback trial', rows=rows, checks=checks)


def boot_L90(G, rng, B=10000):
    """G: participants x widths gain matrix. L90 index of the bootstrapped mean curve."""
    ix = rng.integers(0, len(G), size=(B, len(G))); M = G[ix].mean(1); mx = M.max(1, keepdims=True)
    return (M >= 0.9 * mx).argmax(1)


def main():
    smoke = '--smoke' in sys.argv; evaluate, subs, fpath, _ = targets(); t0 = time.time()
    for s in (subs[:1] if smoke else subs):
        dest = OUT / (f'smoke_sub-{s:03d}.json' if smoke else f'sub-{s:03d}.json')
        if dest.exists(): continue
        r = run_subject(s, evaluate, fpath); r['runtime_s'] = round(time.time() - t0, 1); dest.write_text(json.dumps(r, indent=1))
        g = {(x['input'], x['mode'], x['L']): x for x in r['rows']}
        print(r['participant'], ' '.join(f"L{L}:{g[('eeg', 'pca', L)]['gain']:.3f}" for L in LS), 'shift', np.round(g[('eeg', 'pca', 'full')]['shifted'], 4),
              'noise_full', round(g[('noise', 'pca', 'full')]['gain'], 4), {k: (round(v, 8) if isinstance(v, float) else '') for k, v in r['checks'].items()}, f'{time.time()-t0:.0f}s', flush=True)
    if smoke: return
    per = [json.loads((OUT / f'sub-{s:03d}.json').read_text()) for s in subs]
    def mat(mode, what, inp='eeg'):
        Lm = LS if mode == 'pca' else LS[:-1]
        row = lambda p, L: next(x for x in p['rows'] if x['input'] == inp and x['mode'] == mode and x['L'] == L)
        return np.array([[row(p, L)['gain'] if what == 'gain' else np.mean(row(p, L)['shifted']) for L in Lm] for p in per])
    rng = np.random.default_rng(20261120); G = mat('pca', 'gain'); S = mat('pca', 'shift'); Rn = mat('random', 'gain'); Nz = mat('pca', 'gain', 'noise')
    def ci(v):
        ix = rng.integers(0, len(v), size=(10000, len(v))); return np.quantile(v[ix].mean(1), [.025, .975]).tolist()
    curve = {}
    for i, L in enumerate(LS):
        d = G[:, i] - S[:, i]
        curve[str(L)] = dict(pca_gain=float(G[:, i].mean()), pca_gain_ci=ci(G[:, i]), shift_floor=float(S[:, i].mean()), gain_minus_shift=float(d.mean()), gain_minus_shift_ci=ci(d),
                             n_above_shift=int((d > 0).sum()), noise_gain=float(Nz[:, i].mean()), noise_gain_ci=ci(Nz[:, i]), random_gain=float(Rn[:, i].mean()) if L != 'full' else None, random_gain_ci=ci(Rn[:, i]) if L != 'full' else None)
    m = G.mean(0); L90 = LS[int((m >= 0.9 * m.max()).argmax())]; bi = boot_L90(G, rng); Lnum = np.array([L if L != 'full' else 1472 for L in LS])
    pp = [LS[int((g >= 0.9 * g.max()).argmax())] if g.max() > 0 else None for g in G]
    res = dict(note=__doc__, n=len(per), widths=LS, curve=curve, plateau_gain=float(m.max()), plateau_L=LS[int(m.argmax())], L90=L90, L90_num=int(Lnum[LS.index(L90)]),
               L90_ci=[int(np.quantile(Lnum[bi], .025)), int(np.quantile(Lnum[bi], .975))], L90_bootstrap_mode_share=float(np.mean(Lnum[bi] == Lnum[LS.index(L90)])),
               per_participant_L90={p['participant']: v for p, v in zip(per, pp)}, full_width_gain=float(m[-1]), full_width_shift_floor=float(S[:, -1].mean()),
               full_width_gain_minus_shift_ci=curve['full']['gain_minus_shift_ci'], full_width_n_above_shift=curve['full']['n_above_shift'], runtime_s=round(time.time() - t0, 1),
               checks=dict(max_eegnet_abs_diff=max(p['checks']['eegnet_abs_diff'] for p in per), max_pca64_r_g_abs_diff=max(p['checks']['pca64_r_g_abs_diff'] for p in per),
                           fold_rule=per[0]['checks']['fold_rule']))
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    for L in LS: print(L, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in curve[str(L)].items() if not k.endswith('ci')})
    print('L90', L90, res['L90_ci'], 'plateau', round(m.max(), 4))


if __name__ == '__main__':
    main()
