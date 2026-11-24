"""RUNNER RF8 (C3, 24 Nov 2026): BCI-IV-2a two-session retest with a CALIBRATED torch read-out.
Replaces experiments/ext/bci2a_retest.py (kept unchanged; its load() is imported, its feats() is used only for a cross-check) for the bugs of RUNNER_FIXES items 7-9:
  7  floors for EVERY arm (eeg, eeg_eogreg, eog_only): noise input and shuffled labels;
  8  read-out: GPU multinomial softmax (mi_card_v3 exact-Newton, imported M.newton_logistic) with the L2 penalty chosen INSIDE the training session by 6-fold contiguous-block
     cross-validation, jointly with a temperature T (M.TGRID) calibrated on the out-of-fold logits of the training session (M.select = the mi_card_v3 rule); refit on the
     whole training session; the test session is touched only for scoring. No sklearn, no hand-picked C.
  9  EOG regression with an intercept; coefficients from the TRAIN session only (applied to the test session).
Data: data_raw/bci_iv_2a participants 2-9 (A01E.gdf corrupt), 22 EEG + 3 EOG, 250 Hz, window cue+0.5..2.5 s, rejected trials dropped as in bci2a_retest.load.
Features: log variance in 4 bands (4-8, 8-12, 12-20, 20-30 Hz) x channel, computed in torch (zero-phase Butterworth-4 magnitude response applied by FFT after odd-extension padding;
cross-checked against the scipy sosfiltfilt features of the old script). Bits per trial = log2 p(true class) - log2 train prior (M.bits, M.prior_of).
Arms: eeg | eeg_eogreg (3 recorded EOG channels regressed out, intercept) | eog_only. Directions: T_to_E, E_to_T.
Floors per arm and direction: noise_t = Gaussian white NOISE TIME SERIES (per-channel SD of the session; EEG and EOG independent) through the SAME pipeline (regression, filter, features,
  read-out), 5 seeds; noise_f = N(0,1) features of the arm's width (the old script's floor), 5 seeds; shuffled = train labels permuted, 5 seeds.
Planted positive control (eeg and eeg_eogreg arms): the noise-time input plus a class-specific 10 Hz sinusoid (amp 1.0 and weak 0.1 x channel SD (0.4 saturated at 2.0 bits in the smoke), random phase per trial and channel) on 5 channels
  per class (class k: channels 5k..5k+4); recovered if cohort mean >= 0.2 bits in both directions.
PRE-DECLARED GATE (fixed in the task before this run; never loosened): every cohort-mean blank (noise_t, noise_f, shuffled; every arm, both directions) within +-0.02 bits.
  If a blank is outside, it is reported and nothing is tuned. Further gates: n_T / n_E per participant equal the old run; read-out equals sklearn on one fit (selftest);
  feature extraction vs scipy reported (max abs and correlation).
Usage: python rf8_bci2a.py --out DIR [--subs 2,3,..] [--seeds 5] [--selftest]"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, sosfreqz
from scipy import stats as sst
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext'))
import mi_card_v3 as M
import bci2a_retest as OLD
DEV = 'cuda'; CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
FS = 250; BANDS = OLD.BANDS; PAD = 125; KF = 6; ARMS = ['eeg', 'eeg_eogreg', 'eog_only']; DIRS = ['T_to_E', 'E_to_T']; NT = 500


# ---------------------------------------------------------------- features (torch)
_H = {}
def resp(n):
    """|H(f)|^2 of the order-4 band-pass Butterworth (scipy design), on the rfft grid of length n -> [len(BANDS), n//2+1] GPU float64."""
    if n not in _H:
        f = np.fft.rfftfreq(n, 1 / FS); rows = []
        for lo, hi in BANDS:
            sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos'); _, h = sosfreqz(sos, worN=f, fs=FS); rows.append(np.abs(h) ** 2)
        _H[n] = torch.tensor(np.stack(rows), dtype=torch.float64, device=DEV)
    return _H[n]


def feats(E):
    """E [n, C, 500] float64 GPU -> [n, C*4] log-variance (band-major like bci2a_retest.feats): zero-phase filter by FFT with odd extension (PAD samples each side)."""
    n, C, T = E.shape; left = 2 * E[..., :1] - E[..., 1:PAD + 1].flip(-1); right = 2 * E[..., -1:] - E[..., -PAD - 1:-1].flip(-1)
    xp = torch.cat([left, E, right], -1); H = resp(xp.shape[-1]); Xf = torch.fft.rfft(xp, dim=-1); out = []
    for b in range(len(BANDS)):
        y = torch.fft.irfft(Xf * H[b], n=xp.shape[-1], dim=-1)[..., PAD:-PAD]; out.append(torch.log(y.var(-1, unbiased=False) + 1e-6))
    return torch.cat(out, 1)


def eogreg(Etr, Gtr, Ete, Gte):
    """Per-channel least squares of EEG on the 3 EOG series + intercept, coefficients from TRAIN trials only; residual for train and test."""
    def design(G): n, c, T = G.shape; return torch.cat([G.permute(0, 2, 1).reshape(-1, c), torch.ones(n * T, 1, dtype=G.dtype, device=G.device)], 1)
    Atr = design(Gtr); Btr = Etr.permute(0, 2, 1).reshape(-1, Etr.shape[1]); W = torch.linalg.lstsq(Atr, Btr).solution
    res = lambda E, G: (E.permute(0, 2, 1).reshape(-1, E.shape[1]) - design(G) @ W).reshape(E.shape[0], E.shape[2], E.shape[1]).permute(0, 2, 1).contiguous()
    return res(Etr, Gtr), res(Ete, Gte)


def arm_feats(arm, tr, te):
    (Etr, Gtr), (Ete, Gte) = tr, te
    if arm == 'eeg': return feats(Etr), feats(Ete)
    if arm == 'eeg_eogreg': Rtr, Rte = eogreg(Etr, Gtr, Ete, Gte); return feats(Rtr), feats(Rte)
    return feats(Gtr), feats(Gte)


# ---------------------------------------------------------------- calibrated read-out
def readout(Ftr, ytr, Fte, yte, lams=M.LAMS):
    """Penalty lambda and temperature T chosen on the out-of-fold logits of the TRAIN session (6 contiguous blocks, standardisation from the fold-train rows);
    refit on the whole train session; returns mean bits on the test session and the selection record."""
    n, d = Ftr.shape; L = len(lams); fold = (torch.arange(n, device=DEV) * KF) // n; nmax = n - int(torch.bincount(fold).min())
    Phi = torch.zeros(KF, nmax, d, dtype=torch.float64, device=DEV); Y = torch.zeros(KF, nmax, dtype=torch.long, device=DEV); Mk = torch.zeros(KF, nmax, dtype=torch.float64, device=DEV); EV = []
    for k in range(KF):
        tr = fold != k; A = Ftr[tr]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-8; m = len(A)
        Phi[k, :m] = (A - mu) / sd; Y[k, :m] = ytr[tr]; Mk[k, :m] = 1; EV.append(((Ftr[~tr] - mu) / sd, torch.where(~tr)[0]))
    lam = torch.tensor(lams, dtype=torch.float64, device=DEV).repeat(KF)                       # member order: fold-major, lambda-minor
    W, g = M.newton_logistic(Phi.repeat_interleave(L, 0), Y.repeat_interleave(L, 0), Mk.repeat_interleave(L, 0), lam); W = W.reshape(KF, L, d + 1, 4)
    Zo = torch.zeros(L, n, 4, dtype=torch.float64, device=DEV)
    for k, (Pe, idx) in enumerate(EV): Zo[:, idx] = torch.einsum('nd,ldk->lnk', torch.cat([Pe, torch.ones(len(Pe), 1, dtype=Pe.dtype, device=DEV)], 1), W[k])
    o, T, e = M.select(Zo, ytr)
    mu, sd = Ftr.mean(0), Ftr.std(0, unbiased=False) + 1e-8
    Wf, gf = M.newton_logistic(((Ftr - mu) / sd)[None], ytr[None], torch.ones(1, n, dtype=torch.float64, device=DEV), torch.tensor([lams[o]], dtype=torch.float64, device=DEV))
    Zt = torch.cat([(Fte - mu) / sd, torch.ones(len(Fte), 1, dtype=torch.float64, device=DEV)], 1) @ Wf[0]
    b = M.bits(Zt, yte, T, M.prior_of(ytr)); return float(b.mean()), dict(lam=float(lams[o]), T=T, lam_edge=bool(e['opt_edge']), T_edge=bool(e['T_edge']), grad_final=float(gf), grad_cv=float(g), acc=float((Zt.argmax(-1) == yte).double().mean()))


def selftest(Ftr, ytr, Fte, yte):
    from sklearn.linear_model import LogisticRegression
    lam = 0.05; n = len(ytr); mu, sd = Ftr.mean(0), Ftr.std(0, unbiased=False) + 1e-8
    Wf, _ = M.newton_logistic(((Ftr - mu) / sd)[None], ytr[None], torch.ones(1, n, dtype=torch.float64, device=DEV), torch.tensor([lam], dtype=torch.float64, device=DEV))
    Zt = torch.cat([(Fte - mu) / sd, torch.ones(len(Fte), 1, dtype=torch.float64, device=DEV)], 1) @ Wf[0]; p = torch.softmax(Zt, -1).cpu().numpy()
    clf = LogisticRegression(C=1 / (n * lam), tol=1e-12, max_iter=20000).fit(((Ftr - mu) / sd).cpu().numpy(), ytr.cpu().numpy()); q = clf.predict_proba(((Fte - mu) / sd).cpu().numpy())
    return float(np.abs(p - q).max())


# ---------------------------------------------------------------- data
def session(sub, ses):
    E, G, y, ok = OLD.load(sub, ses)
    return dict(E=torch.tensor(E[ok], dtype=torch.float64, device=DEV), G=torch.tensor(G[ok], dtype=torch.float64, device=DEV), y=torch.tensor(y[ok], device=DEV), n=int(ok.sum()), n_all=len(ok),
                E_np=E[ok], y_np=y[ok])


def noise_like(S, seed, plant=0.0):
    g = torch.Generator(device=DEV); g.manual_seed(seed); E, G = S['E'], S['G']
    sdE = E.std((0, 2), keepdim=True); sdG = G.std((0, 2), keepdim=True)
    En = torch.randn(E.shape, generator=g, device=DEV, dtype=torch.float64) * sdE; Gn = torch.randn(G.shape, generator=g, device=DEV, dtype=torch.float64) * sdG
    if plant:
        t = torch.arange(NT, device=DEV, dtype=torch.float64) / FS; ph = torch.rand(E.shape[0], E.shape[1], 1, generator=g, device=DEV, dtype=torch.float64) * 2 * math.pi
        wave = torch.sin(2 * math.pi * 10 * t[None, None] + ph); cls = S['y']; mask = torch.zeros(E.shape[0], E.shape[1], 1, dtype=torch.float64, device=DEV)
        for k in range(4): mask[cls == k, 5 * k:5 * k + 5] = 1
        En = En + plant * mask * wave * sdE
    return En, Gn


def run_sub(sub, nseed):
    S = {'T': session(sub, 'T'), 'E': session(sub, 'E')}; old = json.loads((ROOT / f'results/ext/bci2a_retest/A{sub:02d}.json').read_text())
    assert (S['T']['n'], S['E']['n']) == (old['n_T'], old['n_E']), ('trial counts differ from the old run', S['T']['n'], S['E']['n'], old['n_T'], old['n_E'])
    out = dict(sub=sub, n_T=S['T']['n'], n_E=S['E']['n'], n_gate_equal_old=True, code_sha256=CODE_SHA256)
    # feature cross-check against the scipy features of the old script (EEG arm, both sessions)
    d = {}
    for s in 'TE':
        a = feats(S[s]['E']).cpu().numpy(); b = OLD.feats(S[s]['E_np']); cc = np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]); d[s] = dict(max_abs=float(np.abs(a - b).max()), corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), col_corr_median=float(np.median(cc)), col_corr_min=float(cc.min()))
    out['feature_xcheck_vs_scipy'] = d
    ses = {'T_to_E': ('T', 'E'), 'E_to_T': ('E', 'T')}; N = {(s, k): noise_like(S[s], 100000 * sub + 10 * k + (0 if s == 'T' else 5)) for s in 'TE' for k in range(nseed)}
    PL = {(s, nm): noise_like(S[s], 777000 + 10 * sub + (0 if s == 'T' else 5), plant=amp) for s in 'TE' for nm, amp in (('planted', 1.0), ('planted_weak', 0.1))}
    for dn, (a, b) in ses.items():
        out[dn] = {}; yt, ye = S[a]['y'], S[b]['y']; real = ((S[a]['E'], S[a]['G']), (S[b]['E'], S[b]['G']))
        for arm in ARMS:
            r = {}; Ftr, Fte = arm_feats(arm, *real); r['real'], r['real_sel'] = readout(Ftr, yt, Fte, ye)
            nt, nf, sh = [], [], []
            for k in range(nseed):
                Fa, Fb = arm_feats(arm, N[(a, k)], N[(b, k)]); nt.append(readout(Fa, yt, Fb, ye)[0])
                g = torch.Generator(device=DEV); g.manual_seed(5000 + 100 * sub + k); nf.append(readout(torch.randn(Ftr.shape, generator=g, device=DEV, dtype=torch.float64), yt, torch.randn(Fte.shape, generator=g, device=DEV, dtype=torch.float64), ye)[0])
                ys = yt[torch.randperm(len(yt), generator=g, device=DEV)]; sh.append(readout(Ftr, ys, Fte, ye)[0])
            r['noise_t'], r['noise_f'], r['shuffled'] = nt, nf, sh
            if arm in ('eeg', 'eeg_eogreg'):
                for nm in ('planted', 'planted_weak'): Pa, Pb = arm_feats(arm, PL[(a, nm)], PL[(b, nm)]); r[nm] = readout(Pa, yt, Pb, ye)[0]
            out[dn][arm] = r
        # scipy-feature cross-check of the new read-out (eeg arm, real data): same read-out, features from the old script
        Fa = torch.tensor(OLD.feats(S[a]['E_np']), device=DEV, dtype=torch.float64); Fb = torch.tensor(OLD.feats(S[b]['E_np']), device=DEV, dtype=torch.float64)
        out[dn]['xcheck_scipy_features_eeg'] = readout(Fa, yt, Fb, ye)[0]
    out['old_run'] = {dn: {k: old[dn][k] for k in ('eeg', 'noise', 'shuffled', 'eeg_eogreg', 'eog_only')} for dn in DIRS}
    return out, S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default='2,3,4,5,6,7,8,9'); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); subs = [int(x) for x in a.subs.split(',')]; assert 1 not in subs
    for s in subs:
        f = out / f'A{s:02d}.json'
        if f.exists() and json.loads(f.read_text()).get('code_sha256') == CODE_SHA256: log(s, 'exists'); continue
        r, S = run_sub(s, a.seeds)
        if a.selftest:
            Ftr, Fte = arm_feats('eeg', (S['T']['E'], S['T']['G']), (S['E']['E'], S['E']['G'])); r['selftest_max_abs_prob_diff_vs_sklearn'] = selftest(Ftr, S['T']['y'], Fte, S['E']['y'])
        f.write_text(json.dumps(r, indent=1)); log(s, 'done', {dn: {arm: round(r[dn][arm]['real'], 3) for arm in ARMS} for dn in DIRS}, 'xcheck', r['feature_xcheck_vs_scipy']['T'])
    summarize(out, subs, a.seeds)


def summarize(out, subs, nseed):
    R = [json.loads((out / f'A{s:02d}.json').read_text()) for s in subs]; n = len(R)
    T = lambda v: torch.tensor(np.asarray(v, np.float64), device=DEV); gb = torch.Generator(device=DEV); gb.manual_seed(0); IX = torch.randint(0, n, (10000, n), generator=gb, device=DEV)
    def stat(v):
        v = np.asarray(v, np.float64); m = T(v)[IX].mean(1); sd = v.std(ddof=1) / math.sqrt(len(v)); q = sst.t.ppf(0.975, len(v) - 1)
        return dict(mean=float(v.mean()), boot_ci95=[float(m.quantile(0.025)), float(m.quantile(0.975))], t_ci95=[float(v.mean() - q * sd), float(v.mean() + q * sd)], n_pos=int((v > 0).sum()), n=len(v),
                    n_abs_le_0p02=int((np.abs(v) <= 0.02).sum()))
    vals = lambda dn, arm, key: [float(np.mean(r[dn][arm][key])) if isinstance(r[dn][arm][key], list) else r[dn][arm][key] for r in R]
    S = dict(NOT_CITABLE='until a blind review', code_sha256=CODE_SHA256, n=n, participants=subs, note='A01 excluded: A01E.gdf corrupt', seeds=nseed, cells={}, contrasts={}, gate={})
    for dn in DIRS:
        for arm in ARMS:
            keys = ['real', 'noise_t', 'noise_f', 'shuffled'] + (['planted', 'planted_weak'] if arm != 'eog_only' else [])
            S['cells'][f'{dn}|{arm}'] = {k: stat(vals(dn, arm, k)) for k in keys}
            S['cells'][f'{dn}|{arm}']['real_minus_noise_t'] = stat(np.array(vals(dn, arm, 'real')) - np.array(vals(dn, arm, 'noise_t')))
            S['cells'][f'{dn}|{arm}']['real_minus_shuffled'] = stat(np.array(vals(dn, arm, 'real')) - np.array(vals(dn, arm, 'shuffled')))
            S['cells'][f'{dn}|{arm}']['old_run_yield_mean'] = float(np.mean([r['old_run'][dn][{'eeg': 'eeg', 'eeg_eogreg': 'eeg_eogreg', 'eog_only': 'eog_only'}[arm]] for r in R]))
            S['cells'][f'{dn}|{arm}']['old_run_noise_shuffled_mean'] = [float(np.mean([r['old_run'][dn]['noise'] for r in R])), float(np.mean([r['old_run'][dn]['shuffled'] for r in R]))]
    for arm in ARMS:
        a_, b_ = np.array(vals('T_to_E', arm, 'real')), np.array(vals('E_to_T', arm, 'real')); S['contrasts'][f'{arm}|corr_T2E_E2T'] = float(np.corrcoef(a_, b_)[0, 1])
        S['contrasts'][f'{arm}|direction_mean'] = stat((a_ + b_) / 2)
    for dn in DIRS + ['direction_mean']:
        g = (lambda arm, k: np.array(vals('T_to_E', arm, k)) / 2 + np.array(vals('E_to_T', arm, k)) / 2) if dn == 'direction_mean' else (lambda arm, k: np.array(vals(dn, arm, k)))
        S['contrasts'][f'eogreg_minus_eeg|{dn}'] = stat(g('eeg_eogreg', 'real') - g('eeg', 'real'))
        S['contrasts'][f'eog_only_minus_noise_t|{dn}'] = stat(g('eog_only', 'real') - g('eog_only', 'noise_t'))
    # gate: every cohort-mean blank within +-0.02 bits, every arm and direction
    bad = {}; worst = 0.0
    for c, d in S['cells'].items():
        for k in ('noise_t', 'noise_f', 'shuffled'):
            m = d[k]['mean']; worst = max(worst, abs(m))
            if abs(m) > 0.02: bad[f'{c}|{k}'] = m
    pers = {f'{c}|{k}': d[k]['n_abs_le_0p02'] for c, d in S['cells'].items() for k in ('noise_t', 'noise_f', 'shuffled')}
    S['gate'] = dict(blanks_inside_0p02_all=len(bad) == 0, worst_abs_cohort_blank=worst, offending=bad, participants_within_0p02_by_cell=pers,
                     n_equal_old_all=all(r['n_gate_equal_old'] for r in R), feature_xcheck_max_abs=max(max(r['feature_xcheck_vs_scipy'][s]['max_abs'] for s in 'TE') for r in R),
                     feature_xcheck_min_corr=min(min(r['feature_xcheck_vs_scipy'][s]['corr'] for s in 'TE') for r in R),
                     feature_xcheck_column_corr_min=min(min(r['feature_xcheck_vs_scipy'][s]['col_corr_min'] for s in 'TE') for r in R),
                     feature_xcheck_column_corr_median_min=min(min(r['feature_xcheck_vs_scipy'][s]['col_corr_median'] for s in 'TE') for r in R),
                     selftest_max_abs_prob_diff_vs_sklearn=[r.get('selftest_max_abs_prob_diff_vs_sklearn') for r in R])
    S['planted_recovered_ge_0.2_both_dirs'] = bool(all(S['cells'][f'{dn}|{arm}']['planted']['mean'] >= 0.2 for dn in DIRS for arm in ('eeg', 'eeg_eogreg')))
    S['xcheck_scipy_features_eeg_mean'] = {dn: float(np.mean([r[dn]['xcheck_scipy_features_eeg'] for r in R])) for dn in DIRS}
    S['selection'] = 'lambda and T chosen on training-session OOF logits only; see per-participant real_sel'
    S['runtime_s'] = round(time.time() - t0); (out / 'summary.json').write_text(json.dumps(S, indent=1)); log('SUMMARY', json.dumps({k: S[k] for k in ('gate', 'planted_recovered_ge_0.2_both_dirs', 'xcheck_scipy_features_eeg_mean')}, indent=1)[:3000])
    for c, d in S['cells'].items(): log(c, {k: round(v['mean'], 3) for k, v in d.items() if isinstance(v, dict) and 'mean' in v})


if __name__ == '__main__':
    main()
