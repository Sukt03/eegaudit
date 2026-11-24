"""A5 (IDEATOR_LIFT_ABOVE_5_20261124): prediction-error carry-over control on ds003458, RIDGE READ-OUT ONLY (registered raw-ERP ridge, full width, results/features/raw,
150-600 ms, the 0.094 bit yield of results/standard/summary.json). NOT EEGNet; nothing here transfers to EEGNet.
Pre-registration: results/ext/a5_pe_carryover/PREREG.json (+ PREREG_STAMP.json), stamped before any scored output.
Three checks on the real data (rules copied from the ideator in PREREG): (i) block-permutation null (blocks of 20 trials, 200 permutations), (ii) lag profile Y(k)/(Y(0) r_k^2) for
k = -2,-1,+1,+2, (iii) previous-trial outcome/choice/PE regressed out in-fold, kept share. Positive/negative controls on PLANTED cells built on the same real features
(slow AR(1)=0.95 state in PCs 20-51 with target = 0.7 state + noise: lag flag must fire; event-locked plant with an AR(0.5) target: flag must stay off).
Usage: python -I -u a5_pe_carryover.py --out DIR [--subjects 1,2]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import d2_pe_tasks as D2
import pe_eogreg as PEG
DEV = 'cuda'; LS = ('full',)
NPERM, BLOCK, LAGS, BOOT, RATIO_MAX, KEEP_MIN, MIN_DET, NREG = 200, 20, [-2, -1, 1, 2], 10000, 1.5, 0.5, 21, 3
AMP, PC_LO, PC_HI, PHI_SLOW, PHI_EVENT, TARGET_LOAD = 0.7, 20, 52, 0.95, 0.5, 0.7
FLAG_MIN, OFF_MIN, PLANT_Y0_MIN, GATE = 20, 20, 0.02, 1e-4


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-300))


def ar1(phi, T, rng):
    e = rng.standard_normal(T); s = np.zeros(T); s[0] = e[0]
    for t in range(1, T): s[t] = phi * s[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return s


def block_perm(y, rng):
    n = len(y); nb = (n + BLOCK - 1) // BLOCK; blocks = [y[i * BLOCK:(i + 1) * BLOCK] for i in range(nb)]; return np.concatenate([blocks[i] for i in rng.permutation(nb)])


def mean_gain(F, Ys, transform=None):
    return D2.pe_gain(F, Ys, LS, transform=transform).mean(0)


def analyse(F, y, tri, s, Qc, Qr, seed):
    """real or planted cell for one participant: lag-0 yield with 3 circular-shift floors, noise-input floor, 200 block-permutation nulls, lag profile, previous-trial-regressed arm."""
    n = len(y); rng = np.random.default_rng(seed); out = {}
    shifts = [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)]; perms = [block_perm(y, rng) for _ in range(NPERM)]
    g = mean_gain(F, np.column_stack([y] + shifts + perms)); out.update(Y0=float(g[0]), shifted=[float(v) for v in g[1:4]], block_perm_null=[float(v) for v in g[4:]])
    out['noise_input'] = float(mean_gain(D2.gnoise(F.shape, 1000 + s), y[:, None])[0])
    nl = np.array(out['block_perm_null']); out['perm_p95'] = float(np.quantile(nl, 0.95)); out['exceeds_perm_p95'] = bool(out['Y0'] > out['perm_p95']); out['perm_p'] = float((1 + (nl >= out['Y0']).sum()) / (1 + len(nl)))
    pos = {int(t): i for i, t in enumerate(tri)}; out['lag'] = {}
    for k in LAGS:
        src = np.array([i for i, t in enumerate(tri) if int(t) + k in pos]); dst = np.array([pos[int(tri[i]) + k] for i in src])
        y0, yk = y[src], y[dst]; gk = mean_gain(F[src], np.column_stack([y0, yk])); r = corr(y0, yk)
        out['lag'][str(k)] = dict(n=int(len(src)), Y0_rows=float(gk[0]), Yk=float(gk[1]), r=r, r2=r * r, denom=float(gk[0] * r * r))
    # previous-trial control: outcome, chosen arm (2 dummies), outcome x arm, previous-trial target value, missing flag; regressed out in-fold (batched lstsq, pe_eogreg.eogreg)
    prev = np.array([int(t) - 1 for t in tri]); has = prev >= 0; pc = np.maximum(prev, 0)
    rp, cp = Qr[pc].astype(float), Qc[pc]; yprev = np.array([y[pos[int(t) - 1]] if (int(t) - 1) in pos else 0.0 for t in tri])
    O = np.c_[rp, cp == 0, cp == 1, rp * (cp == 0), rp * (cp == 1), yprev, ~has | np.array([(int(t) - 1) not in pos for t in tri])].astype(np.float64)
    gp = mean_gain(F, np.column_stack([y] + shifts), transform=PEG.eogreg(O)); out['prev'] = dict(Y=float(gp[0]), shifted=[float(v) for v in gp[1:]], floor=float(gp[1:].mean()))
    pn = [float(mean_gain(F, y[:, None], transform=PEG.eogreg(PEG.permuted(O, 500 + j)))[0]) for j in range(NREG)]
    out['prev']['permuted_regressor_Y'] = pn
    return out


def plant(F, tri, kind, seed):
    """F (n, D) real features -> planted features and the synthetic target (z-scored). kind 'slow': AR(1)=0.95 state s_t on the TRIAL axis added along a unit direction in PCs 20-51
    (amplitude AMP x the mean PC-score SD of PCs 20-51), target = 0.7 s + sqrt(1-0.49) noise. kind 'event': target AR(0.5) on the trial axis, the SAME pattern times the target (event-locked)."""
    rng = np.random.default_rng(seed); Ft = torch.as_tensor(F, dtype=torch.float64, device=DEV); n = Ft.shape[0]; sd = Ft.std(0) + 1e-9
    Z = (Ft - Ft.mean(0)) / sd; U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
    w = torch.as_tensor(rng.standard_normal(PC_HI - PC_LO), dtype=torch.float64, device=DEV); u = w @ Vt[PC_LO:PC_HI]; u = u / u.norm(); lam = float((S[PC_LO:PC_HI] / np.sqrt(n)).mean())
    T = int(tri.max()) + 1
    if kind == 'slow':
        st = ar1(PHI_SLOW, T, rng)[tri]; st = (st - st.mean()) / st.std(); sig = st; y = TARGET_LOAD * st + np.sqrt(1 - TARGET_LOAD ** 2) * rng.standard_normal(n)
    else:
        yy = ar1(PHI_EVENT, T, rng)[tri]; y = (yy - yy.mean()) / yy.std(); sig = y
    y = (y - y.mean()) / y.std()
    Fp = Ft + (AMP * lam * torch.as_tensor(sig, dtype=torch.float64, device=DEV))[:, None] * (u * sd)[None]
    return Fp.cpu().numpy(), y, dict(lambda_pc_sd=lam, amp=AMP)


def lag_flag_participant(r):
    num = sum(r['lag'][str(k)]['Yk'] for k in LAGS); den = sum(r['lag'][str(k)]['denom'] for k in LAGS)
    return dict(ratio=float(num / den) if den > 0 else None, flagged=bool(den > 0 and num / den > RATIO_MAX))


def boot_idx(n):
    g = torch.Generator(device=DEV); g.manual_seed(0); return torch.randint(n, (BOOT, n), generator=g, device=DEV)


def cohort(per):
    """per: list of participant dicts. Returns the pre-registered cohort statistics."""
    n = len(per); IX = boot_idx(n); T = lambda v: torch.tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); q = lambda x: [float(x.quantile(.025)), float(x.quantile(.975))]
    Y0 = T([p['Y0'] for p in per]); fl = T([np.mean(p['shifted']) for p in per]); res = dict(n=n, Y0_mean=float(Y0.mean()), Y0_ci=q(Y0[IX].mean(1)), shift_floor_mean=float(fl.mean()), noise_floor_mean=float(T([p['noise_input'] for p in per]).mean()))
    res['i_block_perm'] = dict(n_exceed=int(sum(p['exceeds_perm_p95'] for p in per)), n=n, need=MIN_DET, passed=bool(sum(p['exceeds_perm_p95'] for p in per) >= MIN_DET), perm_p_median=float(np.median([p['perm_p'] for p in per])))
    res['ii_lag'] = {}
    for k in LAGS:
        Yk = T([p['lag'][str(k)]['Yk'] for p in per]); den = T([p['lag'][str(k)]['denom'] for p in per]); r2 = T([p['lag'][str(k)]['r2'] for p in per])
        pt = float(Yk.mean() / den.mean()); b = Yk[IX].mean(1) / den[IX].mean(1); ci = q(b)
        res['ii_lag'][str(k)] = dict(ratio_of_means=pt, ci95=ci, upper=ci[1], lower=ci[0], Yk_mean=float(Yk.mean()), Yk_ci=q(Yk[IX].mean(1)), denom_mean=float(den.mean()), r2_mean=float(r2.mean()),
                                     ok_upper_le_1p5=bool(ci[1] <= RATIO_MAX), flag_lower_gt_1p5=bool(ci[0] > RATIO_MAX))
    res['ii_passed'] = all(v['ok_upper_le_1p5'] for v in res['ii_lag'].values()); res['ii_flag_slow_state'] = any(v['flag_lower_gt_1p5'] for v in res['ii_lag'].values())
    A = T([p['Y0'] - np.mean(p['shifted']) for p in per]); E = T([p['prev']['Y'] - p['prev']['floor'] for p in per]); PN = T([np.mean(p['prev']['permuted_regressor_Y']) - p['prev']['floor'] for p in per])
    kept = float(E.mean() / A.mean()); res['iii_prev'] = dict(kept_share=kept, ci95=q(E[IX].mean(1) / A[IX].mean(1)), need=KEEP_MIN, passed=bool(kept >= KEEP_MIN), Y_prev_mean=float(T([p['prev']['Y'] for p in per]).mean()),
                                                           permuted_regressor_kept_share_descriptive=float(PN.mean() / A.mean()))
    res['participant_flags_pooled_over_4_lags'] = [lag_flag_participant(p) for p in per]; res['n_flagged'] = int(sum(f['flagged'] for f in res['participant_flags_pooled_over_4_lags']))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=''); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    smoke = bool(a.subjects)
    if not smoke: assert (out / 'PREREG_STAMP.json').exists() and not (out / 'summary.json').exists(), 'stamp PREREG first / summary exists'
    info = D2.dataset('ds003458'); ns = D2.s13b_targets(); Q = np.load(ROOT / ns['QUANT'])
    subs = [int(x) for x in a.subjects.split(',')] if a.subjects else list(range(1, 24)); cells = {'real': [], 'plant_slow': [], 'plant_event': []}; gate = {}; planted_meta = {}
    sb = json.loads((ROOT / 'results/standard/summary.json').read_text())['prediction_error']['raw_erp_bottleneck_ds003458']
    for s in subs:
        fz = np.load(ROOT / f'results/features/raw/sub-{s:03d}.npz'); F = fz['F'].astype(np.float64); tri = fz['tri'].astype(int); y = info['qtil'](s, tri).astype(np.float64)
        Qc, Qr = Q['choice'][s - 1], Q['reward'][s - 1]
        r = analyse(F, y, tri, s, Qc, Qr, 5000 + s); r['participant'] = s; r['n'] = int(len(y)); cells['real'].append(r)
        st = json.loads((ROOT / f'results/standard/pe_bottleneck/sub-{s:03d}.json').read_text()); o = next(x for x in st['rows'] if x['input'] == 'eeg' and x['mode'] == 'pca' and x['L'] == 'full')
        gate[s] = dict(gain_diff=abs(o['gain'] - r['Y0']), shift_diff=max(abs(u - v) for u, v in zip(o['shifted'], r['shifted'])), n_stored=st['n_trials'], n_here=int(len(y)))
        for kind, sd in (('slow', 7000), ('event', 8000)):
            Fp, yp, meta = plant(F, tri, kind, sd + s); rp = analyse(Fp, yp, tri, s, Qc, Qr, sd + 500 + s); rp['participant'] = s; cells['plant_' + kind].append(rp); planted_meta[kind] = meta
        print(s, 'Y0 %.4f' % r['Y0'], 'gate gain diff %.1e' % gate[s]['gain_diff'], 'perm_p95 %.4f' % r['perm_p95'], 'plant slow Y0 %.3f event Y0 %.3f' % (cells['plant_slow'][-1]['Y0'], cells['plant_event'][-1]['Y0']),
              'prev kept(person) %.3f' % (r['prev']['Y'] - r['prev']['floor']), '%.0fs' % (time.time() - t0), flush=True)
        json.dump({k: v[-1] for k, v in cells.items()}, open(out / f'per_participant_sub-{s:03d}.json', 'w'))
    # ---------------- gates
    mean_here = float(np.mean([p['Y0'] for p in cells['real']])); shift_here = float(np.mean([np.mean(p['shifted']) for p in cells['real']]))
    G = dict(per_participant_max_gain_diff=max(v['gain_diff'] for v in gate.values()), per_participant_max_shift_diff=max(v['shift_diff'] for v in gate.values()), all_n_equal=all(v['n_stored'] == v['n_here'] for v in gate.values()),
             mean_yield_here=mean_here, mean_yield_stored=sb['full_width_gain'], mean_shift_floor_here=shift_here, mean_shift_floor_stored=sb['shift_floor'])
    G['passed'] = bool(G['per_participant_max_gain_diff'] <= GATE and G['per_participant_max_shift_diff'] <= GATE and G['all_n_equal'] and (smoke or (abs(mean_here - sb['full_width_gain']) <= GATE and abs(shift_here - sb['shift_floor']) <= GATE)))
    C = {k: cohort(v) for k, v in cells.items()}
    ctl = dict(slow_flagged=C['plant_slow']['n_flagged'], slow_need=FLAG_MIN, event_flagged=C['plant_event']['n_flagged'], event_off=C['plant_event']['n'] - C['plant_event']['n_flagged'], off_need=OFF_MIN,
               slow_Y0_mean=C['plant_slow']['Y0_mean'], event_Y0_mean=C['plant_event']['Y0_mean'], Y0_need=PLANT_Y0_MIN, slow_cohort_flag_lower_gt_1p5=C['plant_slow']['ii_flag_slow_state'], event_cohort_rule_upper_le_1p5=C['plant_event']['ii_passed'])
    # control counts scale with n only in the full run (20 of 23); in a smoke they are reported, not judged
    ctl['passed'] = bool((not smoke) and ctl['slow_flagged'] >= FLAG_MIN and ctl['event_off'] >= OFF_MIN and ctl['slow_Y0_mean'] >= PLANT_Y0_MIN and ctl['event_Y0_mean'] >= PLANT_Y0_MIN)
    R = C['real']; void = (not G['passed']) or (not ctl['passed'])
    event_specific = bool(R['i_block_perm']['passed'] and R['ii_passed'] and R['iii_prev']['passed'])
    verdict = dict(i=R['i_block_perm']['passed'], ii=R['ii_passed'], iii=R['iii_prev']['passed'], gate_passed=G['passed'], controls_passed=ctl['passed'], void=bool(void),
                   result=('VOID (gate or planted control failed): no sentence' if void else ('EVENT-SPECIFIC for the ridge read-out (all three pass)' if event_specific else 'a slow-state route is not excluded (at least one check failed)')))
    summ = dict(NOT_CITABLE='until independent blind review', code_sha256=CODE_SHA256, readout='registered raw-ERP ridge, full width (NOT EEGNet)', smoke=smoke, n=len(subs), constants=dict(NPERM=NPERM, BLOCK=BLOCK, LAGS=LAGS, BOOT=BOOT, RATIO_MAX=RATIO_MAX, KEEP_MIN=KEEP_MIN, MIN_DET=MIN_DET,
                AMP=AMP, PCs=[PC_LO, PC_HI - 1], PHI_SLOW=PHI_SLOW, PHI_EVENT=PHI_EVENT, TARGET_LOAD=TARGET_LOAD), planted_meta=planted_meta, gate=G, controls=ctl, real=C['real'], plant_slow=C['plant_slow'], plant_event=C['plant_event'], verdict=verdict, runtime_s=time.time() - t0)
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1); print('VERDICT', json.dumps(verdict)); print('GATE', json.dumps(G)); print('CONTROLS', json.dumps(ctl))


if __name__ == '__main__':
    main()
