"""GPU-FILL planted-information calibration of the ocular-share pipeline, ds003458 (, 22 Nov 2026).
Question: does the in-fold ocular-regression share (experiments/ext/pe_eogreg.py, same ridge/PCA read-out d2_pe_tasks.pe_gain, same ocular regressors O,
same kept-vs-permuted-null definition) recover a PLANTED nuisance fraction? FALSIFIER: |estimated - true| > 0.20 at any planted fraction -> the share pipeline is retired.
Construction (real EEG features F = results/features/raw, n x 1472, kept trials as pe_eogreg; real ocular regressors O, n x 70):
  target     y ~ N(0,1) iid, drawn per (participant, seed): independent of the real EEG, so ALL target information is planted and real EEG supplies the noise covariance.
  neural     F' = F + a * y * u,            u = random combination of the principal components 20-51 (not the non-stationary leading drift components) components of the participant's own z-scored features, x per-feature SD (not in the span of O except by chance; the neural part keeps (1-RHO^2) of its y-linked variance after regression: expected, reported)
  ocular     latent eye signal s = RHO*y + sqrt(1-RHO^2)*e (RHO=0.5), measured by the ocular regressors O' = O + kappa * s * (sd_O * sign pattern) (kappa 0.3);  F' += (O'-O) @ W,  W = rank-1 (sign pattern of the 70 columns) x (another random top-16-PC topography) * b   (EEG leakage of an ocular signal that the
             measured ocular channels also carry, as real eye movements would)
  fraction f amplitudes a = a0*sqrt(1-f), b = b0*sqrt(f) (information ~ amplitude^2), f in {0, .25, .5, .75}; plus a NULL cell a=b=0 (floor gate)
  TRUE nuisance fraction of the planted yield, measured, not assumed: yield of the neural-only data N, ocular-only data Ob (same pipeline, no regression):  f_true = Yo / (Yo + Yn).
  ESTIMATE (pe_eogreg definition): kept = (yield after in-fold regression of O' out of F') / (yield of F' with a row-permuted O' regressed out: cost of 70 unrelated columns);
             estimated nuisance share = 1 - kept.
  yield = held-out density gain - mean of 3 circular-shift floors (pe_eogreg rule), at width L = 64 (primary) and 'full' (PCA-exposure check).
Gates: G1 (unplanted real-target 'all' gain reproduces results/ext/eogreg_v2 to 1e-9, per participant); G2 (NULL cell |yield| <= 0.02); smoke on 2 participants.
Usage: --out DIR --subjects 1,2,... [--seeds 0,1] [--a0 A --b0 B]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import d2_pe_tasks as D2
import pe_eogreg as PE
DEV = 'cuda'; LS = (64, 'full'); t0 = time.time()
log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
FR = [0.0, 0.25, 0.5, 0.75]; KPC = 32; PC0 = 20; KAPPA = 0.3; RHO = 0.5


def yield_of(F, y, transform=None):
    """{L: yield} = gain - mean(3 circular shift floors) for the pe_gain pipeline."""
    Ys = np.column_stack([y] + [np.roll(y, round(len(y) * q / 4)) for q in (1, 2, 3)])
    sc = D2.pe_gain(F, Ys, LS, transform=transform)
    return {str(L): float(sc[L][:, 0].mean() - sc[L][:, 1:].mean()) for L in LS}


def run_sub(s, out, seeds, a0, b0):
    t1 = time.time(); sub = f'sub-{s:03d}'; info = D2.dataset('ds003458')
    fz = np.load(ROOT / f'results/features/raw/{sub}.npz'); tri0 = fz['tri']; F0 = fz['F'].astype(np.float64)
    O, Ev, keep = PE.ocular_v2(s, tri0); tri = tri0[keep]; F = F0[keep]; n = len(tri); O = O.astype(np.float64)
    yreal = info['qtil'](s, tri).astype(np.float64)
    # G1: unplanted, real target, same arms as pe_eogreg
    g1 = yield_of(F, yreal); Ys = np.column_stack([yreal] + [np.roll(yreal, round(n * q / 4)) for q in (1, 2, 3)])
    sc = D2.pe_gain(F, Ys, LS); gall = float(sc[64][:, 0].mean())
    ref = json.loads((ROOT / f'results/ext/eogreg_v2/eogreg_{sub}.json').read_text())['rows']['all']['64']['gain']
    res = dict(participant=s, n=n, G1_all_gain=gall, G1_saved=ref, G1_abs_diff=abs(gall - ref), real_yield=g1, cells=[])
    Fsd = F.std(0) + 1e-9; Osd = O.std(0) + 1e-9
    for sd in seeds:
        rng = np.random.default_rng(910000 + 100 * s + sd); y = rng.standard_normal(n)
        Fz = (F - F.mean(0)) / Fsd; _, Sv, Vt = np.linalg.svd(Fz, full_matrices=False); skd = Sv[PC0:PC0 + KPC] / np.sqrt(n)      # planted patterns live in the top-KPC principal subspace (smooth detectability)
        pat = lambda: ((rng.standard_normal(KPC) * skd) @ Vt[PC0:PC0 + KPC]) * Fsd
        u = pat(); sg = rng.choice([-1.0, 1.0], O.shape[1]); W = np.outer(sg, pat()) / O.shape[1]       # ocular topography: rank-1, sg @ W = pattern
        sig = RHO * y + np.sqrt(1 - RHO ** 2) * rng.standard_normal(n)       # latent eye signal, correlation RHO with the target
        dO = KAPPA * np.outer(sig, Osd * sg)                             # the measured ocular regressors carry the eye signal at KAPPA SD per unit (only when an ocular component is planted)
        perm = np.random.default_rng(777 + s + sd).permutation(n)
        for f in ['null'] + FR:
            a, b = (0.0, 0.0) if f == 'null' else (a0 * np.sqrt(1 - f), b0 * np.sqrt(f))
            Fn = F + a * np.outer(y, u); Fo = F + b * ((dO / (KAPPA * Osd)) @ W); Fm = F + a * np.outer(y, u) + b * ((dO / (KAPPA * Osd)) @ W); Om = O + (dO if (b > 0) else 0)
            cell = dict(seed=sd, f=f, a=a, b=b)
            cell['mix_all'] = yield_of(Fm, y)
            cell['mix_eogreg'] = yield_of(Fm, y, PE.eogreg(Om))
            cell['mix_permnull'] = yield_of(Fm, y, PE.eogreg(Om[perm]))
            cell['neural_only'] = yield_of(Fn, y)
            cell['ocular_only'] = yield_of(Fo, y)
            res['cells'].append(cell)
        log(f'{sub} seed {sd} done ({time.time()-t1:.0f}s)')
    res['runtime_s'] = time.time() - t1
    (out / f'{sub}.json').write_text(json.dumps(res, indent=1))
    return res


def boot_ratio(num, den, B=10000):
    num = torch.tensor(num, dtype=torch.float64, device=DEV); den = torch.tensor(den, dtype=torch.float64, device=DEV); g = torch.Generator(device=DEV); g.manual_seed(0)
    ix = torch.randint(len(num), (B, len(num)), generator=g, device=DEV); r = num[ix].mean(1) / den[ix].mean(1)
    return dict(value=float(num.mean() / den.mean()), ci95=[float(r.quantile(.025)), float(r.quantile(.975))])


def summarize(out, subs, a0, b0):
    R = [json.loads((out / f'sub-{s:03d}.json').read_text()) for s in subs if (out / f'sub-{s:03d}.json').exists()]
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, n_participants=len(R), a0=a0, b0=b0, bootstrap='participant, 10000 resamples, GPU torch, seed 0',
             G1_max_abs_diff=max(r['G1_abs_diff'] for r in R), rows={})
    S['G1_pass'] = bool(S['G1_max_abs_diff'] < 1e-9)
    fails = []
    for L in ('64', 'full'):
        for f in ['null'] + FR:
            cs = [[c for c in r['cells'] if c['f'] == f] for r in R]
            # participant-level means over seeds
            m = lambda key: [float(np.mean([c[key][L] for c in cc])) for cc in cs]
            Yn, Yo, E, PN, A = m('neural_only'), m('ocular_only'), m('mix_eogreg'), m('mix_permnull'), m('mix_all')
            row = dict(mean_yield_all=float(np.mean(A)), mean_yield_eogreg=float(np.mean(E)), mean_yield_permnull=float(np.mean(PN)), mean_neural_only=float(np.mean(Yn)), mean_ocular_only=float(np.mean(Yo)))
            if f == 'null': row['null_floor_ok_le_0.02'] = bool(max(abs(float(np.mean(A))), abs(float(np.mean(E))), abs(float(np.mean(PN)))) <= 0.02)
            else:
                ftrue = boot_ratio(Yo, [x + z for x, z in zip(Yn, Yo)]); kept = boot_ratio(E, PN)
                est = 1 - kept['value']; row.update(f_planted=f, f_true_measured=ftrue, kept=kept, est_nuisance_share=est, est_ci95=[1 - kept['ci95'][1], 1 - kept['ci95'][0]],
                                                      est_minus_true=est - ftrue['value'], miss_over_020=bool(abs(est - ftrue['value']) > 0.20))
                if L == '64' and row['miss_over_020']: fails.append(f)
            S['rows'][f'L{L}|f={f}'] = row
    S['FALSIFIER'] = dict(rule='|estimated - true| > 0.20 at any planted fraction (L=64)', fired=bool(fails), fractions=fails,
                          verdict='share pipeline RETIRED (planted nuisance fraction missed by > 20 points)' if fails else 'share recovers the planted nuisance fraction within 20 points at every planted fraction (L=64)')
    flips = [f for f in FR if f != 0 and (S['rows'][f'L64|f={f}']['miss_over_020'] != S['rows'][f'Lfull|f={f}']['miss_over_020'])]
    S['pca_exposure_flip'] = dict(fractions_where_verdict_differs_between_L64_and_Lfull=flips, temperature='not run (pe_gain has no temperature axis)')
    S['per_participant_file_sha256'] = {f'sub-{s:03d}': _hl.sha256((out / f'sub-{s:03d}.json').read_bytes()).hexdigest() for s in subs if (out / f'sub-{s:03d}.json').exists()}
    (out / 'summary.json').write_text(json.dumps(S, indent=1)); log(json.dumps({k: v for k, v in S.items() if k != 'per_participant_file_sha256'}, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=','.join(str(i) for i in range(1, 24)))
    ap.add_argument('--seeds', default='0,1'); ap.add_argument('--a0', type=float, default=0.14); ap.add_argument('--b0', type=float, default=0.45); ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); subs = [int(x) for x in a.subjects.split(',')]
    if not a.summarize:
        for s in subs:
            if not (out / f'sub-{s:03d}.json').exists():
                r = run_sub(s, out, [int(x) for x in a.seeds.split(',')], a.a0, a.b0)
                if r['G1_abs_diff'] > 1e-9: raise SystemExit(f'G1 FAIL sub {s}: {r["G1_abs_diff"]}')
    summarize(out, subs, a.a0, a.b0)


if __name__ == '__main__':
    main()
