"""T0 SCREEN of the trained decomposer (runner, 24 Nov 2026).
Implements outputs/FINAL/IDEATOR_TRAINED_DECOMPOSER_20261124.md section 3 ("T0 screen") with the runner's pre-set reading rules (fixed BEFORE any
non-smoke number was seen; the same text is written to results/td_screen/PREREG.json with the script hash before the full fit).

QUESTION. Do cheap per-cell statistics (no new read-out call) predict the finite-sample single-source neural yield Y_N of a planted ds003458 cell
with calibrated quantiles, on HELD-OUT participants and a HELD-OUT RHO?
CELL / TRUTH. Cell = (participant s, seed, f, RHO) of gf_planted_calib_v2 (regenerated with pid_regen's rng stream). Truth = stored `neural_only`
(L = 64, bits, unclipped; yield = gain - mean of 3 shift floors) = finite-sample single-source yield; Y_O = stored `ocular_only` (descriptive).
INPUTS (functions of the MIXED cell only: Fm, y, Om = measured eye regressors; plus the stored joint yield). No recipe parameter (RHO, f, a, b, kappa, cos),
no single-source yield and no A1/A4 yield is an input of the primary feature set S1. Feature groups (35 numbers):
  decoder: joint yield (stored mix_all, L 64 and full)                                                     2
  EEG:     log-spectrum of the 64 whitened PCs of F-standardised (8 bins of 8 PCs, mean log eigenvalue)     8
           target-to-PC map c_y (corr of y with each PC score): band energy n*sum c^2 (8 bands),
           split-half replicate n/2*sum c1*c2 over odd/even contiguous outer blocks (8 bands)               16
  nuisance: E = held-out R^2 of y from [1, Om] (10 outer contiguous folds, ridge 1e-8; protocol sec 6),
           sigma_1..5 of the target-orthogonal fitted map of the PC scores on the eye regressors,
           the permutation threshold theta (P = 49 row permutations, 48th smallest top singular value),
           k = #(sigma_i > theta), and n                                                                    9
  S2 (ablation, descriptive) adds A1 yield (stored mix_eogreg) and the permuted-regressor yield (mix_permnull), L 64 and full: 4 numbers.
SPLIT. Train = ODD participants of results/gpu_fill_20261123/planted_v2_rho{0.25,0.9} (12 each); test = EVEN participants (2..22, 11) at RHO 0.5.
  Cells per participant: seeds 0,1 x f in {null, 0, .25, .5, .75}; null and f = 0 do not depend on RHO, so they enter training once.
MODELS (torch, GPU, float64): linear quantile regression (L2) and an MLP quantile regression (32-32, 5-member ensemble, monotone cumulative-softplus
  head), pinball loss on the five levels .05 .25 .5 .75 .95 summed; hyper-parameters by grouped (participant) 4-fold CV on the TRAINING participants only;
  the PRIMARY model is the one with the lower grouped-CV pinball (decided on training data only). Features standardised on the training rows.
PRE-SET READING (primary = S1 features, Y_N, L 64, test cells f in {.25,.5,.75} at RHO 0.5, even participants; unit = participant, 95% bootstrap over
  the 11 test participants, 10000 resamples, one index tensor, torch generator seed 20261124):
  C1 MAE of the median |q50 - Y_N| <= 0.03 bit        PASS if the upper 97.5% bound <= 0.03, FAIL if the lower 2.5% bound > 0.03, else UNDECIDED
  C2 coverage of [q05, q95] in [0.80, 0.95]           PASS if the 95% interval is inside [0.80, 0.95], FAIL if it is entirely outside, else UNDECIDED
  C3 MAE / MAE(A1) <= 0.75 (>= 25% below A1), A1 error = |stored mix_eogreg - neural_only|, same cells
                                                      PASS if the upper bound <= 0.75, FAIL if the lower bound > 0.75, else UNDECIDED
  VERDICT: FAIL if any criterion is FAIL; PASS if all three PASS; else UNDECIDED. VOID if the read-out jitter floor (below) exceeds 0.02 bit.
  A4 (target-adjusted regression) error is compared where stored cells for the test participants exist; descriptive, decides nothing.
JITTER FLOOR (measured before the fit): neural-only cells (8 participants, seed 0, f .25 .5 .75) re-scored with the rows circularly rolled by n//20 (half an outer
  fold, y rolled with them): floor = sqrt(mean(d^2) / 2) over cells, d = Y_rolled - Y (the SD of Y_N over the two fold alignments). pe_gain has no random
  component; a repeat of the identical call is also run (determinism). Void threshold 0.02 bit at L 64.
CONTROLS (pre-set pass conditions): constant quantiles (no features); noise-input (features replaced by N(0,1)); shuffled truth (training labels permuted, 5 draws);
  positive control (the true Y_N + N(0, 0.01^2) as the only input; tests row alignment). Noise-input and shuffled-truth must have a point MAE > 0.03 and a point
  MAE / MAE(A1) > 0.75, and must not beat the real model; the positive control must reach MAE <= 0.03. A violated control invalidates the screen.
GATES: stored-cell regeneration (re-scored mix_all, neural_only, ocular_only, mix_eogreg vs stored, 1e-4, both widths), regenerated u, v, y, sig == pid_regen.load,
  cached ocular arrays == fresh ocular_v2 (smoke participants), E == results/yield_projector/runner_T9R/scope_E.json (1e-4, where complete), finite values,
  train / test participants disjoint, feature names contain no recipe parameter.
Stages: --stage feats|gate|jitter|fit [--subjects ..] [--smoke]. Outputs in results/td_screen/ (NOT citable)."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "2")
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import d2_pe_tasks as D2
import pe_eogreg as PE
import pid_regen as PR

DEV = 'cuda'; T64 = dict(dtype=torch.float64, device=DEV); t0 = time.time()
log = lambda *m: print(f'[{time.time() - t0:7.1f}s]', *m, flush=True)
OUT = ROOT / 'results/td_screen'; STORED = {r: ROOT / f'results/gpu_fill_20261123/planted_v2_rho{r}' for r in (0.25, 0.5, 0.9)}
OCACHE = ROOT / 'results/yield_projector/cache'          # read-only (bit-exact ocular_v2 arrays of another run); fallback = PE.ocular_v2
KAPPA, KPC, PC0 = 0.3, 32, 20; A0, B0 = 0.14, 0.45
FS = ['null', 0.0, 0.25, 0.5, 0.75]; SEEDS = (0, 1); QS = (0.05, 0.25, 0.5, 0.75, 0.95)
NAMES_S1 = (['joint64', 'jointfull'] + [f'spec{i}' for i in range(8)] + [f'tmap_energy{i}' for i in range(8)] + [f'tmap_rep{i}' for i in range(8)]
            + ['E'] + [f'sv{i}' for i in range(1, 6)] + ['theta', 'k', 'n'])
NAMES_S2 = NAMES_S1 + ['a1_64', 'a1full', 'perm64', 'permfull']
FORBIDDEN = {'rho', 'f', 'a', 'b', 'kappa', 'cos', 'neural_only', 'ocular_only', 'a4', 'a3'}
assert not (set(NAMES_S2) & FORBIDDEN)
ODD = list(range(1, 24, 2)); EVEN = list(range(2, 23, 2))
FN = lambda f: 'null' if f == 'null' else f'f{f:g}'
P_PERM = 49
G = lambda x: torch.as_tensor(np.ascontiguousarray(x), **T64)


def fjson(path, obj): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------- regeneration (same rng stream as gf_planted_calib_v2.run_sub / pid_regen.load)
_BASE = {}
_OCULAR_RAW = PE.ocular_v2
def ocular(s, tri0):
    for d in (OCACHE, OUT / 'cache'):
        p = d / f'ocular_v2_sub-{s:03d}.npz'
        if p.exists():
            z = np.load(p, allow_pickle=True); assert np.array_equal(z['tri0'], tri0), 'ocular cache: trial index mismatch'
            return z['O'], z['Ev'], z['keep']
    O, Ev, keep = _OCULAR_RAW(s, tri0); p = OUT / 'cache' / f'ocular_v2_sub-{s:03d}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, O=O, Ev=Ev, keep=keep, tri0=tri0); return O, Ev, keep
PE.ocular_v2 = ocular            # pid_regen.load then reads the same cached arrays (the gate compares the cache with a fresh ocular_v2 on two participants)


def base(s):
    if s not in _BASE:
        fz = np.load(ROOT / f'results/features/raw/sub-{s:03d}.npz'); tri0 = fz['tri']; F0 = fz['F'].astype(np.float64)
        O, Ev, keep = ocular(s, tri0); _BASE[s] = (F0[keep], O.astype(np.float64))
    return _BASE[s]


def regen(s, rho, sd):
    """-> dict(F, O, y, u, v, sg, sig, Osd, n); rng order: y (n), g_u (32), sg (70), g_v (32), e (n) = gf_planted_calib_v2 lines 49-53 = pid_regen.load."""
    F, O = base(s); n = len(F); Fsd = F.std(0) + 1e-9; Osd = O.std(0) + 1e-9
    rng = np.random.default_rng(910000 + 100 * s + sd); y = rng.standard_normal(n)
    Fz = (F - F.mean(0)) / Fsd; _, Sv, Vt = np.linalg.svd(Fz, full_matrices=False); skd = Sv[PC0:PC0 + KPC] / np.sqrt(n)
    pat = lambda: ((rng.standard_normal(KPC) * skd) @ Vt[PC0:PC0 + KPC]) * Fsd
    u = pat(); sg = rng.choice([-1.0, 1.0], O.shape[1]); v = pat(); sig = rho * y + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return dict(F=F, O=O, y=y, u=u, v=v, sg=sg, sig=sig, Osd=Osd, n=n)


def cells_of(R, f):
    """The three feature matrices of one (cell, f) with the generator's own expressions (gf_planted_calib_v2 lines 54-58)."""
    F, O, y, u, v, sg, sig, Osd = (R[k] for k in ('F', 'O', 'y', 'u', 'v', 'sg', 'sig', 'Osd'))
    a, b = PR.amp(f); W = np.outer(sg, v) / O.shape[1]; dO = KAPPA * np.outer(sig, Osd * sg)
    Fn = F + a * np.outer(y, u); Fo = F + b * ((dO / (KAPPA * Osd)) @ W); Fm = F + a * np.outer(y, u) + b * ((dO / (KAPPA * Osd)) @ W)
    Om = O + (dO if b > 0 else 0)
    return Fn, Fo, Fm, Om


def stored(rho, s):
    p = STORED[rho] / f'sub-{s:03d}.json'
    return json.loads(p.read_text()) if p.exists() else None


def scell(S, sd, f):
    for c in S['cells']:
        if c['seed'] == sd and (c['f'] == f if f == 'null' else (c['f'] != 'null' and abs(c['f'] - f) < 1e-12)): return c
    raise KeyError((sd, f))


# ---------------------------------------------------------------- features (functions of Fm, y, Om and the stored joint yields only)
def e_scope(y, Om):
    """Protocol sec 6: held-out R^2 of y from [1, Om], LS on training rows of the 10 outer contiguous folds (ridge 1e-8), pooled."""
    n = len(y); fo = D2.folds(n); Y = G(y); Og = torch.cat([torch.ones(n, 1, **T64), G(Om)], 1); num = 0.0; den = 0.0
    for b in range(10):
        tr = torch.as_tensor(fo != b, device=DEV); te = ~tr
        A = Og[tr]; B = torch.linalg.solve(A.T @ A + 1e-8 * torch.eye(A.shape[1], **T64), A.T @ Y[tr])
        num += float(((Y[te] - Og[te] @ B) ** 2).sum()); den += float(((Y[te] - Y[tr].mean()) ** 2).sum())
    return 1 - num / den


def features(Fm, y, Om, seed_perm):
    n = len(y); X = G(Fm); X = (X - X.mean(0)) / (X.std(0, unbiased=False) + 1e-9)
    U, S, _ = torch.linalg.svd(X, full_matrices=False); lam = S[:64] ** 2 / n; Z = U[:, :64] * math.sqrt(n)         # whitened scores, unit variance
    yc = G(y); yc = (yc - yc.mean()) / yc.std(unbiased=False)
    spec = torch.log(lam).reshape(8, 8).mean(1)
    fo = D2.folds(n); od = torch.as_tensor(fo % 2 == 1, device=DEV); ev = ~od
    c = Z.T @ yc / n; c1 = Z[od].T @ yc[od] / int(od.sum()); c2 = Z[ev].T @ yc[ev] / int(ev.sum())
    en = (n * c ** 2).reshape(8, 8).sum(1); rep = (n / 2 * c1 * c2).reshape(8, 8).sum(1)
    E = e_scope(y, Om)
    D = torch.stack([torch.ones(n, **T64), yc], 1); Dp = torch.linalg.pinv(D)
    Zp = Z - D @ (Dp @ Z); Ob = G(Om); Op = Ob - D @ (Dp @ Ob)
    Uo, So, _ = torch.linalg.svd(Op, full_matrices=False); r = int((So > 1e-8 * So[0]).sum()); Q = Uo[:, :r]
    top_sv = lambda C: torch.linalg.eigvalsh(C.transpose(-1, -2) @ C).flip(-1).clamp_min(0).sqrt()               # singular values via the 64 x 64 Gram, same route for fitted and permuted maps
    sv = top_sv(Q.T @ Zp)
    g = torch.Generator(device=DEV); g.manual_seed(int(seed_perm))
    perms = torch.stack([torch.randperm(n, generator=g, device=DEV) for _ in range(P_PERM)])
    top = top_sv(Q[perms].transpose(1, 2) @ Zp[None])[:, 0]
    theta = float(torch.sort(top).values[math.ceil(0.95 * (P_PERM + 1)) - 1])                                       # the ceil(0.95 (P+1))-th smallest, 1-based
    k = int((sv[:5] > theta).sum())
    out = [*spec.tolist(), *en.tolist(), *rep.tolist(), E, *sv[:5].tolist(), theta, float(k), float(n)]
    return out


def feats_of_cell(Fm, y, Om, Sc, seed_perm):
    f = features(Fm, y, Om, seed_perm)
    s1 = [Sc['mix_all']['64'], Sc['mix_all']['full']] + f            # order == NAMES_S1: joint(2), spec(8), energy(8), rep(8), E, sv(5), theta, k, n
    s2 = s1 + [Sc['mix_eogreg']['64'], Sc['mix_eogreg']['full'], Sc['mix_permnull']['64'], Sc['mix_permnull']['full']]
    assert len(s1) == len(NAMES_S1) and len(s2) == len(NAMES_S2)
    return s1, s2


def stage_feats(subs, smoke):
    d = OUT / ('cells_smoke' if smoke else 'cells'); d.mkdir(parents=True, exist_ok=True); tag = {}
    for s in subs:
        rhos = (0.25, 0.9) if s % 2 == 1 else (0.5,)
        for rho in rhos:
            p = d / f'sub-{s:03d}_rho{rho}.json'
            if p.exists(): continue
            St = stored(rho, s); assert St is not None, (s, rho); t1 = time.time(); rows = []
            for sd in SEEDS:
                R = regen(s, rho, sd)
                for fi, f in enumerate(FS):
                    Fn, Fo, Fm, Om = cells_of(R, f); Sc = scell(St, sd, f)
                    pseed = (100 * s + sd) * 1000 + 10 * fi + (0 if f in ('null', 0.0) else (1 if rho == 0.25 else 2 if rho == 0.5 else 3))
                    s1, s2 = feats_of_cell(Fm, R['y'], Om, Sc, pseed)
                    assert np.isfinite(s2).all()
                    rows.append(dict(participant=s, seed=sd, f=f, rho=rho, n=R['n'], s1=s1, s2=s2,
                                     YN64=Sc['neural_only']['64'], YNfull=Sc['neural_only']['full'], YO64=Sc['ocular_only']['64'], YOfull=Sc['ocular_only']['full'],
                                     A0_64=Sc['mix_all']['64'], A1_64=Sc['mix_eogreg']['64'], A1_full=Sc['mix_eogreg']['full']))
            fjson(p, dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, participant=s, rho=rho, names_s1=NAMES_S1, names_s2=NAMES_S2, rows=rows,
                          runtime_s=time.time() - t1))
            log(f'feats sub {s} rho {rho}: {len(rows)} cells ({time.time() - t1:.0f}s)')


# ---------------------------------------------------------------- gates and jitter
def stage_gate(smoke):
    pairs = [(1, 0.5), (2, 0.5)] if smoke else [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (1, 0.9), (3, 0.25)]
    seeds = (0,) if smoke else SEEDS; res = dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, rows=[]); worst = 0.0
    for s, rho in pairs:
        St = stored(rho, s)
        for sd in seeds:
            R = regen(s, rho, sd); L = PR.load(s, rho, sd)           # regenerated u, v, y, sig == pid_regen.load, exactly
            for k_ in ('y', 'u', 'v', 'sig'): assert np.array_equal(R[k_], L[k_]), (s, rho, sd, k_)
            assert np.array_equal(R['F'], L['F'])
            for f in ('null', 0.5):
                Fn, Fo, Fm, Om = cells_of(R, f); Sc = scell(St, sd, f); y = R['y']; got = {}
                got['mix_all'] = PR.yield_of(Fm, y)
                if f != 'null':
                    got['neural_only'] = PR.yield_of(Fn, y); got['ocular_only'] = PR.yield_of(Fo, y)
                    Ys = np.column_stack([y] + [np.roll(y, round(len(y) * q / 4)) for q in (1, 2, 3)]); sc = D2.pe_gain(Fm, Ys, PR.LS, transform=PE.eogreg(Om))
                    got['mix_eogreg'] = {str(L_): float(sc[L_][:, 0].mean() - sc[L_][:, 1:].mean()) for L_ in PR.LS}
                for key, val in got.items():
                    for w in ('64', 'full'):
                        dd = abs(val[w] - Sc[key][w]); worst = max(worst, dd); res['rows'].append(dict(s=s, rho=rho, seed=sd, f=f, key=key, width=w, regenerated=val[w], stored=Sc[key][w], absdiff=dd))
            log(f'gate sub {s} rho {rho} seed {sd}: worst so far {worst:.2e}')
    res['max_absdiff'] = worst; res['pass_1e-4'] = bool(worst <= 1e-4)
    # cached ocular arrays == fresh ocular_v2 (smoke participants)
    if True:
        for s in (1, 2):
            fz = np.load(ROOT / f'results/features/raw/sub-{s:03d}.npz'); tri0 = fz['tri']; O1, E1, k1 = ocular(s, tri0); O2, E2, k2 = _OCULAR_RAW(s, tri0)
            res[f'ocular_cache_equals_fresh_sub{s}'] = bool(np.array_equal(O1, O2) and np.array_equal(k1, k2))
    sp = ROOT / 'results/yield_projector/runner_T9R/scope_E.json'; res['E_vs_scope_E'] = 'not available'
    if sp.exists():
        J = json.loads(sp.read_text())
        if J.get('complete'):
            dmax = 0.0
            for s, rho in pairs:
                for sd in seeds:
                    R = regen(s, rho, sd); Fn, Fo, Fm, Om = cells_of(R, 0.5); e = e_scope(R['y'], Om)
                    dmax = max(dmax, abs(e - J['participants'][f'sub-{s:03d}']['E'][str(rho)]['seeds'][sd]))
            res['E_vs_scope_E'] = dict(max_absdiff=dmax, pass_1e_4=bool(dmax <= 1e-4))
        else: res['E_vs_scope_E'] = 'scope_E.json (runner_T9R) not complete at run time'
    fjson(OUT / ('gate_smoke.json' if smoke else 'gate.json'), res); log('GATE', {k: v for k, v in res.items() if k != 'rows'})
    if not res['pass_1e-4']: raise SystemExit('GATE FAIL: regenerated cells differ from stored cells by more than 1e-4')


def stage_jitter(smoke):
    subs = [1, 2] if smoke else list(range(1, 9)); rows = []; det = []
    for s in subs:
        R = regen(s, 0.5, 0); n = R['n']; h = n // 20; y = R['y']
        for f in (0.25, 0.5, 0.75):
            Fn, Fo, Fm, Om = cells_of(R, f); y0 = PR.yield_of(Fn, y); ys = PR.yield_of(np.roll(Fn, h, 0), np.roll(y, h))
            yd1 = PR.yield_of(Fn[h:], y[h:]); yd2 = PR.yield_of(Fn[:n - h], y[:n - h])      # SUPPLEMENTARY (not the pre-set void rule): fold boundaries moved by dropping h rows at the start / end, no wrap-around
            rows.append(dict(s=s, f=f, Y64=y0['64'], Yfull=y0['full'], Y64_rolled=ys['64'], Yfull_rolled=ys['full'], d64=ys['64'] - y0['64'], dfull=ys['full'] - y0['full'], drop_first_d64=yd1['64'] - y0['64'], drop_last_d64=yd2['64'] - y0['64']))
            if f == 0.5: y1 = PR.yield_of(Fn, y); det.append(max(abs(y1[w] - y0[w]) for w in ('64', 'full')))
        log(f'jitter sub {s} done')
    d64 = np.array([r['d64'] for r in rows]); dfu = np.array([r['dfull'] for r in rows]); dd = np.array([[r['drop_first_d64'], r['drop_last_d64']] for r in rows])
    sup = dict(SUPPLEMENTARY_not_the_void_rule=dict(floor64_drop_rows=float(np.sqrt(np.mean(dd ** 2) / 2)), mean_d_drop_first=float(dd[:, 0].mean()), mean_d_drop_last=float(dd[:, 1].mean()),
                                                   sd_d_drop_first=float(dd[:, 0].std(ddof=1)), sd_d_drop_last=float(dd[:, 1].std(ddof=1))))
    res = dict(**sup,NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, participants=subs, rule='floor = sqrt(mean(d^2)/2), d = Y(rows rolled by n//20) - Y, neural-only cells, seed 0, f .25/.5/.75, RHO-free',
               floor64=float(np.sqrt(np.mean(d64 ** 2) / 2)), floorfull=float(np.sqrt(np.mean(dfu ** 2) / 2)), mean_d64=float(d64.mean()), sd_d64=float(d64.std(ddof=1)), max_abs_d64=float(np.abs(d64).max()),
               repeat_identical_call_max_absdiff=float(max(det)), void_if_floor64_gt=0.02, thresholds_void=bool(np.sqrt(np.mean(d64 ** 2) / 2) > 0.02), rows=rows)
    fjson(OUT / ('jitter_smoke.json' if smoke else 'jitter.json'), res); log('JITTER', {k: v for k, v in res.items() if k != 'rows'})



# ---------------------------------------------------------------- quantile models (torch, GPU; E independent models trained at once, one batched tensor program)
QT = torch.tensor(QS, **T64)
def pinball(pred, y):                                  # pred (N, 5), y (N,) -> mean over rows of the sum over levels
    e = y[:, None] - pred; return torch.maximum(QT * e, (QT - 1) * e).sum(1).mean()


def std_batch(X, masks):
    """X (N, d), masks (E, N) bool -> (E, N, d): each model's features standardised on ITS training rows only (sd < 1e-6 -> 1), clipped to +-10."""
    m = masks.double()[:, :, None]; cnt = m.sum(1); mu = (X[None] * m).sum(1) / cnt
    sd = ((((X[None] - mu[:, None]) ** 2) * m).sum(1) / cnt).sqrt(); sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return ((X[None] - mu[:, None]) / sd[:, None]).clamp(-10, 10)


def train_batch(kind, Xs, ys, masks, lams, seed, steps):
    """Full-batch Adam on the masked pinball loss (+ L2 weight lam per model). Xs (E, N, d) standardised, ys (N,) standardised labels, masks (E, N) training rows.
    Returns predictions (E, N, 5) for ALL rows (sorted for 'lin'; monotone by construction for 'mlp'). Independent models: Adam is element-wise, so the summed loss
    trains each model exactly as if it were alone."""
    E, N, d = Xs.shape; g = torch.Generator(device=DEV); g.manual_seed(seed); m = masks.double(); cnt = m.sum(1); lam = torch.as_tensor(lams, **T64)
    rn = lambda *s, sc=1.0: (torch.randn(E, *s, generator=g, **T64) * sc).requires_grad_(True); z = lambda *s: torch.zeros(E, *s, **T64, requires_grad=True)
    if kind == 'lin':
        W = z(d, 5); b = (torch.quantile(ys, QT)[None].expand(E, 5).clone()).requires_grad_(True); P = [W, b]; lr = 0.01
        fwd = lambda: torch.einsum('end,edq->enq', Xs, W) + b[:, None]; reg = lambda: lam * (W ** 2).sum((1, 2))
    else:
        H = 32; W1 = rn(d, H, sc=1 / math.sqrt(d)); b1 = z(H); W2 = rn(H, H, sc=1 / math.sqrt(H)); b2 = z(H); W3 = rn(H, 5, sc=0.01); b3 = z(5); P = [W1, b1, W2, b2, W3, b3]; lr = 0.005
        with torch.no_grad(): b3[:, 0] = float(torch.median(ys))
        def fwd():
            h = torch.tanh(torch.einsum('end,edh->enh', Xs, W1) + b1[:, None]); h = torch.tanh(torch.einsum('enh,ehk->enk', h, W2) + b2[:, None]); o = torch.einsum('enh,ehq->enq', h, W3) + b3[:, None]
            sp = torch.nn.functional.softplus; mm = o[..., 0]; q25 = mm - sp(o[..., 1]); q05 = q25 - sp(o[..., 2]); q75 = mm + sp(o[..., 3]); q95 = q75 + sp(o[..., 4])
            return torch.stack([q05, q25, mm, q75, q95], -1)
        reg = lambda: lam * ((W1 ** 2).sum((1, 2)) + (W2 ** 2).sum((1, 2)) + (W3 ** 2).sum((1, 2)))
    opt = torch.optim.Adam(P, lr=lr)
    for _ in range(steps):
        opt.zero_grad(); e = ys[None, :, None] - fwd(); pb = torch.maximum(QT * e, (QT - 1) * e).sum(-1)                  # (E, N)
        loss = ((pb * m).sum(1) / cnt + reg()).sum(); loss.backward(); opt.step()
    with torch.no_grad():
        o = fwd(); return torch.sort(o, -1).values if kind == 'lin' else o


GRID = dict(lin=[1e-3, 1e-2, 1e-1, 1.0, 10.0], mlp=[1e-3, 1e-2, 1e-1])
STEPS = dict(lin=1500, mlp=1500); MEMBERS = 5; CV_MEMBERS = 3


def select_and_fit(Xtr, ytr, grp, Xte, seed, steps_scale=1.0, cvgrp=None):
    """Grouped CV (by participant, up to 4 folds) over the grids on TRAINING rows only; refit (5-member ensemble for the MLP) on all training rows with the CV-best
    penalty; returns kind -> dict(cv_pinball, lam, pred on Xte) and the primary kind (lower grouped-CV pinball, decided on training data only)."""
    N = len(ytr); ug = sorted(set((grp if cvgrp is None else cvgrp).tolist())); nf = min(4, len(ug)); folds = [ug[i::nf] for i in range(nf)]; out = {}
    mu_y, sd_y = float(ytr.mean()), float(ytr.std() + 1e-12); ys = (ytr - mu_y) / sd_y; Xall = torch.cat([Xtr, Xte], 0)
    gv = (grp if cvgrp is None else cvgrp); fold_rows = [torch.as_tensor(np.isin(gv, fg), device=DEV) for fg in folds]
    for kind in ('lin', 'mlp'):
        steps = max(50, int(STEPS[kind] * steps_scale)); mem = CV_MEMBERS if kind == 'mlp' else 1; masks = []; lams = []; idx = []
        for li, lam in enumerate(GRID[kind]):
            for fi, fr in enumerate(fold_rows):
                for _ in range(mem):
                    mk = torch.zeros(len(Xall), dtype=torch.bool, device=DEV); mk[:N] = ~fr; masks.append(mk); lams.append(lam); idx.append((li, fi))
        masks = torch.stack(masks); pr = train_batch(kind, std_batch(Xall, masks), torch.cat([ys, torch.zeros(len(Xte), **T64)]), masks, lams, seed, steps)[:, :N]
        cv = []
        for li in range(len(GRID[kind])):
            tot = 0.0
            for fi, fr in enumerate(fold_rows):
                sel = [i for i, t in enumerate(idx) if t == (li, fi)]; p = pr[sel].mean(0)[fr]; tot += float(pinball(p, ys[fr])) * int(fr.sum()) * sd_y
            cv.append(tot / N)
        li = int(np.argmin(cv)); lam = GRID[kind][li]; mem = MEMBERS if kind == 'mlp' else 1
        masks = torch.zeros(mem, len(Xall), dtype=torch.bool, device=DEV); masks[:, :N] = True
        pr = train_batch(kind, std_batch(Xall, masks), torch.cat([ys, torch.zeros(len(Xte), **T64)]), masks, [lam] * mem, seed + 7, steps)[:, N:].mean(0) * sd_y + mu_y
        out[kind] = dict(cv_pinball=cv[li], cv_all=cv, lam=lam, pred=pr)
    out['primary'] = min(('lin', 'mlp'), key=lambda k: out[k]['cv_pinball'])
    return out


def load_cells(subs_tr, subs_te, smoke):
    d = OUT / ('cells_smoke' if smoke else 'cells'); tr = []; te = []
    for s in subs_tr:
        for rho in (0.25, 0.9):
            J = json.loads((d / f'sub-{s:03d}_rho{rho}.json').read_text()); tr += [r for r in J['rows'] if not (rho == 0.9 and r['f'] in ('null', 0.0))]
    for s in subs_te: te += json.loads((d / f'sub-{s:03d}_rho0.5.json').read_text())['rows']
    return tr, te


def stage_fit(smoke, subs):
    tr_subs = [s for s in subs if s % 2 == 1]; te_subs = [s for s in subs if s % 2 == 0]; assert not (set(tr_subs) & set(te_subs))
    tr, te = load_cells(tr_subs, te_subs, smoke); log(f'train cells {len(tr)} ({len(tr_subs)} participants), test cells {len(te)} ({len(te_subs)} participants)')
    # duplicates of the RHO-free cells: the two RHO files must agree (dedupe check on the 0.25 copy vs the 0.9 copy)
    d = OUT / ('cells_smoke' if smoke else 'cells'); dup = 0.0
    for s in tr_subs:
        a = {(r['seed'], r['f']): r for r in json.loads((d / f'sub-{s:03d}_rho0.25.json').read_text())['rows'] if r['f'] in ('null', 0.0)}
        b = {(r['seed'], r['f']): r for r in json.loads((d / f'sub-{s:03d}_rho0.9.json').read_text())['rows'] if r['f'] in ('null', 0.0)}
        for k in a: dup = max(dup, float(np.abs(np.array(a[k]['s1']) - np.array(b[k]['s1'])).max()), abs(a[k]['YN64'] - b[k]['YN64']))
    log(f'RHO-free cells identical across the 0.25 and 0.9 files: max abs diff {dup:.2e}')
    S = {n: G(np.array([r[n] for r in tr])) for n in ('s1', 's2')}; T = {n: G(np.array([r[n] for r in te])) for n in ('s1', 's2')}
    ytr = G([r['YN64'] for r in tr]); yte = G([r['YN64'] for r in te]); yotr = G([r['YO64'] for r in tr]); yote = G([r['YO64'] for r in te])
    grp = np.array([r['participant'] for r in tr]); gte = np.array([r['participant'] for r in te]); fte = np.array([r['f'] if r['f'] == 'null' else float(r['f']) for r in te], dtype=object)
    prim = torch.as_tensor([(f not in ('null', 0.0)) for f in fte], device=DEV); a1 = G([abs(r['A1_64'] - r['YN64']) for r in te])
    assert all(np.isfinite(S['s2'].cpu().numpy()).ravel()) and all(np.isfinite(T['s2'].cpu().numpy()).ravel())
    assert not (set(grp.tolist()) & set(gte.tolist()))
    ss = 0.15 if smoke else 1.0; res = dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, smoke=smoke, train_participants=tr_subs, test_participants=te_subs,
                                           n_train_cells=len(tr), n_test_cells=len(te), n_primary_test_cells=int(prim.sum()), names_s1=NAMES_S1, names_s2=NAMES_S2,
                                           rhofree_dup_maxdiff=dup)
    seed_tr = np.array([r['seed'] for r in tr]); cvg = (grp * 10 + seed_tr) if smoke else None        # smoke: one training participant, so groups = (participant, seed); full run: participants
    SEL = lambda X, y, Xt, sd: select_and_fit(X, y, grp, Xt, sd, ss, cvg)
    pidx = {p: i for i, p in enumerate(te_subs)}; pid = torch.as_tensor([pidx[p] for p in gte], device=DEV)
    g = torch.Generator(device=DEV); g.manual_seed(20261124); IX = torch.randint(len(te_subs), (10000, len(te_subs)), generator=g, device=DEV)

    def psum(v, mask):                               # per-participant [sum, count] over the masked cells
        S_ = torch.zeros(len(te_subs), **T64).index_add_(0, pid[mask], v[mask]); C_ = torch.zeros(len(te_subs), **T64).index_add_(0, pid[mask], torch.ones(int(mask.sum()), **T64)); return S_, C_
    def boot(num_v, den_v, mask):                    # ratio of resampled sums (den_v = None -> mean per cell)
        n_, c_ = psum(num_v, mask)
        if den_v is None: r = n_[IX].sum(1) / c_[IX].sum(1); pt = float(n_.sum() / c_.sum())
        else: d_, _ = psum(den_v, mask); r = n_[IX].sum(1) / d_[IX].sum(1); pt = float(n_.sum() / d_.sum())
        return dict(value=pt, ci95=[float(r.quantile(0.025)), float(r.quantile(0.975))])
    def read(pred, name):
        q = pred; mae_cell = (q[:, 2] - yte).abs(); cov_cell = ((yte >= q[:, 0]) & (yte <= q[:, 4])).double()
        r = dict(MAE=boot(mae_cell, None, prim), coverage90=boot(cov_cell, None, prim), MAE_over_A1=boot(mae_cell, a1, prim), A1_MAE=boot(a1, None, prim),
                 pinball_all_test=float(pinball(q, yte)), MAE_all_cells=boot(mae_cell, None, torch.ones_like(prim)))
        for key, mk in (('f0', torch.as_tensor([f == 0.0 for f in fte], device=DEV)), ('null', torch.as_tensor([f == 'null' for f in fte], device=DEV))):
            r[f'{key}_MAE'] = boot(mae_cell, None, mk); r[f'{key}_interval_contains_Y'] = boot(cov_cell, None, mk)
            r[f'{key}_median_abs_le_0.02'] = float(((q[mk, 2].abs()) <= 0.02).double().mean()) if key == 'null' else None
            if key == 'null': r['null_interval_contains_0'] = float(((q[mk, 0] <= 0) & (q[mk, 4] >= 0)).double().mean())
        for fv in (0.25, 0.5, 0.75):
            mk = torch.as_tensor([f == fv for f in fte], device=DEV); r[f'f{fv}'] = dict(MAE=boot(mae_cell, None, mk), coverage90=boot(cov_cell, None, mk), A1_MAE=boot(a1, None, mk),
                                                                                         mean_truth=float(yte[mk].mean()), mean_q50=float(q[mk, 2].mean()))
        r['verdict_readings'] = reading(r); return r
    def reading(r):
        m, c, ra = r['MAE'], r['coverage90'], r['MAE_over_A1']
        c1 = 'PASS' if m['ci95'][1] <= 0.03 else ('FAIL' if m['ci95'][0] > 0.03 else 'UNDECIDED')
        c2 = 'PASS' if (c['ci95'][0] >= 0.80 and c['ci95'][1] <= 0.95) else ('FAIL' if (c['ci95'][1] < 0.80 or c['ci95'][0] > 0.95) else 'UNDECIDED')
        c3 = 'PASS' if ra['ci95'][1] <= 0.75 else ('FAIL' if ra['ci95'][0] > 0.75 else 'UNDECIDED')
        allv = [c1, c2, c3]; verdict = 'FAIL' if 'FAIL' in allv else ('PASS' if all(v == 'PASS' for v in allv) else 'UNDECIDED')
        pt = dict(C1=m['value'] <= 0.03, C2=0.80 <= c['value'] <= 0.95, C3=ra['value'] <= 0.75)
        return dict(C1_MAE_le_0p03=c1, C2_coverage_0p80_0p95=c2, C3_ratio_to_A1_le_0p75=c3, verdict=verdict, point_estimates_meet=pt, all_three_points_meet=bool(all(pt.values())))
    # ---- primary: S1, Y_N
    t1 = time.time(); res['models_S1'] = {}; fits = SEL(S["s1"], ytr, T["s1"], 0); res['primary_model'] = fits['primary']
    for k in ('lin', 'mlp'):
        res['models_S1'][k] = dict(cv_pinball_train=fits[k]['cv_pinball'], lam=fits[k]['lam'], test=read(fits[k]['pred'], k))
    res['primary_S1'] = res['models_S1'][fits['primary']]['test']; res['primary_S1_verdict'] = res['primary_S1']['verdict_readings']['verdict']
    log(f"S1 fitted ({time.time() - t1:.0f}s): primary {fits['primary']}, verdict {res['primary_S1_verdict']}")
    # ---- per-participant files and A4 (descriptive; A4 cells exist only for the test participants the projector run has finished at this time; read only)
    A4 = {}
    for s in te_subs:
        p = ROOT / f'results/yield_projector/runner_T9R/arms_rho0.5/sub-{s:03d}.json'
        if p.exists():
            for c in json.loads(p.read_text())['cells']: A4[(s, c['seed'], float(c['f']))] = c['arms']['A4']['yield']['64']
    qlin, qmlp = fits['lin']['pred'], fits['mlp']['pred']; rows_pp = {s: [] for s in te_subs}
    for i, r in enumerate(te):
        rows_pp[r['participant']].append(dict(seed=r['seed'], f=r['f'], YN64=r['YN64'], YO64=r['YO64'], A0_64=r['A0_64'], A1_64=r['A1_64'], A4_64=A4.get((r['participant'], r['seed'], r['f'] if r['f'] == 'null' else float(r['f']))),
                                              q05_25_50_75_95_lin=qlin[i].tolist(), q05_25_50_75_95_mlp=qmlp[i].tolist()))
    if not smoke:
        for s in te_subs: fjson(OUT / 'per_participant' / f'sub-{s:03d}.json', dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, participant=s, primary_model=fits['primary'], rows=rows_pp[s]))
    a4s = sorted({k[0] for k in A4 if all((k[0], sd, fv) in A4 for sd in SEEDS for fv in (0.25, 0.5, 0.75))})
    if a4s:
        ix4 = {p: i for i, p in enumerate(a4s)}; mk4 = torch.as_tensor([(r['participant'] in ix4) and (r['f'] not in ('null', 0.0)) for r in te], device=DEV); pid4 = torch.as_tensor([ix4.get(p, 0) for p in gte], device=DEV)
        a4e = G([abs(A4[(r['participant'], r['seed'], float(r['f']))] - r['YN64']) if mk4[i] else 0.0 for i, r in enumerate(te)]); g4 = torch.Generator(device=DEV); g4.manual_seed(20261124); IX4 = torch.randint(len(a4s), (10000, len(a4s)), generator=g4, device=DEV)
        def b4(v, den=None):
            n_ = torch.zeros(len(a4s), **T64).index_add_(0, pid4[mk4], v[mk4]); d_ = torch.zeros(len(a4s), **T64).index_add_(0, pid4[mk4], (den if den is not None else torch.ones_like(v))[mk4])
            r_ = n_[IX4].sum(1) / d_[IX4].sum(1); return dict(value=float(n_.sum() / d_.sum()), ci95=[float(r_.quantile(0.025)), float(r_.quantile(0.975))])
        elin = (qlin[:, 2] - yte).abs(); emlp = (qmlp[:, 2] - yte).abs(); eprim = elin if fits['primary'] == 'lin' else emlp
        res['A4_comparison'] = dict(participants=a4s, n_cells=int(mk4.sum()), note='descriptive, decides nothing; A4 cells are the projector run (another agent, NOT reviewed) on the participants finished at run time',
                                    MAE_primary_model=b4(eprim), MAE_lin=b4(elin), MAE_mlp=b4(emlp), MAE_A1=b4(a1), MAE_A4=b4(a4e), ratio_primary_over_A1=b4(eprim, a1), ratio_primary_over_A4=b4(eprim, a4e))
    # ---- constant quantiles
    qc = torch.quantile(ytr, QT)[None].expand(len(yte), 5); res['control_constant_quantiles'] = read(qc, 'const')
    # ---- S2 ablation
    f2 = SEL(S["s2"], ytr, T["s2"], 1); res['ablation_S2_with_A1_inputs'] = dict(primary=f2['primary'], **{k: dict(cv_pinball_train=f2[k]['cv_pinball'], lam=f2[k]['lam'], test=read(f2[k]['pred'], k)) for k in ('lin', 'mlp')})
    log('S2 done')
    # ---- noise-input and shuffled-truth and positive controls (primary-model selection rule applied to each; best-of is NOT taken)
    ctr = {}
    gn = torch.Generator(device=DEV); gn.manual_seed(5); Xn, Xnt = torch.randn(S['s1'].shape, generator=gn, **T64), torch.randn(T['s1'].shape, generator=gn, **T64)
    fn = SEL(Xn, ytr, Xnt, 2); ctr['noise_input'] = read(fn[fn['primary']]['pred'], 'noise'); ctr['noise_input']['primary_kind'] = fn['primary']
    sh = []
    for i in range(5):
        pm = torch.randperm(len(ytr), generator=gn, device=DEV); fsx = SEL(S["s1"], ytr[pm], T["s1"], 10 + i); sh.append(read(fsx[fsx['primary']]['pred'], 'shuf')); sh[-1]['primary_kind'] = fsx['primary']
    ctr['shuffled_truth'] = dict(draws=sh, MAE_point=[x['MAE']['value'] for x in sh], ratio_to_A1_point=[x['MAE_over_A1']['value'] for x in sh], coverage_point=[x['coverage90']['value'] for x in sh])
    xp = (ytr + 0.01 * torch.randn(len(ytr), generator=gn, **T64))[:, None]; xpt = (yte + 0.01 * torch.randn(len(yte), generator=gn, **T64))[:, None]
    fp = SEL(xp, ytr, xpt, 3); ctr['positive_control_label_as_input'] = read(fp[fp['primary']]['pred'], 'pos'); ctr['positive_control_label_as_input']['primary_kind'] = fp['primary']
    res['controls'] = ctr
    nm, nr = ctr['noise_input']['MAE']['value'], ctr['noise_input']['MAE_over_A1']['value']; sm, sr = max(ctr['shuffled_truth']['MAE_point']), min(ctr['shuffled_truth']['ratio_to_A1_point'])
    pm_ = res['primary_S1']['MAE']['value']
    res['control_checks'] = dict(noise_fails_C1_C3_points=bool(nm > 0.03 and nr > 0.75), shuffled_fails_C1_C3_points_all_draws=bool(min(ctr['shuffled_truth']['MAE_point']) > 0.03 and sr > 0.75),
                                 noise_and_shuffled_not_better_than_real=bool(nm >= pm_ and min(ctr['shuffled_truth']['MAE_point']) >= pm_),
                                 positive_control_MAE_le_0p03=bool(ctr['positive_control_label_as_input']['MAE']['value'] <= 0.03))
    res['control_checks']['all_ok'] = bool(all(res['control_checks'].values()))
    # ---- Y_O (descriptive; the RHO of the test (0.5) lies between the training RHOs)
    fo_ = SEL(S["s1"], yotr, T["s1"], 4); qo = fo_[fo_['primary']]['pred']; mo = (qo[:, 2] - yote).abs()
    qco = torch.quantile(yotr, QT)[None].expand(len(yote), 5); mco = (qco[:, 2] - yote).abs()
    res['YO_descriptive'] = dict(primary_kind=fo_['primary'], MAE=boot(mo, None, prim), MAE_constant_quantiles=boot(mco, None, prim), coverage90=boot(((yote >= qo[:, 0]) & (yote <= qo[:, 4])).double(), None, prim), mean_truth=float(yote[prim].mean()))
    # ---- A4 where stored for the test participants (descriptive)
    a4 = {}
    for s in te_subs:
        p = ROOT / f'results/yield_projector/runner_T9R/arms_rho0.5/sub-{s:03d}.json'
        if p.exists(): a4[s] = json.loads(p.read_text())
    res['A4_stored_for_test_participants'] = sorted(a4.keys())
    res['runtime_s'] = time.time() - t0
    # ---- verdict
    jit = OUT / 'jitter.json'; J = json.loads(jit.read_text()) if (jit.exists() and not smoke) else None
    res['jitter_floor64'] = J['floor64'] if J else None
    res['VERDICT'] = ('VOID (jitter floor %.4f > 0.02)' % J['floor64'] if (J and J['thresholds_void']) else res['primary_S1_verdict']) if not smoke else 'SMOKE (no verdict)'
    res['VERDICT_requires_controls_ok'] = res['control_checks']['all_ok']
    fjson(OUT / ('summary_smoke.json' if smoke else 'summary.json'), res)
    log('VERDICT', res['VERDICT'], 'controls_ok', res['control_checks'])
    log('primary S1', json.dumps({k: res['primary_S1'][k] for k in ('MAE', 'coverage90', 'MAE_over_A1', 'A1_MAE', 'verdict_readings')}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--subjects', default=None); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    subs = [int(x) for x in a.subjects.split(',')] if a.subjects else list(range(1, 24))
    pr = OUT / 'PREREG.json'
    if not a.smoke and not pr.exists():                      # the reading rules are fixed in this docstring and stamped before the first non-smoke number
        fjson(pr, dict(written=time.strftime('%Y-%m-%d %H:%M:%S %Z'), code_sha256=CODE_SHA256, rules=__doc__, note='NOT citable; criteria fixed before the full run'))
    if a.stage == 'feats': stage_feats(subs, a.smoke)
    elif a.stage == 'gate': stage_gate(a.smoke)
    elif a.stage == 'jitter': stage_jitter(a.smoke)
    elif a.stage == 'fit': stage_fit(a.smoke, subs)
    else: raise SystemExit('unknown stage')


if __name__ == '__main__':
    main()
