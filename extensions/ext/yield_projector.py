"""YIELD PROJECTOR on planted cells, ds003458 (runner, 23 Nov 2026).
Implements outputs/FINAL/PROTOCOL_yield_projector.md (LOCKED), sections 2-10 and 14:
  sec 2   regeneration of the planted_v2 cells (same rng stream as gf_planted_calib_v2.py / pid_regen.load, with g_u, g_v exposed),
          overlap cells (cos 0, 0.3, 0.7) and the degraded-reference cell (kappa 0.1) at RHO 0.5, f 0.5
  sec 3   scorer = D2.pe_gain (nested calibration, 100 fits, widths 64 and 'full'), yield = gain - mean of 3 circular-shift floors;
          per-column rule: arms that use the target are scored by FOUR one-column calls, each transform closes over ONE target column
  sec 4   arms A0 (none), A1 (PE.eogreg), A2 (projector without the target-orthogonal step), A3 (projector, DECIDES), A3k1 (rank 1 fixed),
          A4 (target-adjusted regression), SAMEPROJ (mixed-cell maps applied to the neural-only features), A3_OCULAR (A3 on the
          ocular-only features with Os). A5 is SKIPPED (protocol sec 4).
  sec 5   gates G1a / G1b (1e-4 against the stored cells; a failure stops the run), regenerated u, v, y, sig == pid_regen.load, finite values
  sec 6   scope proxy E and tau -> scope_E.json, written before any arm is scored (the scoring stages refuse to start without it)
  sec 7-9 --summarize: paired participant differences, 95% participant bootstrap with ONE index tensor (seed 20261123), PASS / FAIL /
          UNDECIDED, the A0-must-fail rule, the decision-table row, gates, floors, k histograms, A4 against A3, overlap / kappa, step R
  sec 14  order of work and step R (projector over the registered read-outs R1 raw features and R2 REVE embedding on the REAL target)

Leakage rules (enforced by construction; checked numerically by --stage check):
  * the hook t(F, M) receives no target. Each transform closes over one training target column y_j; every use of y_j, of the eye
    matrix (A2, A3) and of the features in the FIT is multiplied by the training mask M[k], so rows outside M[k] cannot enter fit k.
  * A1 and A4 subtract Oc B for every row and therefore read test-row eye regressors (by protocol); A2 / A3 never do.
  * a floor call (j = 1..3) closes over the shifted column only.
Numerics: torch float64 on the GPU. Named CPU exceptions: (i) the numpy SVD of the regeneration (must be bit-identical to the
  generator), (ii) the 70 x 70 SVD of the QR factor R of O_p (cuSOLVER batched SVD is ~30x slower than LAPACK here), (iii) MNE in ocular_v2.
  Singular values of C and of the permuted C are taken from eigh of the 64 x 64 Gram C^T C (same route for both, so k compares like with like).
The projector maps (mu, sd, V S^-1 Q, V S Q per fit and column) are held in memory for the same-projector cell and are NOT written to disk
  (about 14 MB per call); per fit k, sigma_1..5 and theta are written. They are reproducible from the seeding rule.
Permutation seed: ((100 s + seed) * 1000 + cell_id) * 10 + j; cell_id = 10 * rho_index + f_index for mixed cells (rho 0.5, 0.9, 0.25 ->
  0, 1, 2; null, f0, .25, .5, .75 -> 0..4), + 100 for the ocular-only fit of the same cell, 200 + i for cos cells (300 + i ocular-only),
  400 kappa 0.1, 900 / 901 step R decoders R1 / R2 (seed 0). A2 uses the same seed as A3.

Usage (always through srun on a GPU node, cwd = project root):
  --stage cache|scopeE|base|arms|f0null|ovk|stepR|main|check  [--subjects 1,3] [--seeds 0,1] [--rho 0.9] [--P 199] [--root DIR]
          [--drop-attr 0.25,0.9] [--cos 0,0.3,0.7]
  --summarize [--root DIR]
"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "3")      # the step has at most 3 CPUs; without this the CPU LAPACK calls oversubscribe the node (measured 30 s for a 70 x 70 batched SVD)
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import d2_pe_tasks as D2
import pe_eogreg as PE

DEV = 'cuda'; LS = (64, 'full'); t0 = time.time()
log = lambda *m: print(f'[{time.time() - t0:8.1f}s]', *m, flush=True)
OUT = ROOT / 'results/yield_projector'
STORED = {r: ROOT / f'results/gpu_fill_20261123/planted_v2_rho{r}' for r in (0.25, 0.5, 0.9)}
ADD11 = {r: OUT / f'planted_v2_rho{r}_add11' for r in (0.25, 0.9)}
A0_AMP, B0_AMP, KAPPA, KPC, PC0 = 0.14, 0.45, 0.3, 32, 20          # generator settings, unchanged (protocol sec 2)
NPC, KMAX, TOL = 64, 5, 1e-4                                        # projector width, rank cap, gate tolerance
RHO_IX = {0.5: 0, 0.9: 1, 0.25: 2}; F_IX = {'null': 0, 0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4}; COS_IX = {0.0: 0, 0.3: 1, 0.7: 2}
T64 = dict(dtype=torch.float64, device=DEV)
G = lambda x: torch.as_tensor(np.ascontiguousarray(x), **T64)
cname = lambda f: 'null' if f == 'null' else f'f{f:g}'


def jdump(path, obj):
    """Atomic JSON write (a killed run never leaves a half-written participant file)."""
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def gate_fail(root, msg):
    """A failed gate stops the run: sentinel file (seen by the other shards) + exit."""
    with open(root / 'GATE_FAILED', 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')
    raise SystemExit('GATE FAIL: ' + msg)


def check_stop(root):
    if (root / 'GATE_FAILED').exists(): raise SystemExit('stopped: GATE_FAILED present in ' + str(root))


# ------------------------------------------------------------------ data
_OCULAR_RAW = PE.ocular_v2


def ocular_cached(s, tri0):
    """PE.ocular_v2 (reads the raw EEG with MNE: the slow CPU part), computed once per participant and stored bit-exactly."""
    p = OUT / 'cache' / f'ocular_v2_sub-{s:03d}.npz'
    if not p.exists():
        O, Ev, keep = _OCULAR_RAW(s, tri0); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(f'tmp{os.getpid()}_{p.name}')
        np.savez(tmp, O=O, Ev=Ev, keep=keep, tri0=tri0); os.replace(tmp, p)
    z = np.load(p); assert np.array_equal(z['tri0'], tri0), 'ocular cache: trial index mismatch'
    return z['O'], z['Ev'], z['keep']


PE.ocular_v2 = ocular_cached            # pid_regen.load (regeneration gate) then reads the same cached arrays instead of re-reading the raw EEG
import pid_regen as PR


def load_sub(s):
    """Features, kept trials and eye regressors exactly as gf_planted_calib_v2.run_sub lines 39-40, 47, 50."""
    fz = np.load(ROOT / f'results/features/raw/sub-{s:03d}.npz'); tri0 = fz['tri']; F0 = fz['F'].astype(np.float64)
    O, Ev, keep = ocular_cached(s, tri0); tri = tri0[keep]; F = F0[keep]; n = len(tri); O = O.astype(np.float64)
    Fsd = F.std(0) + 1e-9; Osd = O.std(0) + 1e-9
    Fz = (F - F.mean(0)) / Fsd; _, Sv, Vt = np.linalg.svd(Fz, full_matrices=False)
    return dict(s=s, F=F, O=O, tri=tri, tri0=tri0, keep=keep, n=n, Fsd=Fsd, Osd=Osd, Sv=Sv, Vt=Vt)


def regen(S, rho, seed):
    """Generator rng stream (draw order y, g_u, sg, g_v, e) with the coefficient vectors exposed."""
    n = S['n']; rng = np.random.default_rng(910000 + 100 * S['s'] + seed); y = rng.standard_normal(n)
    skd = S['Sv'][PC0:PC0 + KPC] / np.sqrt(n); Vk = S['Vt'][PC0:PC0 + KPC]
    pat = lambda g: ((g * skd) @ Vk) * S['Fsd']
    g_u = rng.standard_normal(KPC); u = pat(g_u); sg = rng.choice([-1.0, 1.0], S['O'].shape[1]); g_v = rng.standard_normal(KPC); v = pat(g_v)
    sig = rho * y + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return dict(y=y, u=u, v=v, sg=sg, sig=sig, g_u=g_u, g_v=g_v, pat=pat, rho=rho, seed=seed)


def regen_gate(S, R, root):
    """Protocol sec 2 / 5: u, v, y, sig equal pid_regen.load exactly."""
    ref = PR.load(S['s'], R['rho'], R['seed'])
    ok = all(np.array_equal(R[k], ref[k]) for k in ('u', 'v', 'y', 'sig')) and np.array_equal(S['F'], ref['F'])
    if not ok: gate_fail(root, f'regeneration != pid_regen.load (sub {S["s"]} rho {R["rho"]} seed {R["seed"]})')
    return True


def cell_data(S, R, f, cos=None, kappa=KAPPA):
    """Generator expressions (gf_planted_calib_v2.py lines 52-58, 65). cos: overlap cell (v replaced by v_c, no rng draw moved);
    kappa != KAPPA: degraded-reference cell (only Om changes; the features do not depend on kappa)."""
    F, O, Osd, y, u, sg, sig = S['F'], S['O'], S['Osd'], R['y'], R['u'], R['sg'], R['sig']; info = {}
    a, b = (0.0, 0.0) if f == 'null' else (A0_AMP * np.sqrt(1 - f), B0_AMP * np.sqrt(f))
    v = R['v']
    if cos is not None:
        gu, gv = R['g_u'], R['g_v']; uh = gu / np.linalg.norm(gu); vp = gv - (gv @ uh) * uh; vp = vp / np.linalg.norm(vp)
        gc = np.linalg.norm(gv) * (cos * uh + np.sqrt(1 - cos ** 2) * vp); v = R['pat'](gc)
        info = dict(cos_target=cos, cos_achieved=float(gc @ gu / (np.linalg.norm(gc) * np.linalg.norm(gu))), cos_stored_draw=float(gv @ gu / (np.linalg.norm(gv) * np.linalg.norm(gu))))
    W = np.outer(sg, v) / O.shape[1]
    dO = KAPPA * np.outer(sig, Osd * sg)
    Fn = F + a * np.outer(y, u); Fo = F + b * ((dO / (KAPPA * Osd)) @ W); Fm = F + a * np.outer(y, u) + b * ((dO / (KAPPA * Osd)) @ W)
    Om = O + (dO if (b > 0) else 0); Os = O + dO
    if kappa != KAPPA: Om = O + kappa * np.outer(sig, Osd * sg); info['kappa'] = kappa
    for x in (Fn, Fo, Fm, Om, Os): assert np.isfinite(x).all()
    return dict(f=f, a=float(a), b=float(b), Fn=Fn, Fo=Fo, Fm=Fm, Om=Om, Os=Os, info=info)


def ycols(y):
    n = len(y); return np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])


# ------------------------------------------------------------------ scorer (protocol sec 3)
def _pack(g, sec, calls):
    return dict(g=g, **{'yield': {L: g[L][0] - float(np.mean(g[L][1:])) for L in g}}, sec=round(sec, 3), calls=calls)


def call4(F, Ys, transform=None):
    """ONE four-column pe_gain call (arms that do not use the target). g[L] = mean over rows of the gain of [target, 3 shifts]."""
    t = time.time(); sc = D2.pe_gain(F, Ys, LS, transform=transform)
    g = {str(L): [float(x) for x in sc[L].mean(0)] for L in LS}; torch.cuda.empty_cache()
    return _pack(g, time.time() - t, 1)


def call1(F, Ys, make_t):
    """FOUR one-column pe_gain calls; make_t(j) builds the transform of call j from column j ONLY (per-column rule)."""
    t = time.time(); g = {str(L): [] for L in LS}
    for j in range(4):
        sc = D2.pe_gain(F, Ys[:, j:j + 1], LS, transform=make_t(j))
        for L in LS: g[str(L)].append(float(sc[L][:, 0].mean()))
        torch.cuda.empty_cache()
    return _pack(g, time.time() - t, 4)


# ------------------------------------------------------------------ projector (protocol sec 4, A3 steps 1-6)
_BASIS = {}


def fold_basis(F, M):
    """Steps 1-2 for all fits at once; depends on the features and the masks only, so it is shared by the columns and arms of a cell.
    Per fit k (T = rows with M[k]): mu, sd over T; Fz = (F - mu) / sd; top-64 right singular vectors V and singular values s of Fz[T]
    from the n x n Gram eigh; S = s / sqrt(n_k); whitened scores Z = Fz V S^-1 for ALL rows (unit variance on T)."""
    if _BASIS.get('F') is F and _BASIS.get('M_shape') == tuple(M.shape): return _BASIS['val']
    _BASIS.clear(); m = M.double(); cnt = m.sum(1)
    mu = (m @ F) / cnt[:, None]; Bf = F[None] - mu[:, None]; A = Bf * m[:, :, None]            # A: training rows only (others zero)
    sd = ((A ** 2).sum(1) / cnt[:, None]).sqrt() + 1e-6; Bf /= sd[:, None]; A /= sd[:, None]    # Bf = Fz for all rows; A = Fz[T], centred on T
    e, U = torch.linalg.eigh(A @ A.transpose(1, 2)); e = e.flip(-1)[:, :NPC]; U = U.flip(-1)[:, :, :NPC]
    s = e.clamp_min(0).sqrt(); assert bool((s[:, -1] > 1e-9 * s[:, 0]).all()), 'fewer than 64 non-zero components in a fold'
    V = (A.transpose(1, 2) @ U) / s[:, None, :]; del A, U                                      # (K, D, 64)
    Sd = s / cnt.sqrt()[:, None]; Z = (Bf @ V) / Sd[:, None, :]; del Bf                         # (K, n, 64), all rows
    val = dict(mu=mu, sd=sd, Z=Z, VSi=V / Sd[:, None, :], VS=V * Sd[:, None, :]); del V
    _BASIS.update(F=F, M_shape=tuple(M.shape), val=val); return val


def apply_map(F, mp):
    """Step 6 for ALL rows, original feature space: F_out = F - (((F - mu)/sd) W1) (W2^T) * sd, W1 = V S^-1 Q, W2 = V S Q. -> (K, n, D)."""
    Fz = (F[None] - mp['mu'][:, None]) / mp['sd'][:, None]; A = Fz @ mp['W1']; del Fz
    out = A @ mp['W2'].transpose(1, 2); out *= mp['sd'][:, None]; out.neg_().add_(F[None]); return out


def perm_threshold(Qo, Zp, M, P, seed, chunk=20):
    """Step 5: P row permutations of the TRAINING rows of Q_o; theta = the ceil(0.95 (P+1))-th smallest top singular value of
    Q_o[pi]^T Z_p. Rows are compacted (training rows first) so a permutation never moves a row across the mask."""
    K, n = M.shape; cnt = M.sum(1); base = torch.argsort((~M).to(torch.int8), dim=1, stable=True)
    Qc = Qo.gather(1, base[:, :, None].expand(-1, -1, Qo.shape[2])); Zc = Zp.gather(1, base[:, :, None].expand(-1, -1, Zp.shape[2]))
    pad = torch.arange(n, device=DEV)[None] >= cnt[:, None]
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed)); tops = []
    for c0 in range(0, P, chunk):
        keys = torch.stack([torch.rand(K, n, generator=gen, **T64) for _ in range(min(chunk, P - c0))])     # one draw per permutation: stream independent of chunk
        keys.masked_fill_(pad[None], 2.0); order = keys.argsort(-1); del keys                               # first n_k positions = a permutation of the training rows
        Qp = Qc[None].expand(order.shape[0], -1, -1, -1).gather(2, order[..., None].expand(-1, -1, -1, Qc.shape[2]))
        Cp = Qp.transpose(2, 3) @ Zc[None]; del Qp, order
        tops.append(torch.linalg.eigvalsh(Cp.transpose(2, 3) @ Cp)[..., -1].clamp_min(0).sqrt()); del Cp
    r = -(-(95 * (P + 1)) // 100)                                                                # ceil(0.95 (P+1)): 19 -> 19 (max), 99 -> 95, 199 -> 190
    return torch.cat(tops).kthvalue(r, dim=0).values


def projector(Oc, ycol, P, seed, rec, target_orth=True, kfix=None):
    """A3 (target_orth, permutation rank rule), A2 (target_orth False), A3k1 (kfix 1). Oc (n, 70) eye matrix, ycol (n,) the ONE
    target column of this call. Everything fitted below is multiplied by the training mask of its fit."""
    def t(F, M):
        B = fold_basis(F, M); m = M.double()[:, :, None]; K, n = M.shape
        D = torch.stack([torch.ones(n, **T64)] + ([ycol] if target_orth else []), 1); Dm = D[None] * m                 # step 3 design, training rows
        Gd = Dm.transpose(1, 2) @ Dm
        res = lambda X: (X - D[None] @ torch.linalg.solve(Gd, Dm.transpose(1, 2) @ X)) * m                             # residual on T, zero elsewhere
        Zp = res(B['Z']); Op = res(Oc[None].expand(K, -1, -1))
        Qr, R = torch.linalg.qr(Op); Ur, sr, _ = torch.linalg.svd(R.cpu()); keep = sr > 1e-8 * sr[:, :1]               # step 4: SVD of O_p = Qr (Ur sr Vr^T)
        Qo = Qr @ (Ur * keep[:, None, :]).to(DEV); del Qr, Op                                                          # orthonormal basis of col(O_p); zero on rows outside T
        C = Qo.transpose(1, 2) @ Zp; ev, q = torch.linalg.eigh(C.transpose(1, 2) @ C); sigma = ev.flip(-1).clamp_min(0).sqrt(); q = q.flip(-1)
        if kfix is not None: k = torch.full((K,), int(kfix), device=DEV); theta = torch.full((K,), float('nan'), **T64)
        else: theta = perm_threshold(Qo, Zp, M, P, seed); k = (sigma > theta[:, None]).sum(1).clamp(max=KMAX)          # step 5
        Qk = q[:, :, :KMAX] * (torch.arange(KMAX, device=DEV)[None] < k[:, None])[:, None, :].double()                # columns beyond k are zero (k = 0: identity)
        mp = dict(mu=B['mu'], sd=B['sd'], W1=B['VSi'] @ Qk, W2=B['VS'] @ Qk)
        rec['k'].append(k.cpu().tolist()); rec['sigma'].append([[round(float(x), 6) for x in r_] for r_ in sigma[:, :KMAX].cpu().tolist()])
        rec['theta'].append([None if x != x else round(float(x), 6) for x in theta.cpu().tolist()]); rec['rank_O'].append(keep.sum(1).tolist()); rec['maps'].append(mp)
        out = apply_map(F, mp); assert bool(torch.isfinite(out).all()); return out
    return t


def new_rec(): return dict(k=[], sigma=[], theta=[], rank_O=[], maps=[])


def rec_out(rec): return {k: rec[k] for k in ('k', 'sigma', 'theta', 'rank_O')}


def a4_transform(Oc, ycol):
    """A4: per fit least squares of F[T] on [1, Oc, y_j][T] (ridge 1e-8 as PE.eogreg); only the eye part Oc B_O is subtracted, for
    all rows (reads test-row eye regressors by protocol; the target column enters through training rows only)."""
    X = torch.cat([torch.ones(len(Oc), 1, **T64), Oc, ycol[:, None]], 1); p = Oc.shape[1]
    def t(F, M):
        m = M.double()[:, :, None]; Xm = X[None] * m
        Bc = torch.linalg.solve(Xm.transpose(1, 2) @ Xm + 1e-8 * torch.eye(X.shape[1], **T64), Xm.transpose(1, 2) @ (F[None] * m))
        return F[None] - Oc[None] @ Bc[:, 1:1 + p]
    return t


def scope_E(y, Oe):
    """Protocol sec 6: held-out R^2 of y from [1, Oe]; least squares on the training rows of the 10 outer folds (ridge 1e-8), pooled."""
    y = G(y); X = torch.cat([torch.ones(len(y), 1, **T64), G(Oe)], 1); M = D2.nested_masks(len(y))[0][:10]; m = M.double()
    Xm = X[None] * m[:, :, None]; c = torch.linalg.solve(Xm.transpose(1, 2) @ Xm + 1e-8 * torch.eye(X.shape[1], **T64), Xm.transpose(1, 2) @ (y[None] * m)[:, :, None])
    yh = (X[None] @ c)[..., 0]; yb = (m @ y) / m.sum(1); te = (~M).double()
    return float(1 - (((y[None] - yh) ** 2) * te).sum() / (((y[None] - yb[:, None]) ** 2) * te).sum())


def pseed(s, seed, cid, j): return ((100 * s + seed) * 1000 + cid) * 10 + j


# ------------------------------------------------------------------ stages
def meta(s, S, stage, a, **kw):
    return dict(NOT_CITABLE='until a blind bug review and audit', code_sha256=CODE_SHA256, protocol='outputs/FINAL/PROTOCOL_yield_projector.md', stage=stage,
                participant=s, n=S['n'], seeds=a.seeds, P=a.P, written=time.strftime('%Y-%m-%d %H:%M:%S %Z'), **kw)


def stored_cells(path):
    return {(c['seed'], c['f']): c for c in json.loads(path.read_text())['cells']}


def stage_scopeE(a):
    """scope_E.json: E per participant, seed and RHO (O' = O + dO, independent of f for f > 0) + the kappa 0.1 cell; tau = participant
    median of the seed-mean E at RHO 0.5. Written before any arm is scored."""
    dest = a.root / 'scope_E.json'
    if dest.exists(): log('scope_E.json exists'); return
    per = {}
    for s in a.subjects:
        S = load_sub(s); r = dict(n=S['n'], E={}, E_kappa0p1={})
        for rho in (0.25, 0.5, 0.9):
            es = []
            for sd in a.seeds:
                R = regen(S, rho, sd); es.append(scope_E(R['y'], S['O'] + KAPPA * np.outer(R['sig'], S['Osd'] * R['sg'])))
                if rho == 0.5: r['E_kappa0p1'][str(sd)] = scope_E(R['y'], S['O'] + 0.1 * np.outer(R['sig'], S['Osd'] * R['sg']))
            r['E'][str(rho)] = dict(seeds=es, mean=float(np.mean(es)))
        r['E_kappa0p1']['mean'] = float(np.mean([r['E_kappa0p1'][str(sd)] for sd in a.seeds])); per[f'sub-{s:03d}'] = r
        log(f'scopeE sub {s}', {k: round(v['mean'], 4) for k, v in r['E'].items()})
    e5 = [per[k]['E']['0.5']['mean'] for k in per]
    jdump(dest, dict(NOT_CITABLE='until a blind bug review and audit', code_sha256=CODE_SHA256, written=time.strftime('%Y-%m-%d %H:%M:%S %Z'), written_unix=time.time(),
                     rule='E = held-out R^2 of y from [1, O + dO], 10 outer folds, ridge 1e-8, pooled; participant E = mean of seeds; tau = participant median at RHO 0.5',
                     n_participants=len(per), complete=len(per) == 23, seeds=a.seeds, tau=float(np.median(e5)),
                     median_E={str(r): float(np.median([per[k]['E'][str(r)]['mean'] for k in per])) for r in (0.25, 0.5, 0.9)},
                     median_E_kappa0p1=float(np.median([per[k]['E_kappa0p1']['mean'] for k in per])), participants=per))
    log('scope_E.json written, tau', float(np.median(e5)))


def need_scope(a):
    if not (a.root / 'scope_E.json').exists(): raise SystemExit('scope_E.json missing: run --stage scopeE first (protocol sec 6)')


def stage_base(a, s, rho):
    """A0 (mix_all), raw NEURAL and OCULAR single-source cells, A1 by the 4-call route, on the stored cells; G1a / G1b at 1e-4.
    RHO 0.5 carries the null and f 0 cells (run once). At f 0 neural_only is the mixed cell and ocular_only is the null cell's features
    (a = 0 or b = 0 makes the feature matrices identical), so those values are copied, not recomputed."""
    dest = a.root / f'base_rho{rho}' / f'sub-{s:03d}.json'
    if dest.exists(): return
    need_scope(a); check_stop(a.root); t1 = time.time(); S = load_sub(s); add = None; torch.cuda.reset_peak_memory_stats()
    sp = STORED[rho] / f'sub-{s:03d}.json'
    if not sp.exists():                                                   # one of the 11 new participants (protocol sec 2)
        for w in range(240):
            if (ADD11[rho] / 'summary.json').exists(): break
            if (OUT / 'ADD11_FAILED').exists(): raise SystemExit('the add11 generator run failed (see scripts/slurm/outputs/yield_projector_add11.err)')
            if w == 0: log(f'waiting for {ADD11[rho]}/summary.json')
            time.sleep(60)
        sp = ADD11[rho] / f'sub-{s:03d}.json'; add = json.loads(sp.read_text()); ref5 = stored_cells(STORED[0.5] / f'sub-{s:03d}.json')
        d_n = max(abs(c['neural_only'][L] - ref5[(c['seed'], c['f'])]['neural_only'][L]) for c in add['cells'] for L in ('64', 'full'))
        if add['G1_abs_diff'] > 1e-9 or d_n > TOL: gate_fail(a.root, f'add11 sub {s} rho {rho}: G1 {add["G1_abs_diff"]:.2e}, neural_only vs stored rho0.5 {d_n:.2e}')
        add = dict(file=str(sp), G1_abs_diff=add['G1_abs_diff'], neural_only_vs_stored_rho0p5_max=d_n, code_sha256=None)
    st = stored_cells(sp); cells = []; gmax = dict(G1a=0.0, G1b=0.0)
    for sd in a.seeds:
        R = regen(S, rho, sd); regen_gate(S, R, a.root); Ys = G(ycols(R['y'])); nullA0 = None
        for f in ((['null', 0.0] if rho == 0.5 else []) + [0.25, 0.5, 0.75]):
            C = cell_data(S, R, f); arms = {}
            arms['A0'] = call4(G(C['Fm']), Ys)
            if f == 'null': nullA0 = arms['A0']
            arms['NEURAL'] = call4(G(C['Fn']), Ys) if C['b'] > 0 else dict(arms['A0'], copied_from='A0 (Fn == Fm when b = 0)', sec=0, calls=0)
            arms['OCULAR'] = call4(G(C['Fo']), Ys) if C['a'] > 0 and C['b'] > 0 else dict(nullA0, copied_from='null cell A0 (Fo == F when b = 0)', sec=0, calls=0)
            eo = PE.eogreg(C['Om']); arms['A1'] = call1(G(C['Fm']), Ys, lambda j: eo)
            ref = st[(sd, f)]; d = {}
            for key, arm in (('mix_all', 'A0'), ('neural_only', 'NEURAL'), ('ocular_only', 'OCULAR'), ('mix_eogreg', 'A1')):
                d[key] = {L: abs(arms[arm]['yield'][L] - ref[key][L]) for L in ('64', 'full')}
            g1a = max(max(d[k].values()) for k in ('mix_all', 'neural_only', 'ocular_only')); g1b = max(d['mix_eogreg'].values())
            gmax['G1a'] = max(gmax['G1a'], g1a); gmax['G1b'] = max(gmax['G1b'], g1b)
            cells.append(dict(seed=sd, cell=cname(f), f=f, rho=rho, a=C['a'], b=C['b'], arms=arms, gate_abs_diff=d, stored={k: ref[k] for k in d}))
            if add is None and (g1a > TOL or g1b > TOL): gate_fail(a.root, f'G1 sub {s} rho {rho} seed {sd} f {f}: G1a {g1a:.3e} G1b {g1b:.3e}')
        log(f'base rho {rho} sub {s} seed {sd} done ({time.time() - t1:.0f}s) G1a {gmax["G1a"]:.2e} G1b {gmax["G1b"]:.2e}')
    jdump(dest, dict(meta(s, S, 'base', a, rho=rho), stored_file=str(sp), stored_is_add11=add is not None, add11_gate=add, regen_equals_pid_regen=True,
                     gate=dict(tol=TOL, G1a_max_abs_diff=gmax['G1a'], G1b_max_abs_diff=gmax['G1b'], binding=add is None,
                               note='G1a/G1b bind on the stored participants; for the 11 new participants the binding gates are add11_gate and the differences here are informational'),
                     cells=cells, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_s=time.time() - t1))


def proj_arms(a, s, sd, cid, Fm, Fn, Fo, Om, Os, Ys, attr=True, single=True):
    """Projector arms of one cell. Fm / Om: the cell the projector is fitted on. Fn: neural-only features for the same-projector cell
    (None to skip). Fo / Os: ocular-only features and their eye matrix for A3_OCULAR (None to skip)."""
    arms = {}; col = lambda j: Ys[:, j].clone()
    r3 = new_rec(); arms['A3'] = dict(call1(Fm, Ys, lambda j: projector(Om, col(j), a.P, pseed(s, sd, cid, j), r3)), **rec_out(r3))
    if Fn is not None:
        arms['SAMEPROJ'] = call1(Fn, Ys, lambda j: (lambda F, M: apply_map(F, r3['maps'][j])))
    del r3['maps']
    if attr:
        r2 = new_rec(); arms['A2'] = dict(call1(Fm, Ys, lambda j: projector(Om, col(j), a.P, pseed(s, sd, cid, j), r2, target_orth=False)), **rec_out(r2)); del r2
        r1 = new_rec(); arms['A3k1'] = dict(call1(Fm, Ys, lambda j: projector(Om, col(j), 0, 0, r1, kfix=1)), **rec_out(r1)); del r1
        arms['A4'] = call1(Fm, Ys, lambda j: a4_transform(Om, col(j)))
    if Fo is not None:
        ro = new_rec(); arms['A3_OCULAR'] = dict(call1(Fo, Ys, lambda j: projector(Os, col(j), a.P, pseed(s, sd, cid + 100, j), ro)), **rec_out(ro)); del ro
    _BASIS.clear(); torch.cuda.empty_cache()
    return arms


def stage_arms(a, s, rho, f0null=False):
    """Projector arms on the mixed cells of one RHO (f > 0), or on the f 0 and null cells (RHO 0.5 stream)."""
    name = 'arms_f0null' if f0null else f'arms_rho{rho}'; dest = a.root / name / f'sub-{s:03d}.json'
    if dest.exists(): return
    need_scope(a); check_stop(a.root)
    if not (a.root / f'base_rho{rho}' / f'sub-{s:03d}.json').exists(): raise SystemExit(f'base file missing for sub {s} rho {rho}: gates must run before the arms')
    t1 = time.time(); S = load_sub(s); attr = f0null or (rho not in a.drop_attr); cells = []; torch.cuda.reset_peak_memory_stats()
    for sd in a.seeds:
        R = regen(S, rho, sd); Ys = G(ycols(R['y']))
        for f in (['null', 0.0] if f0null else [0.25, 0.5, 0.75]):
            C = cell_data(S, R, f); cid = 10 * RHO_IX[rho] + F_IX[f]; pl = (not f0null)
            arms = proj_arms(a, s, sd, cid, G(C['Fm']), G(C['Fn']) if pl else None, G(C['Fo']) if pl else None, G(C['Om']), G(C['Os']) if pl else None, Ys, attr=attr)
            cells.append(dict(seed=sd, cell=cname(f), f=f, rho=rho, cell_id=cid, arms=arms))
            log(f'{name} sub {s} seed {sd} {cname(f)}: ' + ' '.join(f'{k} {v["yield"]["64"]:+.4f} ({v["sec"]:.1f}s)' for k, v in arms.items()))
    jdump(dest, dict(meta(s, S, name, a, rho=rho), attribution_arms_run=attr, cells=cells, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_s=time.time() - t1))


def stage_ovk(a, s):
    """DESCRIPTIVE cells at RHO 0.5, f 0.5: overlap cos in a.cos (all arms; neural-only truth = the stored f 0.5 cell) and kappa 0.1
    (A1, A2, A3, A3k1, A4, same-projector; A0 and truth = the stored f 0.5 cell). A1 by one four-column call (G1b has shown the routes agree)."""
    dest = a.root / 'ovk' / f'sub-{s:03d}.json'
    if dest.exists(): return
    need_scope(a); check_stop(a.root); t1 = time.time(); S = load_sub(s); cells = []; torch.cuda.reset_peak_memory_stats()
    for sd in a.seeds:
        R = regen(S, 0.5, sd); Ys = G(ycols(R['y']))
        for c in a.cos:
            C = cell_data(S, R, 0.5, cos=c); cid = 200 + COS_IX[c]; Fm = G(C['Fm']); Fo = G(C['Fo'])
            arms = dict(A0=call4(Fm, Ys), OCULAR=call4(Fo, Ys), A1=call4(Fm, Ys, PE.eogreg(C['Om'])))
            arms.update(proj_arms(a, s, sd, cid, Fm, G(C['Fn']), Fo, G(C['Om']), G(C['Os']), Ys))
            cells.append(dict(seed=sd, cell=f'cos{c:g}', f=0.5, rho=0.5, cell_id=cid, info=C['info'], arms=arms))
        C = cell_data(S, R, 0.5, kappa=0.1); Fm = G(C['Fm'])
        arms = dict(A1=call4(Fm, Ys, PE.eogreg(C['Om']))); arms.update(proj_arms(a, s, sd, 400, Fm, G(C['Fn']), None, G(C['Om']), None, Ys))
        cells.append(dict(seed=sd, cell='kappa0.1', f=0.5, rho=0.5, cell_id=400, info=C['info'], arms=arms))
        log(f'ovk sub {s} seed {sd} done ({time.time() - t1:.0f}s)')
    jdump(dest, dict(meta(s, S, 'ovk', a, rho=0.5), cos=a.cos, cells=cells, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_s=time.time() - t1))


_INFO = {}


def stage_stepR(a, s):
    """Protocol sec 14.2 (DESCRIPTIVE): projector over the registered read-outs on the REAL target. R1 = raw features, R2 = REVE embedding."""
    dest = a.root / 'stepR' / f'sub-{s:03d}.json'
    if dest.exists(): return
    need_scope(a); check_stop(a.root); t1 = time.time(); S = load_sub(s); torch.cuda.reset_peak_memory_stats()
    if 'i' not in _INFO: _INFO['i'] = D2.dataset('ds003458')
    y = _INFO['i']['qtil'](s, S['tri']).astype(np.float64); Ys = G(ycols(y)); O = S['O']; Og = G(O); n = S['n']
    rz = np.load(ROOT / f'results/features/reve/sub-{s:03d}.npz'); assert np.array_equal(rz['tri'], S['tri0']), 'REVE trial index != raw trial index'
    saved = json.loads((ROOT / f'results/ext/eogreg_v2/eogreg_sub-{s:03d}.json').read_text())['rows']
    res = dict(meta(s, S, 'stepR', a), E_real=scope_E(y, O), decoders={}, A1_perm_rule='PE.permuted(O, 500) (registered row-permuted regressors, seed 500)')
    for di, (dn, F) in enumerate((('R1_raw', S['F']), ('R2_reve', rz['F'].astype(np.float64)[S['keep']]))):
        Fg = G(F); col = lambda j: Ys[:, j].clone(); arms = {}
        arms['A0'] = call4(Fg, Ys); arms['A1'] = call4(Fg, Ys, PE.eogreg(O)); arms['A1_PERM'] = call4(Fg, Ys, PE.eogreg(PE.permuted(O, 500)))
        r3 = new_rec(); arms['A3'] = dict(call1(Fg, Ys, lambda j: projector(Og, col(j), a.P, pseed(s, 0, 900 + di, j), r3)), **rec_out(r3)); del r3
        arms['A4'] = call1(Fg, Ys, lambda j: a4_transform(Og, col(j))); _BASIS.clear()
        d = dict(n_features=int(F.shape[1]), cell_id=900 + di, arms=arms)
        if dn == 'R1_raw':
            g0 = abs(arms['A0']['g']['64'][0] - saved['all']['64']['gain']); g1 = abs(arms['A1']['g']['64'][0] - saved['all_eogreg']['64']['gain']) if 'all_eogreg' in saved else None
            d['GR1'] = dict(all_gain_abs_diff=g0, tol_all=1e-9, A1_gain_abs_diff=g1, tol_A1=TOL, A1_key='rows.all_eogreg.64.gain')
            if g0 > 1e-9 or (g1 is not None and g1 > TOL): gate_fail(a.root, f'GR1 sub {s}: all {g0:.2e} A1 {g1}')
        res['decoders'][dn] = d
    res.update(peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_s=time.time() - t1); jdump(dest, res)
    log(f'stepR sub {s} done ({time.time() - t1:.0f}s)')


def stage_main(a):
    """Order of protocol sec 14.1 for this shard's participants (scope_E.json must already exist): gates on the stored cells; RHO 0.5
    all cells and arms; f 0 and null; RHO 0.9; RHO 0.25; overlap and kappa cells. Step R is launched last by the driver."""
    need_scope(a)
    for s in a.subjects: stage_base(a, s, 0.5)
    for rho in (0.9, 0.25):
        for s in a.subjects:
            if (STORED[rho] / f'sub-{s:03d}.json').exists(): stage_base(a, s, rho)
    for s in a.subjects: stage_arms(a, s, 0.5)
    for s in a.subjects: stage_arms(a, s, 0.5, f0null=True)
    for rho in (0.9, 0.25):
        for s in a.subjects: stage_base(a, s, rho); stage_arms(a, s, rho)
    for s in a.subjects: stage_ovk(a, s)


def stage_check(a):
    """Code checks and timing on ONE cell (participant a.subjects[0], RHO a.rho, seed 0, f 0.5). Sanity only: prints no arm yield except
    the k = 0 identity difference. (1) A3 with k forced to 0 equals A0; (2) a projector fitted on a shuffled training target differs
    from A3; (3) leakage: changing the target / eye rows OUTSIDE a fit's training mask leaves that fit's output unchanged;
    (4) permutation check: top singular values from eigvalsh equal torch.linalg.svdvals; (5) seconds per call for P = 19, 99, 199."""
    s = a.subjects[0]; S = load_sub(s); R = regen(S, a.rho, 0); C = cell_data(S, R, 0.5); Ys = G(ycols(R['y'])); Fm = G(C['Fm']); Om = G(C['Om'])
    M = D2.nested_masks(S['n'])[0]; out = dict(code_sha256=CODE_SHA256, participant=s, rho=a.rho)
    a0 = call4(Fm, Ys); r = new_rec(); z = call1(Fm, Ys, lambda j: projector(Om, Ys[:, j].clone(), 0, 0, r, kfix=0))
    out['k0_identity_max_abs_diff_vs_A0'] = max(abs(z['g'][L][j] - a0['g'][L][j]) for L in z['g'] for j in range(4))
    y0 = Ys[:, 0].clone(); r = new_rec(); X3 = projector(Om, y0, 99, 12345, r)(Fm, M); k3 = r['k'][0]
    gp = torch.Generator(device=DEV); gp.manual_seed(7); ysh = y0[torch.randperm(len(y0), generator=gp, device=DEV)]
    r = new_rec(); Xs = projector(Om, ysh, 99, 12345, r)(Fm, M)
    out['shuffled_target_projector'] = dict(rel_frobenius_diff_vs_A3=float((Xs - X3).norm() / (X3 - Fm[None]).norm().clamp_min(1e-300)), k_hist_A3=np.bincount(k3, minlength=6).tolist(), k_hist_shuffled=np.bincount(r['k'][0], minlength=6).tolist())
    del Xs; leak = {}
    for kf in (0, 37):                                                     # one outer and one inner fit
        te = ~M[kf]; yb = y0.clone(); yb[te] = 5.0 + torch.arange(int(te.sum()), **T64); Ob = Om.clone(); Ob[te] = Ob[te] * -3.0 + 11.0
        r = new_rec(); Xa = projector(Om, yb, 99, 12345, r)(Fm, M)[kf]; leak[f'A3_fit{kf}_target_rows_outside_T'] = float((Xa - X3[kf]).abs().max())
        r = new_rec(); Xa = projector(Ob, y0, 99, 12345, r)(Fm, M)[kf]; leak[f'A3_fit{kf}_eye_rows_outside_T'] = float((Xa - X3[kf]).abs().max())
        r = new_rec(); X2 = projector(Om, y0, 99, 12345, r, target_orth=False)(Fm, M)[kf]
        r = new_rec(); Xa = projector(Ob, yb, 99, 12345, r, target_orth=False)(Fm, M)[kf]; leak[f'A2_fit{kf}_target_and_eye_rows_outside_T'] = float((Xa - X2).abs().max())
        X4 = a4_transform(Om, y0)(Fm, M)[kf]; Xa = a4_transform(Om, yb)(Fm, M)[kf]; leak[f'A4_fit{kf}_target_rows_outside_T'] = float((Xa - X4).abs().max())
        Fb = Fm.clone(); Fb[te] = Fb[te] * 2.0 + 1.0; _BASIS.clear(); r = new_rec(); projector(Om, y0, 99, 12345, r)(Fb, M)
        r0 = new_rec(); _BASIS.clear(); projector(Om, y0, 99, 12345, r0)(Fm, M)
        leak[f'A3_fit{kf}_map_feature_rows_outside_T'] = max(float((r['maps'][0][key][kf] - r0['maps'][0][key][kf]).abs().max()) for key in ('mu', 'sd', 'W1', 'W2'))
        del Xa, X2, X4, Fb
    out['leakage_max_abs_change'] = leak; del X3
    B = fold_basis(Fm, M); out['whitened_train_variance_range'] = [float(x) for x in (((B['Z'] ** 2) * M.double()[:, :, None]).sum(1) / M.sum(1)[:, None]).aminmax()]
    Cc = torch.randn(300, 70, 64, generator=gp, **T64); out['eigvalsh_vs_svdvals_max_abs_diff'] = float((torch.linalg.eigvalsh(Cc.transpose(1, 2) @ Cc)[:, -1].sqrt() - torch.linalg.svdvals(Cc)[:, 0]).abs().max())
    tim = {}
    for P in (19, 99, 199):
        ts = []
        for rep in range(2):
            r = new_rec(); torch.cuda.synchronize(); t = time.time(); D2.pe_gain(Fm, Ys[:, :1], LS, transform=projector(Om, y0, P, 1, r)); torch.cuda.synchronize(); ts.append(time.time() - t)
        tim[f'A3_P{P}_sec_per_one_column_call'] = min(ts)
    torch.cuda.synchronize(); t = time.time(); D2.pe_gain(Fm, Ys[:, :1], LS); torch.cuda.synchronize(); tim['A0_sec_per_one_column_call'] = time.time() - t
    out['timing'] = tim; out['peak_gpu_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
    jdump(a.root / f'check_sub-{s:03d}.json', out); log(json.dumps(out, indent=1))


# ------------------------------------------------------------------ summariser (protocol sec 5, 7, 8, 9, 14.2)
def summarize(a):
    root = a.root; rd = lambda d: {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted((root / d).glob('sub-*.json'))}
    stages = ['base_rho0.5', 'base_rho0.9', 'base_rho0.25', 'arms_rho0.5', 'arms_f0null', 'arms_rho0.9', 'arms_rho0.25', 'ovk', 'stepR']
    FL = {st: rd(st) for st in stages}; seeds = a.seeds
    # flat table: T[(rho, cell)][arm][participant][seed] = arm dict.  f 0 and null live at rho 0.5 only.
    T = {}
    for st in stages[:-1]:
        for s, r in FL[st].items():
            for c in r['cells']:
                for arm, v in c['arms'].items(): T.setdefault((c['rho'], c['cell']), {}).setdefault(arm, {}).setdefault(s, {})[c['seed']] = v
    for k in ('cos0', 'cos0.3', 'cos0.7', 'kappa0.1'):                      # truth (and A0 for kappa) of the descriptive cells = the stored f 0.5 cell
        if (0.5, k) in T and (0.5, 'f0.5') in T:
            T[(0.5, k)]['NEURAL'] = T[(0.5, 'f0.5')]['NEURAL']
            if k == 'kappa0.1': T[(0.5, k)]['A0'] = T[(0.5, 'f0.5')]['A0']
    allsub = sorted(set().union(*[set(v) for v in FL.values()])) if any(FL.values()) else []
    complete = all(sorted(FL[st]) == list(range(1, 24)) for st in stages)
    nB = 23 if complete else max(1, len(allsub)); g = torch.Generator(device=DEV); g.manual_seed(20261123)
    IXfull = torch.randint(0, nB, (10000, nB), generator=g, device=DEV)     # ONE index tensor, reused for every cell, arm and criterion
    qt = torch.tensor([.025, .975], **T64)

    def vec(key, arm, fn):
        """{participant: mean over seeds of fn(arm dict)} for participants that have every seed."""
        d = T.get(key, {}).get(arm, {}); return {s: float(np.mean([fn(d[s][sd]) for sd in seeds])) for s in d if all(sd in d[s] for sd in seeds)}

    def ix_for(subs):
        if complete and len(subs) == 23: return IXfull
        gg = torch.Generator(device=DEV); gg.manual_seed(20261123); return torch.randint(0, len(subs), (10000, len(subs)), generator=gg, device=DEV)

    def boot(d):
        """Mean and 95% participant-bootstrap interval of {participant: value}."""
        if not d: return None
        subs = sorted(d); v = G([d[s] for s in subs]); m = v[ix_for(subs)].mean(1)
        return dict(mean=float(v.mean()), ci95=torch.quantile(m, qt).cpu().tolist(), n=len(subs), sd=float(v.std(unbiased=True)) if len(subs) > 1 else None)

    def bratio(num, den):
        subs = sorted(set(num) & set(den))
        if not subs: return None
        nu, de = G([num[s] for s in subs]), G([den[s] for s in subs]); IX = ix_for(subs); r = nu[IX].mean(1) / de[IX].mean(1)
        return dict(ratio=float(nu.mean() / de.mean()), ci95=torch.quantile(r, qt).cpu().tolist(), n=len(subs))

    def paired(key, arm, key2, arm2, L, absdiff_vs=None):
        x, y = vec(key, arm, lambda v: v['yield'][L]), vec(key2, arm2, lambda v: v['yield'][L]); return boot({s: x[s] - y[s] for s in set(x) & set(y)})

    r_abs = lambda b, bound: None if b is None else ('PASS' if (b['ci95'][0] >= -bound and b['ci95'][1] <= bound) else 'FAIL' if (b['ci95'][1] < -bound or b['ci95'][0] > bound) else 'UNDECIDED')
    r_d = lambda b: None if b is None else ('PASS' if (b['ci95'][0] >= -0.02 and b['ci95'][1] <= 0.02) else 'FAIL' if b['ci95'][0] > 0.02 else 'UNDECIDED')
    r_b = lambda b: None if b is None else ('PASS' if b['ci95'][0] >= 0.90 else 'FAIL' if b['ci95'][1] < 0.90 else 'UNDECIDED')

    def khist(key, arm):
        d = T.get(key, {}).get(arm, {}); ks = [v['k'] for sv in d.values() for v in sv.values() if 'k' in v]
        if not ks: return None
        tg = np.bincount(np.concatenate([k[0] for k in ks]), minlength=KMAX + 1); sh = np.bincount(np.concatenate([np.ravel(k[1:]) for k in ks]), minlength=KMAX + 1)
        per = {s: float(np.mean([np.mean(np.asarray(v['k'][0]) >= 1) for v in sv.values()])) for s, sv in d.items()}
        return dict(target=tg.tolist(), shifted=sh.tolist(), share_target_fits_k_ge1=float(tg[1:].sum() / tg.sum()), share_shifted_fits_k_ge1=float(sh[1:].sum() / sh.sum()), per_participant_share_target_k_ge1=per)

    def credit(reading, a0_reading, kh):
        """A0-must-fail rule (ruling D)."""
        binding = a0_reading == 'FAIL'; applied = bool(kh and kh['share_target_fits_k_ge1'] >= 0.5)
        status = reading if reading != 'PASS' else ('PASS' if binding and applied else 'NON-BINDING' if not binding else 'NO CORRECTION APPLIED')
        return dict(reading=reading, A0_reading=a0_reading, binding=binding, correction_applied=applied, credited=status)

    S = dict(NOT_CITABLE='until a blind bug review and audit', protocol='outputs/FINAL/PROTOCOL_yield_projector.md', code_sha256_summariser=CODE_SHA256, written=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
             root=str(root), seeds=seeds, complete=complete, RESULT_STATUS='full grid present' if complete else 'PARTIAL DATA: code-path output only, NOT a result',
             present={st: sorted(FL[st]) for st in stages}, cells_present=sorted(f'rho{k[0]}|{k[1]}: ' + ','.join(sorted(v)) for k, v in T.items()),
             code_sha256_in_files=sorted({r['code_sha256'] for v in FL.values() for r in v.values()}), P_in_files=sorted({r['P'] for v in FL.values() for r in v.values()}),
             attribution_arms_run={st: sorted({r.get('attribution_arms_run') for r in FL[st].values()}, key=str) for st in stages if st.startswith('arms')},
             A5='SKIPPED (protocol sec 4: needs a nuisance reduction that is not in the paper)',
             bootstrap='participant, 10000 resamples, torch GPU, seed 20261123, one (10000 x 23) index tensor' + ('' if complete else ' [PARTIAL: index tensor over the participants present]'),
             multiplicity='no correction claimed (protocol sec 9); no family-wise claim',
             read_out_note='the read-out re-standardises and re-decomposes the projected features, so removed directions are replaced by PCs 65 and up at L 64 (protocol sec 3)')
    # ---- gates
    gate = dict(tol=TOL, sentinel_GATE_FAILED=(root / 'GATE_FAILED').exists(), G1=[], add11=[])
    for rho in (0.5, 0.9, 0.25):
        b = FL[f'base_rho{rho}']; bind = {s: r for s, r in b.items() if r['gate']['binding']}; d = dict(rho=rho, n_stored_participants=len(bind), expected=23 if rho == 0.5 else 12)
        d['G1a_max_abs_diff'] = max([r['gate']['G1a_max_abs_diff'] for r in bind.values()], default=None); d['G1b_max_abs_diff'] = max([r['gate']['G1b_max_abs_diff'] for r in bind.values()], default=None)
        if rho != 0.5:                                                     # null and f 0 of the stored RHO files against the RHO 0.5 stream values (run once)
            mx = 0.0; nchk = 0
            for s in bind:
                if s not in FL['base_rho0.5'] or not (STORED[rho] / f'sub-{s:03d}.json').exists(): continue
                st = stored_cells(STORED[rho] / f'sub-{s:03d}.json')
                for c in FL['base_rho0.5'][s]['cells']:
                    if c['cell'] in ('null', 'f0'):
                        for key, arm in (('mix_all', 'A0'), ('neural_only', 'NEURAL'), ('ocular_only', 'OCULAR'), ('mix_eogreg', 'A1')):
                            for L in ('64', 'full'): mx = max(mx, abs(c['arms'][arm]['yield'][L] - st[(c['seed'], c['f'])][key][L])); nchk += 1
            d['null_f0_from_rho0p5_stream_max_abs_diff'] = mx; d['null_f0_values_checked'] = nchk
        vals = [d[k] for k in ('G1a_max_abs_diff', 'G1b_max_abs_diff', 'null_f0_from_rho0p5_stream_max_abs_diff') if k in d]
        d['pass'] = bool(len(bind) == d['expected'] and all(v is not None and v <= TOL for v in vals)); gate['G1'].append(d)
        for s, r in b.items():
            if not r['gate']['binding']: gate['add11'].append(dict(rho=rho, participant=s, **r['add11_gate'], ours_vs_add11_G1a=r['gate']['G1a_max_abs_diff'], ours_vs_add11_G1b=r['gate']['G1b_max_abs_diff']))
    gate['add11_pass'] = bool(len(gate['add11']) == 22 and all(x['G1_abs_diff'] <= 1e-9 and x['neural_only_vs_stored_rho0p5_max'] <= TOL for x in gate['add11']))
    gate['regen_equals_pid_regen'] = bool(all(r['regen_equals_pid_regen'] for st in stages[:3] for r in FL[st].values()))
    allv = [x for arms in T.values() for d in arms.values() for sv in d.values() for v in sv.values() for L in v['g'] for x in v['g'][L]]
    gate['finite'] = bool(np.isfinite(allv).all()) if allv else None
    gate['scope_E_written_before_first_arm'] = None
    if (root / 'scope_E.json').exists():
        E = json.loads((root / 'scope_E.json').read_text()); S['scope_E'] = {k: E[k] for k in ('written', 'tau', 'median_E', 'median_E_kappa0p1', 'n_participants', 'complete')}; tau = E['tau']
        first = min([p.stat().st_mtime for st in stages for p in (root / st).glob('sub-*.json')], default=None)
        gate['scope_E_written_before_first_arm'] = bool(first is None or E['written_unix'] < first)
    else: E, tau = None, None
    # ---- floors (every cell and arm, each shifted column, |participant mean gain| <= 0.01) and readings
    floors = dict(bound=0.01, violations=[], max_abs=0.0); KH = {}; RD = {}
    for key in sorted(T, key=str):
        for arm in T[key]:
            for L in ('64', 'full'):
                for j in (1, 2, 3):
                    v = vec(key, arm, lambda x: x['g'][L][j])
                    if v:
                        mval = float(np.mean(list(v.values()))); floors['max_abs'] = max(floors['max_abs'], abs(mval))
                        if abs(mval) > 0.01: floors['violations'].append(dict(rho=key[0], cell=key[1], arm=arm, L=L, shift=j, mean_gain=mval, n=len(v)))
            kh = khist(key, arm)
            if kh: KH[f'rho{key[0]}|{key[1]}|{arm}'] = kh
    floors['pass'] = not floors['violations']; S['floors'] = floors; S['k_histograms'] = KH
    for L in ('64', 'full'):
        Y = lambda key, arm: vec(key, arm, lambda v: v['yield'][L]); R_ = RD[L] = dict(cells={}, a_per_rho={})
        for key in sorted(T, key=str):
            rho, cell = key; arms = T[key]; row = dict(yields={arm: boot(Y(key, arm)) for arm in arms})
            if 'NEURAL' in arms and cell not in ('null', 'f0'):
                for arm in ('A0', 'A1', 'A2', 'A3', 'A3k1', 'A4', 'SAMEPROJ'):
                    if arm in arms: b = paired(key, arm, key, 'NEURAL', L); row[f'{arm}_minus_neural'] = dict(b, reading_abs_0p03=r_abs(b, 0.03)) if b else None
                a0r = (row.get('A0_minus_neural') or {}).get('reading_abs_0p03'); kh = KH.get(f'rho{rho}|{cell}|A3')
                if row.get('A3_minus_neural'):
                    row['a'] = dict(credit(row['A3_minus_neural']['reading_abs_0p03'], a0r, kh), ratio_A3_over_neural=bratio(Y(key, 'A3'), Y(key, 'NEURAL')))
                    if 'SAMEPROJ' in arms:
                        b = paired(key, 'A3', key, 'SAMEPROJ', L); row['a_prime'] = dict(credit(r_abs(b, 0.03), a0r, kh), A3_minus_sameproj=b, note="(a') alone never gives a PASS",
                                                                                          eye_leak_A3_minus_sameproj=b, projector_loss_sameproj_minus_neural=paired(key, 'SAMEPROJ', key, 'NEURAL', L))
                    if 'A4' in arms:
                        n_, x3, x4 = Y(key, 'NEURAL'), Y(key, 'A3'), Y(key, 'A4'); b = boot({s: abs(x4[s] - n_[s]) - abs(x3[s] - n_[s]) for s in set(n_) & set(x3) & set(x4)})
                        r4 = row['A4_minus_neural']['reading_abs_0p03']; r3 = row['A3_minus_neural']['reading_abs_0p03']
                        row['A4_vs_A3'] = dict(absA4_minus_absA3=b, definition='per participant |seed-mean(A4 - neural)| - |seed-mean(A3 - neural)|', A4_reading=r4, A3_reading=r3,
                                               A4_closer_than_A3=bool(b['ci95'][1] < 0 or (r4 == 'PASS' and r3 != 'PASS')))
            if 'A3_OCULAR' in arms:
                b3 = boot(Y(key, 'A3_OCULAR')); b0 = boot(Y(key, 'OCULAR')); row['d'] = dict(credit(r_d(b3), r_d(b0), KH.get(f'rho{rho}|{cell}|A3_OCULAR')), A3_ocular=b3, untreated_ocular=b0, read=(rho != 0.25))
            if cell == 'f0' and 'A3' in arms:
                row['b'] = {arm: dict(bratio(Y(key, arm), Y(key, 'A0')), reading_ge_0p90=r_b(bratio(Y(key, arm), Y(key, 'A0')))) for arm in ('A3', 'A1', 'A2', 'A3k1', 'A4') if arm in arms and bratio(Y(key, arm), Y(key, 'A0'))}
            if cell == 'null':
                row['G2'] = {arm: dict(b, ok=bool(abs(b['mean']) <= 0.02 and b['ci95'][0] <= 0 <= b['ci95'][1])) for arm, b in row['yields'].items() if b}
                if 'A3' in row['G2']: row['c'] = 'PASS' if row['G2']['A3']['ok'] else 'FAIL'
            R_['cells'][f'rho{rho}|{cell}'] = row
        for rho in (0.5, 0.9, 0.25):
            cr = [R_['cells'].get(f'rho{rho}|f{f:g}', {}).get('a') for f in (0.25, 0.5, 0.75)]
            if any(c is None for c in cr): R_['a_per_rho'][str(rho)] = None; continue
            nb = [c for c in cr if c['binding']]
            R_['a_per_rho'][str(rho)] = dict(reading='FAIL' if any(c['reading'] == 'FAIL' for c in cr) else 'PASS' if (nb and all(c['credited'] == 'PASS' for c in nb)) else 'UNDECIDED',
                                             n_binding=len(nb), credited=[c['credited'] for c in cr], role={0.5: 'enters the decision', 0.9: 'identifiability bound (predicted FAIL), never widens the scope', 0.25: 'expected non-binding: harm check only'}[rho])
    S['gates'] = gate; S['readings'] = RD
    S['G2_null_all_arms_pass'] = {L: (all(v['ok'] for v in RD[L]['cells']['rho0.5|null']['G2'].values()) if 'rho0.5|null' in RD[L]['cells'] else None) for L in RD}
    # ---- decision-table row (RHO 0.5, L 64 primary; full width shown, never resolves)
    def decide(L):
        C = RD[L]['cells']; need = [f'rho0.5|f{f:g}' for f in (0.25, 0.5, 0.75)]
        if not all(k in C and 'a' in C[k] and 'a_prime' in C[k] and 'd' in C[k] for k in need) or 'b' not in C.get('rho0.5|f0', {}) or 'c' not in C.get('rho0.5|null', {}): return dict(row=None, reason='cells missing')
        a_ = [C[k]['a'] for k in need]; ap = [C[k]['a_prime'] for k in need]; d_ = [C[k]['d'] for k in need]; b_ = C['rho0.5|f0']['b']['A3']['reading_ge_0p90']; c_ = C['rho0.5|null']['c']
        inp = dict(a=[x['credited'] for x in a_], a_prime=[x['reading'] for x in ap], d=[x['credited'] for x in d_], b=b_, c=c_)
        a_fail = [i for i, x in enumerate(a_) if x['reading'] == 'FAIL']; d_fail = any(x['reading'] == 'FAIL' for x in d_); und = 'UNDECIDED' in inp['a'] + inp['d'] + [b_]
        if any(ap[i]['reading'] != 'PASS' for i in a_fail) or d_fail or c_ == 'FAIL': row = 'PIVOT'
        elif a_fail: row = 'REFINE (a FAIL where a-prime PASSES; projector strips background eye directions)'
        elif b_ == 'FAIL': row = 'REFINE (b FAIL, nothing else fails)'
        elif all(x == 'PASS' for x in inp['a'] + inp['d'] + [b_, c_]): row = 'PROCEED'
        elif und: row = 'UNDECIDED and no FAIL: ' + ('extension to seeds 2,3 is triggered (one look, protocol sec 9)' if len(seeds) == 2 else 'REFINE (after the extension)')
        else: row = 'no table row: no FAIL, no UNDECIDED, but a cell is NON-BINDING or NO CORRECTION APPLIED (not a PASS)'
        return dict(row=row, inputs=inp)
    gates_ok = bool(all(x['pass'] for x in gate['G1']) and gate['add11_pass'] and gate['regen_equals_pid_regen'] and gate['finite'] and floors['pass'] and not gate['sentinel_GATE_FAILED']
                    and all(v for v in S['G2_null_all_arms_pass'].values()) and gate['scope_E_written_before_first_arm'])
    S['gates_all_pass'] = gates_ok
    S['decision'] = dict(applies_to='RHO 0.5, L 64 (primary)', valid=bool(complete and gates_ok), why_invalid=None if (complete and gates_ok) else ('partial data' if not complete else 'a gate failed (G1 / add11 / regen / finite / floors / G2 / scope_E order)'),
                         L64=decide('64'), Lfull_second_readout=decide('full'), widths_agree=decide('64').get('row') == decide('full').get('row'),
                         known_weakness='at f .75 the truth is 0.032 bit and A1 already sits inside 0.03, so only f .25 and f .5 discriminate (protocol sec 7)',
                         never_change_verdict='overlap, kappa 0.1, A2, A3k1, A4 and full width (protocol sec 8)')
    # ---- step R (descriptive)
    if FL['stepR']:
        SR = dict(note='DESCRIPTIVE, no truth on real data, cannot PASS or FAIL (protocol sec 14.2)', n=len(FL['stepR']), decoders={})
        Er = {s: r['E_real'] for s, r in FL['stepR'].items()}
        SR['E_real'] = dict(per_participant=Er, median=float(np.median(list(Er.values()))), tau=tau, median_le_tau=None if tau is None else bool(np.median(list(Er.values())) <= tau))
        for dn in ('R1_raw', 'R2_reve'):
            D_ = {s: r['decoders'][dn]['arms'] for s, r in FL['stepR'].items()}; dd = {}
            kshare = {s: float(np.mean(np.asarray(v['A3']['k'][0]) >= 1)) for s, v in D_.items()}; kall = np.concatenate([v['A3']['k'][0] for v in D_.values()])
            for L in ('64', 'full'):
                Yv = lambda arm: {s: v[arm]['yield'][L] for s, v in D_.items()}; df = lambda x, y_: boot({s: Yv(x)[s] - Yv(y_)[s] for s in D_})
                fl = {arm: [float(np.mean([v[arm]['g'][L][j] for v in D_.values()])) for j in (1, 2, 3)] for arm in ('A0', 'A1', 'A1_PERM', 'A3', 'A4')}
                dd[L] = dict(yields={arm: boot(Yv(arm)) for arm in ('A0', 'A1', 'A1_PERM', 'A3', 'A4')}, A3_minus_A0=df('A3', 'A0'), A3_minus_A1=df('A3', 'A1'), A3_minus_A4=df('A3', 'A4'),
                             shifted_floor_means=fl, floors_within_0p01=bool(all(abs(x) <= 0.01 for v in fl.values() for x in v)))
            n_ok = sum(1 for s in D_ if kshare[s] >= 0.5 and tau is not None and Er[s] <= tau)
            dd['k'] = dict(hist_target=np.bincount(kall, minlength=KMAX + 1).tolist(), share_target_fits_k_ge1=float(np.mean(kall >= 1)), per_participant_share=kshare,
                           n_participants_k_ge1_in_half=sum(1 for v in kshare.values() if v >= 0.5), n_participants_E_le_tau_and_k=n_ok)
            dd['wording_branch'] = ('estimate wording allowed' if (SR['E_real']['median_le_tau'] and dd['k']['n_participants_k_ge1_in_half'] > len(D_) / 2) else
                                    'upper-bound wording: ' + ('E above tau' if not SR['E_real']['median_le_tau'] else 'no direction passed the rank rule in most participants (layer applies no correction)'))
            if dn == 'R1_raw': dd['GR1'] = dict(all_max_abs_diff=max(r['decoders'][dn]['GR1']['all_gain_abs_diff'] for r in FL['stepR'].values()), A1_max_abs_diff=max((r['decoders'][dn]['GR1']['A1_gain_abs_diff'] or 0) for r in FL['stepR'].values()), A1_key='rows.all_eogreg.64.gain')
            SR['decoders'][dn] = dd
        S['stepR'] = SR
    S['cost'] = {st: dict(participants=len(v), runtime_s_sum=float(sum(r['runtime_s'] for r in v.values())), peak_gpu_gb_max=max((r.get('peak_gpu_gb', 0) for r in v.values()), default=None)) for st, v in FL.items()}
    jdump(root / 'summary.json', S)
    log(json.dumps({k: S[k] for k in ('RESULT_STATUS', 'present', 'P_in_files', 'gates', 'floors', 'G2_null_all_arms_pass', 'gates_all_pass', 'cost')}, indent=1)[:5000])
    log('summary written to', root / 'summary.json')
    if complete or a.test_md:                                              # protocol sec 10: numbers-only table in outputs/ (no interpretation)
        fm = lambda b: 'NA' if not b else f"{b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
        C = RD['64']['cells']; md = [f'# yield_projector summary ()', '', f'- written {S["written"]}; P {S["P_in_files"]}; seeds {seeds}; code {S["code_sha256_in_files"]}',
                                     f'- source: {root}/summary.json; gates_all_pass = {gates_ok}; floors max |mean| = {floors["max_abs"]:.4f}; tau = {tau}',
                                     f'- decision-table row (RHO 0.5, L 64): {S["decision"]["L64"].get("row")} (valid = {S["decision"]["valid"]}); full width: {S["decision"]["Lfull_second_readout"].get("row")}',
                                     f'- {S["decision"]["known_weakness"]}', f'- {S["read_out_note"]}', '', '## Cells, L 64: participant mean yield [95% bootstrap interval], bits', '']
        for ck in sorted(C):
            md.append(f'### {ck}'); md += [f'- {arm}: {fm(b)}' for arm, b in C[ck]['yields'].items()]
            for crit in ('a', 'a_prime', 'd'):
                if crit in C[ck]: md.append(f'- criterion {crit}: reading {C[ck][crit]["reading"]}, A0 {C[ck][crit]["A0_reading"]}, binding {C[ck][crit]["binding"]}, correction applied {C[ck][crit]["correction_applied"]}, credited {C[ck][crit]["credited"]}')
            if 'A3_minus_neural' in C[ck] and C[ck]['A3_minus_neural']: md.append(f'- A3 - neural-only: {fm(C[ck]["A3_minus_neural"])}; A0 - neural-only: {fm(C[ck].get("A0_minus_neural"))}')
            if 'A4_vs_A3' in C[ck]: md.append(f'- |A4 - n| - |A3 - n|: {fm(C[ck]["A4_vs_A3"]["absA4_minus_absA3"])}; A4 closer than A3: {C[ck]["A4_vs_A3"]["A4_closer_than_A3"]}')
            if 'b' in C[ck]: md.append('- criterion b ratios to neural-only: ' + '; '.join(f'{arm} {v["ratio"]:.3f} [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}] {v["reading_ge_0p90"]}' for arm, v in C[ck]['b'].items()))
            if 'c' in C[ck]: md.append(f'- criterion c: {C[ck]["c"]}')
            md.append('')
        if 'stepR' in S:
            md.append('## Step R (descriptive), L 64')
            for dn, dd in S['stepR']['decoders'].items():
                md.append(f'- {dn}: ' + '; '.join(f'{arm} {fm(b)}' for arm, b in dd['64']['yields'].items()) + f'; A3 - A0 {fm(dd["64"]["A3_minus_A0"])}; k hist {dd["k"]["hist_target"]}; {dd["wording_branch"]}')
            md.append(f'- E real median {S["stepR"]["E_real"]["median"]:.4f} against tau {tau}')
        dest = (ROOT / 'outputs' / f'yield_projector_P{"-".join(str(p) for p in S["P_in_files"])}_seeds{"-".join(str(x) for x in seeds)}_{time.strftime("%Y%m%d-%H%M")}.md') if complete else root / 'summary_codepath_test.md'
        dest.write_text('\n'.join(md) + '\n'); log('table written to', dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default=''); ap.add_argument('--summarize', action='store_true'); ap.add_argument('--root', default=str(OUT))
    ap.add_argument('--subjects', default=','.join(str(i) for i in range(1, 24))); ap.add_argument('--seeds', default='0,1'); ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--P', type=int, default=199); ap.add_argument('--drop-attr', default=''); ap.add_argument('--cos', default='0,0.3,0.7'); ap.add_argument('--threads', type=int, default=0); ap.add_argument('--test-md', action='store_true')
    a = ap.parse_args(); a.root = Path(a.root); a.root.mkdir(parents=True, exist_ok=True)
    a.subjects = [int(x) for x in a.subjects.split(',')]; a.seeds = [int(x) for x in a.seeds.split(',')]
    a.drop_attr = [float(x) for x in a.drop_attr.split(',') if x]; a.cos = [float(x) for x in a.cos.split(',') if x]
    assert a.P >= 19 and all(c in COS_IX for c in a.cos) and all(r in (0.25, 0.9) for r in a.drop_attr)       # RHO 0.5 is never cut (protocol sec 14.1)
    if a.threads: torch.set_num_threads(a.threads)
    log(f'yield_projector stage={a.stage or "summarize"} subjects={a.subjects} seeds={a.seeds} P={a.P} drop_attr={a.drop_attr} cos={a.cos} root={a.root} sha={CODE_SHA256[:12]}')
    if a.summarize: return summarize(a)
    if a.stage == 'cache':
        for s in a.subjects: S = load_sub(s); log(f'cache sub {s}: n {S["n"]}, O {S["O"].shape}')
    elif a.stage == 'scopeE': stage_scopeE(a)
    elif a.stage == 'base':
        for s in a.subjects: stage_base(a, s, a.rho)
    elif a.stage == 'arms':
        for s in a.subjects: stage_arms(a, s, a.rho)
    elif a.stage == 'f0null':
        for s in a.subjects: stage_arms(a, s, 0.5, f0null=True)
    elif a.stage == 'ovk':
        for s in a.subjects: stage_ovk(a, s)
    elif a.stage == 'stepR':
        for s in a.subjects: stage_stepR(a, s)
    elif a.stage == 'main': stage_main(a)
    elif a.stage == 'check': stage_check(a)
    else: raise SystemExit('unknown --stage')
    log('done', a.stage)


if __name__ == '__main__':
    with torch.no_grad(): main()
