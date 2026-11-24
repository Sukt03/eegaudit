"""pilot_proximal_A.py: Pilot A (reduced-rank proximal / errors-in-variables removal of the ocular nuisance on planted cells). RUNNER implementation of
outputs/FINAL/PROTOCOL_proximal_A.md (24 Nov 2026). ONE hash-pinned script, stages:
  stamp | s0 | p0 | reach | p1 | p6 | test8 | guards | perm | grid | swap | desc | summary     (+ --smoke: 2 participants, a handful of cells, writes to results/pilot_proximal_cmi/A_smoke)
Run: srun -n1 -c3 -t <min> python -I -u experiments/ext/pilot_proximal_A.py --stage <stage>
Constants are read from the CONSTANTS_JSON block of the protocol (hashed in the PREREG). Numerics: torch float64 on the GPU; named CPU exceptions as in td_cells.py (numpy rng draws,
numpy SVD in td_cells.pb, MNE inside the cached ocular arrays, JSON I/O, scalar bookkeeping).
"""
import hashlib as _hl
SCRIPT_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, re, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "3")
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import td_cells as TC          # noqa: E402  (imports td_screen, d2_pe_tasks, pe_eogreg, pid_regen)
import td_screen as TS         # noqa: E402
import pid_regen as PR         # noqa: E402
import d2_pe_tasks as D2       # noqa: E402
import pe_eogreg as PE         # noqa: E402

SMOKE = '--smoke' in sys.argv
DEV = 'cuda'; T64 = dict(dtype=torch.float64, device=DEV); t0 = time.time()
log = lambda *m: print(f'[{time.time() - t0:8.1f}s]', *m, flush=True)
G = TC.G
PROTOCOL = ROOT / 'outputs/FINAL/PROTOCOL_proximal_A.md'
RES = ROOT / ('results/pilot_proximal_cmi/A_smoke' if SMOKE else 'results/pilot_proximal_cmi/A')
HOLD = 'cluster'; HOLD_JOB = 0
STEP = dict(stamp=1, s0=2, p0=3, reach=4, p1=5, p6=6, test8=7, guards=8, perm=9, grid=10, swap=11, desc=12, summary=13)


def _proc_start():
    try:
        tick = os.sysconf('SC_CLK_TCK'); st = int(open('/proc/self/stat').read().rsplit(')', 1)[1].split()[19]); bt = [int(l.split()[1]) for l in open('/proc/stat') if l.startswith('btime')][0]
        return bt + st / tick
    except Exception: return time.time() - 15.0
T_START = min(_proc_start(), time.time())


def load_constants():
    m = re.search(r'```json CONSTANTS_JSON\n(.*?)\n```', PROTOCOL.read_text(), re.S); return json.loads(m.group(1))
C = load_constants()
CONST_SHA = _hl.sha256(json.dumps(C, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
ALLP = C['participants']; SMOKE_PARTS = [2, 5]
PL = SMOKE_PARTS if SMOKE else ALLP                       # participants for read-out stages
SEEDS2 = [0] if SMOKE else C['seeds_primary']; SEED0 = C['seed_grid']
FS3 = [0.5] if SMOKE else [0.25, 0.5, 0.75]
RHO2 = (0.5, 0.25)                                         # RHO .5 first
IA = list(range(23)) + [69]; IB = list(range(23, 69))      # block A (proxy, 24 columns) / block B (instrument, 46 columns)
fk = lambda f: f'f{f:g}'
LKEYS = ('64', 'full')
CALLS = 0
ZV = [0]                                                    # count of zero-variance proxy columns met in a training set (set to z = 0)


def jdump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f'.tmp{os.getpid()}'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def jload(path): return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- ledger, stamp, budget
def ledger_sum_h():
    p = RES / 'ledger.jsonl'
    if not p.exists(): return 0.0
    return sum(json.loads(l)['dur_s'] for l in p.read_text().splitlines() if l.strip() and 'dur_s' in json.loads(l)) / 3600.0


def ledger_line(obj):
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / 'ledger.jsonl', 'a') as fh: fh.write(json.dumps(obj) + '\n')


def ledger_end(stage, calls, error=None):
    t_end = time.time(); dur = t_end - T_START; prev = ledger_sum_h()
    ledger_line(dict(step=STEP[stage], stage=stage, hold=HOLD, t_start_unix=T_START, t_end_unix=t_end, dur_s=dur, calls=int(calls), s_per_call=dur / max(calls, 1), script_sha256=SCRIPT_SHA256[:12],
                     sum_dur_h_so_far=prev + dur / 3600.0, smoke=SMOKE, **({'error': error} if error else {})))


def mod_sha(name): return _hl.sha256(Path(ROOT / f'experiments/ext/{name}.py').read_bytes()).hexdigest()


def verify_stamp():
    if SMOKE: return {}
    p = RES / 'PREREG.json'
    if not p.exists(): raise SystemExit('PREREG.json missing: run --stage stamp first')
    st = jload(p)
    if st['script_sha256'] != SCRIPT_SHA256: raise SystemExit('ABORT: script sha256 differs from the stamp (protocol sec 11: new hash and new PREREG required)')
    if st['protocol_sha256'] != _hl.sha256(PROTOCOL.read_bytes()).hexdigest() or st['constants_sha256'] != CONST_SHA: raise SystemExit('ABORT: protocol or constants differ from the stamp')
    for k, v in st['imported_module_sha256'].items():
        if mod_sha(k) != v: raise SystemExit(f'ABORT: imported module {k} differs from the stamp')
    return st


def plan():
    p = RES / 'plan.json'; return jload(p) if p.exists() else None


class Unreached(Exception): pass


STEP_CALLS = dict(p0=[(552, 'A')], reach=[(1104, 'A_orc')], p1=[(1104, 'A')], p6=[(920, 'A'), (23, 'plain')], test8=[(120, 'plain'), (960, 'A'), (192, 'A4')], guards=[(1104, 'G1'), (1104, 'G2'), (1104, 'G3')],
                  perm=[(552, 'A')], grid=[(276, 'A'), (276, 'A4')], swap=[(552, 'A')], desc=[(552, 'A'), (552, 'A4'), (1104, 'A')])


def project_h(stage, dropped=()):
    pl = plan(); rates = pl['rates']; items = list(STEP_CALLS[stage])
    if stage == 'perm' and 'perm_seed1' in dropped: items = [(c // 2, k) for c, k in items]
    if stage == 'grid' and 'grid_readouts' in dropped: items = []
    if stage in ('swap',) and 'swap' in dropped: items = []
    if stage == 'desc' and 'desc' in dropped: items = []
    return sum(c * 1.25 * rates[k] for c, k in items) / 3600.0


def budget_guard(stage):
    """Ledger rule (sec 10): do not start a step if summed + calls x 1.25 x measured s/call / 3600 > cap. Drop order applied through plan.json."""
    if SMOKE: return
    pl = plan()
    if pl is None: raise SystemExit('plan.json missing (run s0)')
    dropped = pl.get('dropped', [])
    if (stage == 'swap' and 'swap' in dropped) or (stage == 'desc' and 'desc' in dropped): raise Unreached(f'{stage} dropped by the drop order (budget)')
    proj = project_h(stage, dropped); summed = ledger_sum_h()
    log(f'budget guard {stage}: summed {summed:.4f} h + projected {proj:.4f} h vs cap {C["ledger_cap_h"]}')
    if summed + proj > C['ledger_cap_h']: raise Unreached(f'{stage}: summed {summed:.4f} + projected {proj:.4f} > cap {C["ledger_cap_h"]} h')


# ---------------------------------------------------------------- generation
def gen_D(s, seed, fam='R0'):
    P = TC.pb(s); th = TC.fam_theta(fam)
    rng = np.random.default_rng(910000 + 100 * s + seed) if fam == 'R0' else np.random.default_rng([200 + TC.FAMCODE[fam], s, seed])
    return P, TC.draw_data(P, th, rng)


def mk_cell(D, f, rho):
    a, b = PR.amp(f); Fn, Fo, Fm, Om = TC.assemble(D, a, b, rho, TC.KAPPA0)
    return dict(Fn=Fn, Fm=Fm, Om=Om, y=D['y'], a=a, b=b, rho=rho, f=f, D=D)


def s_lat_of(D, rho): return rho * D['y'] + math.sqrt(1 - rho ** 2) * D['es'][0]
def blocks(Om): return G(Om[:, IA]), G(Om[:, IB])


# ---------------------------------------------------------------- estimators
EYE = lambda p: torch.eye(p, **T64)


def _zs(P, m, cnt):
    mu = (m @ P) / cnt[:, None]; Pc = P[None] - mu[:, None]; sd = ((Pc ** 2 * m[:, :, None]).sum(1) / cnt[:, None]).sqrt(); bad = sd <= 1e-8
    ZV[0] += int(bad.sum()); inv = torch.where(bad, torch.zeros_like(sd), 1.0 / torch.where(bad, torch.ones_like(sd), sd)); return Pc * inv[:, None]


def _isq(S):
    e, V = torch.linalg.eigh(S); return (V * e.clamp_min(1e-12).rsqrt()[:, None, :]) @ V.transpose(1, 2)


def cca_parts(PA, PB, yG, M, r=3, lam=0.1, eps=1e-8):
    """Steps 1-3 of estimator A (sec 3): z-score with training rows, residualise [1, y_j] (training LS, ridge eps), centre, ridge partial CCA on training rows. All rows get the indices."""
    m = M.double(); cnt = m.sum(1); ZA = _zs(PA, m, cnt); ZB = _zs(PB, m, cnt)
    Dm = torch.stack([torch.ones_like(yG), yG], 1); Dk = Dm[None] * m[:, :, None]; GD = Dk.transpose(1, 2) @ Dk + eps * EYE(2)
    def resid(Z):
        Cf = torch.linalg.solve(GD, Dk.transpose(1, 2) @ Z); R = Z - Dm[None] @ Cf; return R - ((R * m[:, :, None]).sum(1) / cnt[:, None])[:, None]
    RA, RB = resid(ZA), resid(ZB); RAm = RA * m[:, :, None]
    Saa = RAm.transpose(1, 2) @ RA / cnt[:, None, None] + lam * EYE(PA.shape[1]); Sbb = (RB * m[:, :, None]).transpose(1, 2) @ RB / cnt[:, None, None] + lam * EYE(PB.shape[1])
    Sab = RAm.transpose(1, 2) @ RB / cnt[:, None, None]; IAm, IBm = _isq(Saa), _isq(Sbb)
    U, S, Vh = torch.linalg.svd(IAm @ Sab @ IBm, full_matrices=False)
    wa = IAm @ U[:, :, :r]; wb = IBm @ Vh[:, :r, :].transpose(1, 2)
    return dict(ZA=ZA, ZB=ZB, wa=wa, wb=wb, zA=ZA @ wa, zB=ZB @ wb, S=S)


def a_parts(F, PA, PB, yG, M, r=3, lam=0.1, eps=1e-8):
    """Estimator A (sec 3 steps 1-5): returns the removal Rm (K, n, D) and the internals."""
    cp = cca_parts(PA, PB, yG, M, r, lam, eps); zA, zB = cp['zA'], cp['zB']; K, n = M.shape; m = M.double()[:, :, None]
    one = torch.ones(K, n, 1, **T64); yk = yG[None, :, None].expand(K, n, 1)
    X1 = torch.cat([one, zB, yk], 2); X1m = X1 * m; Delta = torch.linalg.solve(X1m.transpose(1, 2) @ X1m + eps * EYE(r + 2), X1m.transpose(1, 2) @ zA)
    zAh = X1 @ Delta; X2 = torch.cat([one, yk, zAh], 2); X2m = X2 * m; G2 = X2m.transpose(1, 2) @ X2m
    Gamma = torch.linalg.solve(G2 + eps * EYE(r + 2), X2m.transpose(1, 2) @ F)
    Gz = Gamma[:, 2:2 + r]; DB = Delta[:, 1:1 + r]; Rm = (zB @ DB) @ Gz
    return dict(Rm=Rm, zB=zB, zA=zA, G2=G2, Delta=Delta, Gz=Gz, DB=DB, cp=cp)


def a_transform(PA, PB, ycol, r=3, lam=0.1):
    def t(F, M): return F[None] - a_parts(F, PA, PB, ycol, M, r, lam)['Rm']
    return t


def g1_transform(Oc, ycol):
    """G1 rank-3 reduced-rank removal (sec 3)."""
    X = torch.cat([torch.ones(len(Oc), 1, **T64), Oc, ycol[:, None]], 1); p = Oc.shape[1]
    def t(F, M):
        m = M.double()[:, :, None]; Xm = X[None] * m
        Bc = torch.linalg.solve(Xm.transpose(1, 2) @ Xm + 1e-8 * EYE(X.shape[1]), Xm.transpose(1, 2) @ (F[None] * m)); Bo = Bc[:, 1:1 + p]
        Q = (Oc[None] * m).transpose(1, 2) @ Oc[None] + 1e-8 * EYE(p); Lc = torch.linalg.cholesky(Q); H = Lc.transpose(1, 2) @ Bo
        Vh = torch.linalg.svd(H, full_matrices=False)[2]; V3 = Vh[:, :3].transpose(1, 2)
        return F[None] - ((Oc[None] @ Bo) @ V3) @ V3.transpose(1, 2)
    return t


def g3_transform(Oc, ycol, ridge=0.1):
    """G3 ridge A4: penalty 1e-8 on [1] and y_j, ridge x mean(diag(Xm'Xm)[1:71]) on the 70 Oc entries."""
    X = torch.cat([torch.ones(len(Oc), 1, **T64), Oc, ycol[:, None]], 1); p = Oc.shape[1]
    def t(F, M):
        m = M.double()[:, :, None]; Xm = X[None] * m; XtX = Xm.transpose(1, 2) @ Xm; K = XtX.shape[0]
        pen = torch.full((K, X.shape[1]), 1e-8, **T64); pen[:, 1:1 + p] = ridge * torch.diagonal(XtX, dim1=1, dim2=2)[:, 1:1 + p].mean(1, keepdim=True)
        Bc = torch.linalg.solve(XtX + torch.diag_embed(pen), Xm.transpose(1, 2) @ (F[None] * m)); return F[None] - Oc[None] @ Bc[:, 1:1 + p]
    return t


def yield_cols(Fm, y, make_t):
    """td_cells.a4_yield scoring for a per-column transform: 4 calls D2.pe_gain(Fm, Ys[:, j:j+1], (64, 'full'), transform=t_j); yield = col 0 - mean(cols 1-3)."""
    global CALLS
    Ys = TC.ycols(y); g = {'64': [], 'full': []}
    for j in range(4):
        sc = D2.pe_gain(Fm, Ys[:, j:j + 1], TC.LS, transform=make_t(G(Ys[:, j])))
        for L in TC.LS: g[str(L)].append(float(sc[L][:, 0].mean()))
        torch.cuda.empty_cache()
    CALLS += 4
    return {L: g[L][0] - float(np.mean(g[L][1:])) for L in g}


def plain_yield(F, y):
    global CALLS
    CALLS += 1; r = TC.yld(TC.readout(F, y)); torch.cuda.empty_cache(); return r


def a4_est(Fm, y, Om):
    global CALLS
    CALLS += 4; return TC.a4_yield(Fm, y, Om)


def arm_yield(arm, cell, PA=None, PB=None, perm=None):
    """Yield of one deployable / guard arm on a cell dict (keys Fm, y, Om, D, rho)."""
    Fm, y, Om = cell['Fm'], cell['y'], cell['Om']
    if arm in ('A', 'A_r1', 'A_swap', 'A_perm'):
        if PA is None: PA, PB = blocks(Om)
        if arm == 'A_perm': PB = PB[torch.as_tensor(perm, device=DEV)]
        if arm == 'A_swap': return yield_cols(Fm, y, lambda yc: a_transform(PB, PA, yc))
        return yield_cols(Fm, y, lambda yc: a_transform(PA, PB, yc, r=1 if arm == 'A_r1' else C['r'], lam=C['lam_cca']))
    if arm == 'A_orc': S = G(s_lat_of(cell['D'], cell['rho']))[:, None]; return yield_cols(Fm, y, lambda yc: TC.a4_transform(S, yc))
    if arm == 'A4': return a4_est(Fm, y, Om)
    Oc = G(Om)
    if arm == 'G1': return yield_cols(Fm, y, lambda yc: g1_transform(Oc, yc))
    if arm == 'G2': Ob = G(Om[:, 23:69]); return yield_cols(Fm, y, lambda yc: TC.a4_transform(Ob, yc))
    if arm == 'G3': return yield_cols(Fm, y, lambda yc: g3_transform(Oc, yc, C['g3_ridge']))
    raise ValueError(arm)


# ---------------------------------------------------------------- flags
_MASK = {}
def masks(n):
    if n not in _MASK:
        M, tests, trains = D2.nested_masks(n); _MASK[n] = (M, torch.as_tensor(tests[0], device=DEV))
    return _MASK[n]


def e_flag(y, Om):
    e = float(TS.e_scope(y, Om)); return e, bool(e > C['e_thr'])


def s1_stat(PA, PB, yG, fo):
    """Held-out pooled squared correlation of the first canonical pair (10 contiguous outer folds, CCA refitted on training folds, [1, y] partialled with training coefficients)."""
    n = len(yG); Mo = torch.as_tensor(np.stack([fo != b for b in range(10)]), device=DEV); cp = cca_parts(PA, PB, yG, Mo, r=1, lam=C['lam_cca'])
    zz = torch.cat([cp['zA'][:, :, :1], cp['zB'][:, :, :1]], 2); m = Mo.double()[:, :, None]
    Dm = torch.stack([torch.ones_like(yG), yG], 1); Dk = Dm[None] * m
    Cf = torch.linalg.solve(Dk.transpose(1, 2) @ Dk + 1e-8 * EYE(2), Dk.transpose(1, 2) @ zz); R = zz - Dm[None] @ Cf
    rows = torch.cat([R[b][torch.as_tensor(fo == b, device=DEV)] for b in range(10)], 0); rows = rows - rows.mean(0)
    return float((rows[:, 0] * rows[:, 1]).sum() ** 2 / ((rows[:, 0] ** 2).sum() * (rows[:, 1] ** 2).sum()))


def s1_test(PA, PB, y, s, seed, rho):
    n = len(y); fo = D2.folds(n); yG = G(y); obs = s1_stat(PA, PB, yG, fo); rng = np.random.default_rng([991, s, seed, int(1000 * rho)])
    nulls = sorted(s1_stat(PA, PB[torch.as_tensor(rng.permutation(n), device=DEV)], yG, fo) for _ in range(C['s1_nshuffle']))
    thr = nulls[17]                                        # 18th smallest of 19 = 2nd largest (see ATTEMPTS: reading of "18th largest ... fires with probability about 0.95")
    return dict(obs=obs, thr=thr, null_max=nulls[-1], flag=bool(obs <= thr), flag_max=bool(obs <= nulls[-1]))


def s23(F, PA, PB, y):
    n = len(y); M, te0 = masks(n); M0 = M[:1]; yG = G(y)
    pa = a_parts(F, PA, PB, yG, M0, C['r'], C['lam_cca']); ps = a_parts(F, PB, PA, yG, M0, C['r'], C['lam_cca'])
    a = pa['Rm'][0][te0].reshape(-1); b = ps['Rm'][0][te0].reshape(-1); den = float(a.norm() * b.norm()); cos = float((a * b).sum()) / den if den > 0 else float('nan')
    cond = float(torch.linalg.cond(pa['G2'][0]))
    return dict(s2_cos=cos, s2_flag=bool(not (cos >= C['s2_tau'])), s3_cond=cond, s3_flag=bool(cond > C['s3_cond']))


def cell_flags(cell, PA, PB, s1r):
    e, ef = e_flag(cell['y'], cell['Om']); r23 = s23(cell['Fm'], PA, PB, cell['y'])
    return dict(E=e, eflag=ef, s1=s1r['obs'], s1_thr=s1r['thr'], s1_flag=s1r['flag'], s1_flag_max=s1r['flag_max'], **r23, sflag=bool(s1r['flag'] or r23['s2_flag'] or r23['s3_flag']))


# ---------------------------------------------------------------- stored inputs (truth, A4 comparator)
def stored_any(rho, s):
    S = TS.stored(rho, s)
    if S is not None: return S
    p = ROOT / f'results/yield_projector/planted_v2_rho{rho:g}_add11/sub-{s:03d}.json'; return jload(p) if p.exists() else None


def t9r(rho, s): return jload(ROOT / f'results/yield_projector/runner_T9R/arms_rho{rho:g}/sub-{s:03d}.json')


_TR = {}
def stored_inputs():
    if not _TR:
        _TR['truth'] = jload(RES / 'truth.json'); _TR['a4'] = jload(RES / 'a4_stored.json')
    return _TR['truth'], _TR['a4']


def truthN(s, seed, f, L): return stored_inputs()[0]['N'][str(s)][str(seed)][fk(f)][L]
def truthNull(s, seed, L): return stored_inputs()[0]['null'][str(s)][str(seed)][L]
def a4s(rho, s, seed, f, L): return stored_inputs()[1][f'{rho:g}'][str(s)][str(seed)][fk(f)][L]


def build_truth_a4():
    truth = dict(N={}, null={}); a4 = {f'{r:g}': {} for r in (0.25, 0.5, 0.9)}
    for s in ALLP:
        S = TS.stored(0.5, s); truth['N'][str(s)] = {}; truth['null'][str(s)] = {}
        for sd in (0, 1):
            truth['null'][str(s)][str(sd)] = {L: float(TS.scell(S, sd, 'null')['mix_all'][L]) for L in LKEYS}
            truth['N'][str(s)][str(sd)] = {fk(f): {L: float(TS.scell(S, sd, f)['neural_only'][L]) for L in LKEYS} for f in (0.25, 0.5, 0.75)}
        for r in (0.25, 0.5, 0.9):
            T = t9r(r, s); d = a4[f'{r:g}'].setdefault(str(s), {})
            for c in T['cells']:
                if c['cell'] in ('f0.25', 'f0.5', 'f0.75') and c['seed'] in (0, 1): d.setdefault(str(c['seed']), {})[c['cell']] = {L: float(c['arms']['A4']['yield'][L]) for L in LKEYS}
    return truth, a4


# ---------------------------------------------------------------- statistics
def make_ix(n, seed=None, B=None):
    g = torch.Generator(device=DEV); g.manual_seed(C['boot_seed'] if seed is None else seed); return torch.randint(0, n, (B or C['boot_B'], n), generator=g, device=DEV)


def mae6(E): return float(np.abs(E.mean(0)).mean())
def pl_mae(E): return float(np.abs(E).mean())


def boot_ratio(EX, EY, IX):
    """sec 7: MAE_b(X) = mean over cells |mean_{i in IX[b]} E_X[i, c]|, ratio_b = MAE_b(X)/MAE_b(Y); returns point ratio, UB97.5, UB98.75 and the lower end."""
    X = torch.as_tensor(EX, **T64); Y = torch.as_tensor(EY, **T64)
    mx = X[IX].mean(1).abs().mean(1); my = Y[IX].mean(1).abs().mean(1); r = mx / my
    q = torch.quantile(r, torch.tensor([0.025, C['ub_q'], 0.9875], **T64)).cpu().tolist()
    return dict(point=mae6(EX) / mae6(EY), lb2_5=q[0], ub97_5=q[1], ub98_75=q[2])


def power_table(EA4):
    """Descriptive: A4's demeaned participant scatter x k with random participant sign flips, bias b per cell, 100 repetitions; pass = point ratio <= 0.75 and UB97.5 < 1."""
    nP = EA4.shape[0]; IX = make_ix(nP); gen = torch.Generator(device=DEV); gen.manual_seed(C['boot_seed'] + 1); Ed = EA4 - EA4.mean(0); out = {}
    for k in (1.0, 1.5, 2.0):
        for b in (0.0, 0.003, 0.005, 0.0065, 0.0078):
            ok = 0
            for _ in range(100):
                sg = torch.where(torch.rand(nP, 1, generator=gen, device=DEV) < 0.5, -1.0, 1.0).double().cpu().numpy()
                Es = b + k * sg * Ed; br = boot_ratio(Es, EA4, IX); ok += int(br['point'] <= 0.75 and br['ub97_5'] < 1)
            out[f'k{k}_b{b}'] = ok / 100.0
    return out


# ---------------------------------------------------------------- per-participant drivers
def run_stage(sdir, parts, fn):
    d = RES / sdir; d.mkdir(parents=True, exist_ok=True)
    for s in parts:
        p = d / f'sub-{s:03d}.json'
        if p.exists(): continue
        t1 = time.time(); c1 = CALLS; cells = fn(s); jdump(p, dict(participant=s, script_sha256=SCRIPT_SHA256[:12], NOT_CITABLE='until a blind bug review', cells=cells))
        log(f'{sdir} sub {s}: {len(cells)} cells, {CALLS - c1} calls, {time.time() - t1:.1f}s')


def load_recs(*sdirs):
    out = {}
    for sd in sdirs:
        for p in sorted((RES / sd).glob('sub-*.json')):
            d = jload(p); out.setdefault(d['participant'], []).extend(d['cells'])
    return out


def find(cells, arm, **kw):
    r = [c for c in cells if arm in c.get('est', {}) and all(c.get(k) == v for k, v in kw.items())]
    if len(r) != 1: raise KeyError((arm, kw, len(r)))
    return r[0]


def err_matrix(recs, arm, cells, L, seeds, itt=False, truth='N'):
    """Participant-level error (mean over seeds of estimate - truth), shape (participants, cells); cells = [(rho, f)]; itt: flagged (flag E) participant-seeds carry A4's stored value."""
    parts = sorted(recs); E = np.zeros((len(parts), len(cells)))
    for i, s in enumerate(parts):
        for j, (rho, f) in enumerate(cells):
            v = []
            for sd in seeds:
                c = find(recs[s], arm, rho=rho, seed=sd, f=f); e = c['est'][arm][L]
                if itt and c['eflag']: e = a4s(rho, s, sd, f, L)
                v.append(e - truthN(s, sd, f, L))
            E[i, j] = np.mean(v)
    return E


def a4_matrix(parts, cells, L, seeds):
    return np.array([[np.mean([a4s(rho, s, sd, f, L) - truthN(s, sd, f, L) for sd in seeds]) for (rho, f) in cells] for s in parts])


SIX = [(r, f) for r in RHO2 for f in (0.25, 0.5, 0.75)]
def six_cells(): return [(r, f) for r in RHO2 for f in FS3]


# ---------------------------------------------------------------- evaluators (also used by summary)
def ev_p0():
    recs = load_recs('p0'); res = {}
    for kind, rho, thr in (('ocular', 0.5, C['p0_ocular']), ('ocular', 0.25, C['p0_ocular']), ('null', 0.5, C['p0_null'])):
        row = {}
        for L in LKEYS:
            e = np.array([np.mean([find(recs[s], 'A', kind=kind, rho=rho, seed=sd)['est']['A'][L] - truthNull(s, sd, L) for sd in SEEDS2]) for s in sorted(recs)]); row[L] = dict(cohort_mean=float(e.mean()), sd=float(e.std(ddof=1)) if len(e) > 1 else None, n=len(e))
        row['pass'] = bool(abs(row['64']['cohort_mean']) <= thr); row['thr'] = thr; res[f'{kind}_rho{rho:g}' if kind == 'ocular' else 'null'] = row
    res['pass'] = bool(all(v['pass'] for v in res.values())); return res


def ev_reach():
    recs = load_recs('reach'); cells = six_cells(); res = {}
    for L in LKEYS:
        E = err_matrix(recs, 'A_orc', cells, L, SEEDS2); res[L] = dict(mae6=mae6(E), cohort_err=[float(x) for x in E.mean(0)], cells=[list(c) for c in cells])
    res['pass'] = bool(res['64']['mae6'] <= C['reach']); res['bar'] = C['reach']; return res


def ev_arm(recs, arm, cells, L, itt):
    return err_matrix(recs, arm, cells, L, SEEDS2, itt=itt)


def ev_p1(rho_only=None):
    """P1 core (a-d), secondary, with and without the ITT replacement. rho_only=0.5: clauses c, d only (stage-1 first pass)."""
    recs = load_recs('p1_rho0.5', 'p1_rho0.25') if rho_only is None else load_recs('p1_rho0.5'); parts = sorted(recs); IX = make_ix(len(parts)); out = {}
    for itt in (True, False):
        k = 'itt' if itt else 'raw'; o = {}; c5 = [(0.5, f) for f in FS3]
        E5 = ev_arm(recs, 'A', c5, '64', itt); o['mae_rho5'] = mae6(E5); o['cohort_err_rho5'] = [float(x) for x in E5.mean(0)]
        o['clause_c'] = bool(o['mae_rho5'] <= C['bar_rho5']); o['clause_d_rho5'] = bool(np.abs(E5.mean(0)).max() <= C['cell_cap'])
        if rho_only is None:
            cs = six_cells(); E6 = ev_arm(recs, 'A', cs, '64', itt); A4E = a4_matrix(parts, cs, '64', SEEDS2); br = boot_ratio(E6, A4E, IX)
            o.update(mae6=mae6(E6), mae6_A4=mae6(A4E), ratio=br, cohort_err=[float(x) for x in E6.mean(0)], cells=[list(c) for c in cs], plmae=pl_mae(E6), plmae_A4=pl_mae(A4E),
                     clause_a=bool(mae6(E6) <= C['bar_p1']), clause_b=bool(br['ub97_5'] < 1), clause_d=bool(np.abs(E6.mean(0)).max() <= C['cell_cap']),
                     mae_rho25=mae6(E6[:, 3:]) if len(cs) == 6 else None)
            o['core_pass'] = bool(o['clause_a'] and o['clause_b'] and o['clause_c'] and o['clause_d'])
            o['secondary_plmae'] = bool(o['plmae'] <= C['pl_limit'])
            Ef = ev_arm(recs, 'A', cs, 'full', itt); A4f = a4_matrix(parts, cs, 'full', SEEDS2); brf = boot_ratio(Ef, A4f, IX)
            o['full'] = dict(mae6=mae6(Ef), mae6_A4=mae6(A4f), ratio=brf, pass_=bool(mae6(Ef) <= C['full_mae'] and brf['ub97_5'] < C['full_ub']))
            o['full_pass'] = o['full']['pass_']
            if len(parts) > 2 and 2 in parts:
                keep = [i for i, s in enumerate(parts) if s != 2]; o['sens_without_p2'] = dict(mae6=mae6(E6[keep]), mae6_A4=mae6(A4E[keep]))
        else: o['core_pass_rho5'] = bool(o['clause_c'] and o['clause_d_rho5'])
        out[k] = o
    return out


def ev_guards():
    recs = load_recs('guards'); recA = load_recs('p1_rho0.5', 'p1_rho0.25'); cs = six_cells(); parts = sorted(recs); IX = make_ix(len(parts)); res = {}
    # recs of the guards carry est G1..G3; A comes from p1 (same participants / cells); ITT as in P1
    EA = err_matrix({s: recA[s] for s in parts}, 'A', cs, '64', SEEDS2, itt=True); res['A_mae6'] = mae6(EA); arms = {}
    for g_ in ('G1', 'G2', 'G3'):
        E = err_matrix(recs, g_, cs, '64', SEEDS2, itt=True); arms[g_] = E; res[g_] = dict(mae6=mae6(E), cohort_err=[float(x) for x in E.mean(0)])
    res['ratio_A_G1'] = boot_ratio(EA, arms['G1'], IX); mn = min(('G1', 'G2', 'G3'), key=lambda k: res[k]['mae6']); res['argmin'] = mn
    res['ratio_A_min'] = boot_ratio(EA, arms[mn], IX); res['p5_a'] = bool(res['ratio_A_G1']['ub97_5'] < 1)
    res['p5_b'] = bool(res['A_mae6'] <= C['p5_factor'] * res[mn]['mae6'] and res['ratio_A_min']['ub97_5'] < 1); res['p5_pass'] = bool(res['p5_a'] and res['p5_b']); return res


def ev_perm():
    rp = load_recs('perm'); recA = load_recs('p1_rho0.5', 'p1_rho0.25'); parts = sorted(rp); c5 = [(0.5, f) for f in FS3]; seeds = SEEDS2 if not (plan() and 'perm_seed1' in plan().get('dropped', [])) else [0]
    Ep = err_matrix(rp, 'A_perm', c5, '64', seeds); EA = err_matrix({s: recA[s] for s in parts}, 'A', c5, '64', seeds, itt=True)
    fire = [bool(find(rp[s], 'A_perm', rho=0.5, seed=sd, f=FS3[0])['s1_flag']) for s in parts for sd in seeds]
    res = dict(mae_perm=mae6(Ep), mae_A=mae6(EA), ratio=mae6(Ep) / mae6(EA), s1_fire=float(np.mean(fire)), n_ps=len(fire), s1_fire_max_reading=float(np.mean([bool(find(rp[s], 'A_perm', rho=0.5, seed=sd, f=FS3[0])['s1_flag_max']) for s in parts for sd in seeds])))
    res['pass'] = bool(res['mae_perm'] >= C['p2_factor'] * res['mae_A'] and res['s1_fire'] >= C['p2_s1']); return res


def ev_swap():
    rs = load_recs('swap'); recA = load_recs('p1_rho0.5', 'p1_rho0.25'); parts = sorted(rs); res = dict(cells=[])
    for f in FS3:
        a = np.array([np.mean([find(recA[s], 'A', rho=0.5, seed=sd, f=f)['est']['A']['64'] for sd in SEEDS2]) for s in parts]); b = np.array([np.mean([find(rs[s], 'A_swap', rho=0.5, seed=sd, f=f)['est']['A_swap']['64'] for sd in SEEDS2]) for s in parts])
        res['cells'].append(dict(f=f, diff=float(abs(a.mean() - b.mean())), mean_A=float(a.mean()), mean_swap=float(b.mean())))
    res['pass'] = bool(all(c['diff'] <= C['p3_tol'] for c in res['cells'])); return res


def ev_p6():
    """P6 (broken cell a) and P4b (credit / failure cells; false-positive cap) and cell (b)."""
    rs = load_recs('p6'); parts = sorted(rs); recA = load_recs('p1_rho0.5', 'p1_rho0.25'); res = dict(a=[], b=[])
    tol = {float(k): v for k, v in C['p6_tol'].items()}
    for rho in RHO2:
        ref = np.array([find(rs[s], 'A', kind='a', rho=rho, c=0.0)['est']['A']['64'] - rs[s][0]['truth_eta']['64'] for s in parts]); ref_ce = float(ref.mean())
        for c in (0.05, 0.1, 0.2):
            cells = [find(rs[s], 'A', kind='a', rho=rho, c=c) for s in parts]; e = np.array([x['est']['A']['64'] - rs[s][0]['truth_eta']['64'] for x, s in zip(cells, parts)]); ce = float(e.mean())
            sfire = float(np.mean([x['sflag'] for x in cells])); efire = float(np.mean([x['eflag'] for x in cells])); shift = abs(ce - ref_ce)
            res['a'].append(dict(rho=rho, c=c, cohort_err=ce, ref_cohort_err=ref_ce, shift=shift, tol=tol[c], within_tol=bool(shift <= tol[c]), s_fire=sfire, e_fire=efire, pass_=bool(shift <= tol[c] or sfire >= C['p6_flag_min']),
                                 credit=bool(efire <= C['credit_e_max']), failure=bool(shift > tol[c])))
        refc = [find(rs[s], 'A', kind='a', rho=rho, c=0.0) for s in parts]
        res['a_ref_%g' % rho] = dict(s_fire=float(np.mean([x['sflag'] for x in refc])), e_fire=float(np.mean([x['eflag'] for x in refc])), s1_fire=float(np.mean([x['s1_flag'] for x in refc])), cohort_err=ref_ce)
        std = np.array([np.mean([find(recA[s], 'A', rho=rho, seed=0, f=0.5)['est']['A']['64']]) - truthN(s, 0, 0.5, '64') for s in parts]); bc = [find(rs[s], 'A', kind='b', rho=rho) for s in parts]
        eb = np.array([x['est']['A']['64'] - truthN(s, 0, 0.5, '64') for x, s in zip(bc, parts)]); shift = abs(float(eb.mean()) - float(std.mean())); sfire = float(np.mean([x['sflag'] for x in bc])); efire = float(np.mean([x['eflag'] for x in bc]))
        res['b'].append(dict(rho=rho, cohort_err=float(eb.mean()), std_cohort_err=float(std.mean()), shift=shift, tol=C['p6_tol_b'], within_tol=bool(shift <= C['p6_tol_b']), s_fire=sfire, e_fire=efire, pass_=bool(shift <= C['p6_tol_b'] or sfire >= C['p6_flag_min']),
                             credit=bool(efire <= C['credit_e_max']), failure=bool(shift > C['p6_tol_b'])))
    res['c05_pass'] = bool(all(x['pass_'] for x in res['a'] if x['c'] == 0.05)); res['n_pass_a'] = int(sum(x['pass_'] for x in res['a'])); res['dead_p6'] = bool((not res['c05_pass']) or res['n_pass_a'] < C['p6_cells_min'])
    res['refine_p6'] = bool(any((not x['pass_']) for x in res['a'] if x['c'] >= 0.1) or any((not x['pass_']) for x in res['b']))
    allc = res['a'] + res['b']; fcred = [x for x in allc if x['credit'] and x['failure']]
    p1c = [c for s in sorted(recA) for c in recA[s] if 'sflag' in c]; fp_p1 = float(np.mean([c['sflag'] for c in p1c])) if p1c else None
    fp_ref = [res['a_ref_%g' % rho]['s_fire'] for rho in RHO2]; fpcap = bool(fp_p1 is not None and fp_p1 <= C['s_fp_max'] and max(fp_ref) <= C['s_fp_max'])
    res['p4b'] = dict(n_failure_credit=len(fcred), s_fire_min=min([x['s_fire'] for x in fcred]) if fcred else None, credit_failure_clause=('N/A' if not fcred else bool(all(x['s_fire'] >= C['credit_s_min'] for x in fcred))),
                      fp_cap_p1_cells=fp_p1, fp_cap_ref_cells=fp_ref, fp_cap_pass=fpcap, n_p1_cells=len(p1c))
    res['p4b']['clause'] = bool(fpcap and res['p4b']['credit_failure_clause'] in (True, 'N/A'))
    return res


def ev_test8():
    rg = load_recs('test8'); regen = jload(RES / 'test8_regen.json'); parts = sorted(rg) if rg else []; res = dict(families={}, regen=regen['summary'])
    if not parts: res['reached'] = False; return res
    fams = [f for f in ('R2', 'R3', 'R4') if regen['summary'].get(f, {}).get('pass')]
    IX = make_ix(len(parts), C['test8_boot_seed'], C['boot_B']); EAall, EBall = [], []
    for fam in fams:
        seeds = [100, 101] if fam in ('R2', 'R3') else [202]
        if SMOKE: seeds = seeds[:1]
        cs = [(r, f) for r in RHO2 for f in FS3]
        def one(arm, L):
            E = np.zeros((len(parts), len(cs)))
            for i, s in enumerate(parts):
                for j, (rho, f) in enumerate(cs):
                    v = []
                    for sd in seeds:
                        c = next(x for x in rg[s] if x['fam'] == fam and x['seed'] == sd and x['rho'] == rho and x['f'] == f); v.append(c['est'][arm][L] - c['truth'][L])
                    E[i, j] = np.mean(v)
            return E
        EA, EB = one('A', '64'), one('A4', '64'); EAall.append(EA); EBall.append(EB)
        res['families'][fam] = dict(mae6_A=mae6(EA), mae6_A4=mae6(EB), pass_=bool(mae6(EA) <= C['test8_family_factor'] * mae6(EB) and mae6(EA) <= C['test8_family_abs']), cohort_err_A=[float(x) for x in EA.mean(0)], cohort_err_A4=[float(x) for x in EB.mean(0)])
    res['unreached_families'] = [f for f in ('R2', 'R3', 'R4') if f not in fams]
    if EAall:
        br = boot_ratio(np.concatenate(EAall, 1), np.concatenate(EBall, 1), IX); res['pooled'] = br; res['pooled_pass'] = bool(br['ub97_5'] < C['test8_pooled_ub'])
    res['pass'] = bool(len(fams) == 3 and all(v['pass_'] for v in res['families'].values()) and res.get('pooled_pass', False)); res['reached'] = True; return res


def ev_grid():
    rg = load_recs('grid'); parts = sorted(rg); fl = jload(RES / 'flags_E.json'); res = {}
    cs = [(0.8, f) for f in FS3]; EA = err_matrix(rg, 'A', cs, '64', SEED0); EB = a4_matrix_from(rg, cs, '64')
    ans = [[not c['eflag'] for c in [find(rg[s], 'A', rho=0.8, seed=0, f=f) for f in FS3]] for s in parts]; ans = np.array(ans)
    res['A_mae6'] = mae6(EA); res['A4_mae6'] = mae6(EB); res['cohort_err_A'] = [float(x) for x in EA.mean(0)]; res['cohort_err_A4'] = [float(x) for x in EB.mean(0)]
    res['n_answered_ps'] = int(ans[:, 0].sum()); res['e_fire_rho0.8'] = fl['fire'].get('0.8', {}).get('seed0')
    res['err_answered'] = None
    if ans[:, 0].sum() >= C['answered_min']:
        ea = []
        for j, f in enumerate(FS3):
            for arm in ('A', 'A4'):
                E = (EA if arm == 'A' else EB)[:, j]; ea.append(abs(float(E[ans[:, j]].mean())))
        res['err_answered'] = max(ea); res['answered_clause'] = bool(max(ea) < C['answered_err'])
    else: res['answered_clause'] = True; res['answered_note'] = 'holds by abstention (fewer than 5 answered participant-seeds)'
    res['e_clause'] = bool(res['e_fire_rho0.8'] >= C['p4_rho8_min']); res['pass'] = bool(res['e_clause'] and res['answered_clause']); return res


def a4_matrix_from(recs, cs, L):
    parts = sorted(recs); return np.array([[np.mean([find(recs[s], 'A4', rho=rho, seed=sd, f=f)['est']['A4'][L] - truthN(s, sd, f, L) for sd in SEED0]) for (rho, f) in cs] for s in parts])


def ev_p4a():
    fl = jload(RES / 'flags_E.json')['fire']; res = {}
    res['rho0.9_fire'] = fl['0.9']['all']; res['rho_low_fire'] = (fl['0.25']['n_fire'] + fl['0.5']['n_fire']) / (fl['0.25']['n'] + fl['0.5']['n']); res['rho0.8_fire_seed0'] = fl['0.8']['seed0']
    res['clause_rho9'] = bool(res['rho0.9_fire'] >= C['p4_rho9_min']); res['clause_low'] = bool(res['rho_low_fire'] <= C['p4_low_max']); res['dead_class_pass'] = bool(res['clause_rho9'] and res['clause_low'])
    # answered clause at RHO .9 (A4 + flag E, stored A4): evaluated only if >= 5 answered participant-seeds
    E = jload(RES / 'flags_E.json')['E']['0.9']; ans = []; errs = {}
    for f in (0.25, 0.5, 0.75):
        v = [a4s(0.9, s, sd, f, '64') - truthN(s, sd, f, '64') for s in ALLP for sd in (0, 1) if not E[str(s)][sd] > C['e_thr']]; errs[fk(f)] = (len(v), float(np.mean(v)) if v else None)
    n_ans = errs[fk(0.5)][0]; res['rho0.9_answered'] = n_ans; res['rho0.9_answered_cohort_err'] = errs
    res['answered_clause_rho9'] = True if n_ans < C['answered_min'] else bool(all(abs(e[1]) < C['answered_err'] for e in errs.values()))
    return res


# ---------------------------------------------------------------- stages
def stage_stamp():
    if SMOKE: raise SystemExit('stamp is not a smoke stage')
    RES.mkdir(parents=True, exist_ok=True); extra = [p.name for p in RES.iterdir() if p.name not in ('PREREG.json', 'ledger.jsonl', 'ATTEMPTS.md')]
    if extra: raise SystemExit(f'stamp refuses: scored outputs exist {extra}')
    if (RES / 'PREREG.json').exists(): log('PREREG exists:', (RES / 'PREREG.json').read_text()[:300]); return 0
    obj = dict(protocol_path=str(PROTOCOL), protocol_sha256=_hl.sha256(PROTOCOL.read_bytes()).hexdigest(), script_path=str(Path(__file__).resolve()), script_sha256=SCRIPT_SHA256,
               imported_module_sha256={m: mod_sha(m) for m in ('td_cells', 'td_screen', 'pid_regen', 'd2_pe_tasks', 'pe_eogreg')}, constants_sha256=CONST_SHA, bootstrap_seed=C['boot_seed'],
               stamped_clock=time.strftime('%Y-%m-%d %H:%M:%S %Z'), stamped_unix=time.time(), hold=f'{HOLD} {HOLD_JOB}', python=sys.executable, scored_outputs_exist=False,
               smoke_before_stamp='s0 --smoke and stage smokes wrote to results/pilot_proximal_cmi/A_smoke (read by no stage)')
    jdump(RES / 'PREREG.json', obj); log('stamped', obj['stamped_clock']); return 0


def stage_s0():
    global CALLS
    verify_stamp(); out = {}; fails = []; ALL = C['participants']
    if (RES / 'gates.json').exists():                       # a failed gate may be rerun ONCE: keep the first attempt
        k = 1
        while (RES / f'gates_attempt{k}.json').exists(): k += 1
        (RES / 'gates.json').rename(RES / f'gates_attempt{k}.json')
    # ---- stored inputs, A4 frozen-number check (read-only, all 23 participants)
    truth, a4 = build_truth_a4(); jdump(RES / 'truth.json', truth); jdump(RES / 'a4_stored.json', a4); _TR.clear()
    cs = [(r, f) for r in RHO2 for f in (0.25, 0.5, 0.75)]; EA4 = a4_matrix(ALL, cs, '64', [0, 1]); EA4f = a4_matrix(ALL, cs, 'full', [0, 1]); fz = C['a4_frozen_L64']
    chk = dict(mae6=mae6(EA4), mae_rho25=mae6(EA4[:, 3:]), mae_rho5=mae6(EA4[:, :3]), participant_mae=pl_mae(EA4), mae6_full=mae6(EA4f), cohort_err=[float(x) for x in EA4.mean(0)], cells=[list(c) for c in cs])
    chk['expected_signed'] = dict(rho25=[-0.02339, -0.01465, -0.00484], rho5=[-0.006958, 0.002218, 0.010651])
    # cells order is RHO .5 then .25: cohort_err[:3] = RHO .5, [3:] = RHO .25
    chk['pass'] = bool(abs(chk['mae6'] - fz['mae6']) < 1e-5 and abs(chk['mae_rho25'] - fz['mae_rho25']) < 1e-5 and abs(chk['mae_rho5'] - fz['mae_rho5']) < 1e-5 and abs(chk['participant_mae'] - fz['participant_mae']) < 1e-5
                       and abs(chk['mae6_full'] - C['a4_frozen_full']['mae6']) < 1e-5 and np.allclose(chk['cohort_err'][3:], chk['expected_signed']['rho25'], atol=2e-5) and np.allclose(chk['cohort_err'][:3], chk['expected_signed']['rho5'], atol=2e-5))
    out['a4_frozen'] = chk; log('A4 frozen check', chk)
    if not chk['pass']: fails.append('a4_frozen')
    # ---- G-REGEN
    pairs = [(2, 0.5), (5, 0.25), (11, 0.9), (14, 0.5), (17, 0.25), (23, 0.9), (2, 0.25), (8, 0.9)]        # the smoke also runs all 8 pairs (code-path check of the add11 files)
    rows = []; worst = 0.0; worst4 = 0.0
    for s, rho in pairs:
        St = stored_any(rho, s); assert St is not None, (s, rho); P, D = gen_D(s, 0); L = PR.load(s, rho, 0); sig = rho * D['y'] + np.sqrt(1 - rho ** 2) * D['es'][0]
        exact = bool(np.array_equal(D['y'], L['y']) and np.array_equal(D['us'][0], L['u']) and np.array_equal(D['vraw'][0], L['v']) and np.array_equal(sig, L['sig'])); rows.append(dict(s=s, rho=rho, key='exact_y_u_v_sig', ok=exact))
        if not exact: fails.append(f'regen_exact_{s}_{rho}')
        T9 = {(c['seed'], c['cell']): c['arms']['A4']['yield'] for c in t9r(rho, s)['cells']}
        for f in ('null', 0.25, 0.75):
            cell = mk_cell(D, f, rho); y = D['y']; Sc = TS.scell(St, 0, f); got = {'mix_all': plain_yield(cell['Fm'], y)}
            if f != 'null':
                a, b = PR.amp(f); got['neural_only'] = plain_yield(cell['Fn'], y); Fo = TC.assemble(D, a, b, rho, TC.KAPPA0)[1]; got['ocular_only'] = plain_yield(Fo, y)
                CALLS += 1; got['mix_eogreg'] = TC.yld(TC.readout(cell['Fm'], y, PE.eogreg(cell['Om'])))
            for key, val in got.items():
                for w in LKEYS:
                    dd = abs(val[w] - Sc[key][w]); worst = max(worst, dd); rows.append(dict(s=s, rho=rho, f=str(f), key=key, width=w, regenerated=val[w], stored=Sc[key][w], absdiff=dd))
            if f != 'null':
                v = a4_est(cell['Fm'], y, cell['Om'])
                for w in LKEYS:
                    dd = abs(v[w] - T9[(0, fk(f))][w]); worst4 = max(worst4, dd); rows.append(dict(s=s, rho=rho, f=str(f), key='A4', width=w, regenerated=v[w], stored=T9[(0, fk(f))][w], absdiff=dd))
        log(f'G-REGEN ({s},{rho}): worst {worst:.2e} worst A4 {worst4:.2e}')
    out['g_regen'] = dict(pairs=pairs, max_absdiff_cells=worst, max_absdiff_A4=worst4, pass_=bool(worst <= 1e-4 and worst4 <= 1e-4 and not any(f_.startswith('regen_exact') for f_ in fails)), rows=rows)
    if not out['g_regen']['pass_']: fails.append('g_regen')
    # ---- G-LEAK, G-IDENT (2 participants, RHO .5, f .5, seed 0)
    gl, gi = [], []
    for s in SMOKE_PARTS:
        P, D = gen_D(s, 0); cell = mk_cell(D, 0.5, 0.5); PA, PB = blocks(cell['Om']); n = len(D['y']); M, te0 = masks(n); M0 = M[:1]; F = cell['Fm']; ycol = G(TC.ycols(D['y'])[:, 1])
        base = a_transform(PA, PB, ycol)(F, M0); idx = torch.nonzero(~M0[0]).squeeze(1); gg = torch.Generator(device=DEV); gg.manual_seed(5); pm = idx[torch.randperm(len(idx), generator=gg, device=DEV)]
        yp = ycol.clone(); yp[idx] = ycol[pm]; assert not torch.equal(yp, ycol); alt = a_transform(PA, PB, yp)(F, M0); d1 = float((base - alt).abs().max()); gl.append(dict(s=s, max_abs_diff=d1, pass_=bool(d1 <= 1e-12)))
        Mall = M[:20]; ap = a_parts(F, PA, PB, ycol, Mall); zB = ap['zB']; m = Mall.double()[:, :, None]; Xz = torch.cat([torch.ones(20, n, 1, **T64), ycol[None, :, None].expand(20, n, 1), zB], 2); Xm = Xz * m
        Bz = torch.linalg.solve(Xm.transpose(1, 2) @ Xm + 1e-8 * EYE(Xz.shape[2]), Xm.transpose(1, 2) @ F); Rid = zB @ Bz[:, 2:5]; rel = float((ap['Rm'] - Rid).norm() / ap['Rm'].norm()); gi.append(dict(s=s, rel_diff=rel, pass_=bool(rel <= 1e-6)))
    out['g_leak'] = dict(rows=gl, pass_=all(x['pass_'] for x in gl)); out['g_ident'] = dict(rows=gi, pass_=all(x['pass_'] for x in gi))
    for k in ('g_leak', 'g_ident'):
        if not out[k]['pass_']: fails.append(k)
    log('G-LEAK', gl, 'G-IDENT', gi)
    # ---- G-RELIAB (seed 0, RHO .25 and .5, all participants), E (all RHO), S1 on null / placebo, cos(sg_B, span wb)
    rel = {0.25: [], 0.5: []}; Eall = {f'{r:g}': {} for r in (0.25, 0.5, 0.6, 0.7, 0.8, 0.9)}; s1null = []; coss = []
    for s in ALL:
        for seed in (0, 1):
            P, D = gen_D(s, seed); y = D['y']; n = len(y)
            for rho in (0.25, 0.5, 0.6, 0.7, 0.8, 0.9):
                Om = TC.assemble(D, 0.0, PR.B0, rho, TC.KAPPA0)[3]; Eall[f'{rho:g}'].setdefault(str(s), [None, None])[seed] = float(TS.e_scope(y, Om))
        P, D = gen_D(s, 0); y = D['y']; n = len(y); fo = D2.folds(n); yG = G(y); e = D['es'][0]
        for rho in (0.25, 0.5):
            Om = TC.assemble(D, 0.0, PR.B0, rho, TC.KAPPA0)[3]; PA, PB = blocks(Om); Mo = torch.as_tensor(np.stack([fo != b for b in range(10)]), device=DEV); cp = cca_parts(PA, PB, yG, Mo, C['r'], C['lam_cca']); zB = cp['zB']; eG = G(e)
            num = den = 0.0
            for b in range(10):
                tr = torch.as_tensor(fo != b, device=DEV); te = ~tr; X1 = torch.stack([torch.ones_like(yG), yG], 1); X2 = torch.cat([X1, zB[b]], 1)
                c1 = torch.linalg.solve(X1[tr].T @ X1[tr] + 1e-8 * EYE(2), X1[tr].T @ eG[tr]); c2 = torch.linalg.solve(X2[tr].T @ X2[tr] + 1e-8 * EYE(X2.shape[1]), X2[tr].T @ eG[tr])
                num += float(((eG[te] - X2[te] @ c2) ** 2).sum()); den += float(((eG[te] - X1[te] @ c1) ** 2).sum())
            rel[rho].append(1 - num / den)
            if rho == 0.5:
                Mall = torch.ones(1, n, dtype=torch.bool, device=DEV); cpa = cca_parts(PA, PB, yG, Mall, C['r'], C['lam_cca']); sgB = G(D['sgs'][0][23:69]); W = cpa['wb'][0]; Q_, _ = torch.linalg.qr(W); proj = Q_ @ (Q_.T @ sgB); coss.append(float(proj.norm() / sgB.norm()))
        for seed in (0, 1):
            P, D = gen_D(s, seed); y = D['y']; n = len(y); PA, PB = blocks(D['O']); real = s1_test(PA, PB, y, s, seed, 0.5)
            pm = np.random.default_rng([777, s, seed]).permutation(n); plc = s1_test(PA, PB[torch.as_tensor(pm, device=DEV)], y, s, seed, 0.5); s1null.append(dict(s=s, seed=seed, real_O_flag=real['flag'], real_O_obs=real['obs'], placebo_flag=plc['flag'], placebo_flag_max=plc['flag_max'], placebo_obs=plc['obs']))
        log(f's0 diagnostics sub {s} done')
    out['g_reliab'] = {f'{r:g}': dict(median=float(np.median(v)), min=float(np.min(v)), values=v) for r, v in rel.items()}
    out['g_reliab']['pass_'] = bool(all(np.median(v) >= C['reliab_median'] and np.min(v) >= C['reliab_min'] for v in rel.values()))
    if not out['g_reliab']['pass_']: fails.append('g_reliab')
    log('G-RELIAB', {k: (v['median'], v['min']) for k, v in out['g_reliab'].items() if k != 'pass_'})
    # E: fires, gate against stored scope_E (where the stored participant exists)
    sE = jload(ROOT / 'results/yield_projector/runner_T9R/scope_E.json')['participants']; worstE = 0.0
    for r in (0.25, 0.5, 0.9):
        for s in ALL:
            for sd in (0, 1): worstE = max(worstE, abs(Eall[f'{r:g}'][str(s)][sd] - sE[f'sub-{s:03d}']['E'][f'{r:g}']['seeds'][sd]))
    fire = {}
    for r, d in Eall.items():
        vals = np.array([d[str(s)] for s in ALL]); f_ = vals > C['e_thr']; fire[r] = dict(n=int(f_.size), n_fire=int(f_.sum()), all=float(f_.mean()), seed0=float(f_[:, 0].mean()), range=[float(vals.min()), float(vals.max())])
    jdump(RES / 'flags_E.json', dict(NOT_CITABLE='until a blind bug review', e_thr=C['e_thr'], E=Eall, fire=fire, max_absdiff_vs_stored_scope_E=worstE)); out['E'] = dict(fire=fire, max_absdiff_vs_stored=worstE, pass_=bool(worstE <= 1e-4))
    if not out['E']['pass_']: fails.append('E_vs_stored')
    log('E fire', fire, 'vs stored', worstE)
    out['s1_calibration'] = dict(rows=s1null, real_O_fire=float(np.mean([x['real_O_flag'] for x in s1null])), placebo_fire=float(np.mean([x['placebo_flag'] for x in s1null])), placebo_fire_max_reading=float(np.mean([x['placebo_flag_max'] for x in s1null])))
    out['cos_sgB_span_wb'] = dict(values=coss, median=float(np.median(coss)), min=float(np.min(coss))); out['power_table'] = power_table(EA4); out['A4_signed_errors'] = chk['cohort_err']
    # ---- G-TIME smoke: A, A_orc, G1, G2, G3, A4, plain on one cell for 2 participants
    rates = {}; tm = {k: [] for k in ('A', 'A_orc', 'A4', 'G1', 'G2', 'G3', 'plain')}
    for rep, s in enumerate(SMOKE_PARTS):
        P, D = gen_D(s, 0); cell = mk_cell(D, 0.5, 0.5); PA, PB = blocks(cell['Om'])
        for k in tm:
            for w in (0, 1):                                   # first pass warms up, second is timed
                torch.cuda.synchronize(); t1 = time.time(); c1 = CALLS
                if k == 'plain': plain_yield(cell['Fm'], cell['y'])
                else: arm_yield(k, cell, PA, PB)
                torch.cuda.synchronize(); dt = time.time() - t1
                if w == 1: tm[k].append(dt / max(CALLS - c1, 1))
    rates = {k: float(np.mean(v)) for k, v in tm.items()}; out['g_time'] = dict(s_per_call=rates, measured=tm)
    dropped = []; tot = lambda: sum(project_h_rates(st, rates, dropped) for st in ('p0', 'reach', 'p1', 'p6', 'test8', 'guards', 'perm', 'grid', 'swap', 'desc')) - project_h_rates('desc', rates, dropped) + 0.14
    # primary total = steps 3-11 (p0 .. swap) + 0.14 h overhead; desc (step 12) is outside the primary total and the first to go when the cap is binding (sec 10)
    proj_total = tot(); order = ['swap', 'grid_readouts', 'perm_seed1']              # step 12 (desc) is last in the run order: the ledger guard drops it when the cap would be exceeded
    for d_ in order:
        if proj_total <= C['ledger_cap_h']: break
        dropped.append(d_); proj_total = tot()
    out['g_time'].update(projected_primary_h=proj_total, dropped=dropped, projected_desc_h=project_h_rates('desc', rates, dropped), per_step_h={st: project_h_rates(st, rates, dropped) for st in STEP_CALLS})
    jdump(RES / 'plan.json', dict(rates=rates, dropped=dropped, projected_primary_h=proj_total, per_step_h=out['g_time']['per_step_h'])); log('G-TIME', out['g_time'])
    out['fails'] = fails; out['row'] = 'VOID_row1' if any(f_ in fails for f_ in ('g_regen', 'g_leak', 'g_ident', 'regen_exact', 'a4_frozen', 'E_vs_stored')) else ('VOID_row2' if 'g_reliab' in fails else 'gates_pass')
    out['ZV_zero_variance_columns'] = ZV[0]; out['NOT_CITABLE'] = 'until a blind bug review'; jdump(RES / 'gates.json', out)
    log('S0 done; fails', fails, 'row', out['row']); return CALLS


def project_h_rates(stage, rates, dropped):
    items = list(STEP_CALLS[stage])
    if stage == 'perm' and 'perm_seed1' in dropped: items = [(c // 2, k) for c, k in items]
    if (stage == 'grid' and 'grid_readouts' in dropped) or (stage == 'swap' and 'swap' in dropped) or (stage == 'desc' and 'desc' in dropped): items = []
    return sum(c * 1.25 * rates[k] for c, k in items) / 3600.0


def require_gates():
    g = jload(RES / 'gates.json')
    if g['row'] != 'gates_pass': raise SystemExit(f'STOP: gates {g["row"]} (fails {g["fails"]})')


def verdicts():
    p = RES / 'verdicts.json'; return jload(p) if p.exists() else {}


def set_verdict(stage, ok, row, result, **kw):
    v = verdicts(); v[stage] = {'pass': bool(ok), 'row': row, 'result': result, **kw}; jdump(RES / 'verdicts.json', v)


def require_rows_1_5():
    require_gates(); v = verdicts()
    for st in ('p0', 'reach', 'p1'):
        if st not in v: raise SystemExit(f'STOP: {st} has no verdict yet')
        if not v[st]['pass']:
            if SMOKE: log(f'smoke: {st} failed (ignored in smoke, 2 participants)'); continue
            raise SystemExit(f'STOP: {st} failed (outcome-table row {v[st]["row"]}): later stages are not run')


def stage_p0():
    verify_stamp(); require_gates(); budget_guard('p0')
    def fn(s):
        out = []
        for sd in SEEDS2:
            P, D = gen_D(s, sd)
            for kind, rho, f in (('ocular', 0.5, 1.0), ('ocular', 0.25, 1.0), ('null', 0.5, 'null')):
                cell = mk_cell(D, f, rho); out.append(dict(kind=kind, rho=rho, seed=sd, f=f, est={'A': arm_yield('A', cell)}))
        return out
    run_stage('p0', PL, fn); r = ev_p0(); set_verdict('p0', r['pass'], None if r['pass'] else 3, r); jdump(RES / 'p0_result.json', r); log('P0', r)
    return CALLS


def stage_reach():
    verify_stamp(); require_gates(); v = verdicts()
    if ('p0' not in v or not v['p0']['pass']) and not SMOKE: raise SystemExit('STOP: P0 not passed (row 3)')
    budget_guard('reach')
    def fn(s):
        out = []
        for rho in RHO2:
            for sd in SEEDS2:
                P, D = gen_D(s, sd)
                for f in FS3: cell = mk_cell(D, f, rho); out.append(dict(rho=rho, seed=sd, f=f, est={'A_orc': arm_yield('A_orc', cell)}))
        return out
    run_stage('reach', PL, fn); r = ev_reach(); set_verdict('reach', r['pass'], None if r['pass'] else 4, r); jdump(RES / 'reach_result.json', r); log('REACH', r)
    return CALLS


def p1_pass(rho, s):
    out = []
    for sd in SEEDS2:
        P, D = gen_D(s, sd); y = D['y']; cell0 = mk_cell(D, FS3[0], rho); PA, PB = blocks(cell0['Om']); s1r = s1_test(PA, PB, y, s, sd, rho)
        for f in FS3:
            cell = cell0 if f == FS3[0] else mk_cell(D, f, rho); fl = cell_flags(cell, PA, PB, s1r); out.append(dict(rho=rho, seed=sd, f=f, est={'A': arm_yield('A', cell, PA, PB)}, **fl))
    return out


def stage_p1():
    verify_stamp(); require_gates(); v = verdicts()
    if ('p0' not in v or not v['p0']['pass'] or 'reach' not in v or not v['reach']['pass']) and not SMOKE: raise SystemExit('STOP: P0 / REACH not passed (rows 3, 4)')
    budget_guard('p1')
    run_stage('p1_rho0.5', PL, lambda s: p1_pass(0.5, s)); r1 = ev_p1(rho_only=0.5); log('P1 RHO .5 first pass', r1)
    jdump(RES / 'p1_rho0.5_result.json', r1); first_ok = bool(r1['itt']['core_pass_rho5'])
    if not first_ok and not SMOKE:
        set_verdict('p1', False, 5, r1, stopped_after='rho0.5 (clause c or d failed)'); log('STAGE-1 DECISION: STOP (rho .5 clause c or d failed) -> DEAD row 5'); return CALLS
    run_stage('p1_rho0.25', PL, lambda s: p1_pass(0.25, s)); r = ev_p1(); jdump(RES / 'p1_result.json', r); log('P1 full', r)
    ok = bool(r['itt']['core_pass']); set_verdict('p1', ok, None if ok else 5, r); log('STAGE-1 DECISION:', 'CONTINUE' if ok else 'STOP -> DEAD row 5', 'itt core', ok, 'raw core', r['raw']['core_pass'])
    return CALLS


def stage_p6():
    verify_stamp(); require_rows_1_5(); budget_guard('p6')
    def fn(s):
        out = []; P, D = gen_D(s, 0); y = D['y']; n = len(y); eta = np.random.default_rng([313, s, 0]).standard_normal(n); s2 = np.random.default_rng([314, s, 0]).choice([-1.0, 1.0], 70); a, b = PR.amp(0.5)
        Deta = dict(D); Deta['Nn'] = G(np.outer(y + 0.5 * eta, D['us'][0])); Fn_eta = TC.assemble(Deta, a, 0.0, 0.5, TC.KAPPA0)[0]; truth_eta = plain_yield(Fn_eta, y)
        for rho in RHO2:
            Fm = TC.assemble(Deta, a, b, rho, TC.KAPPA0)[2]; Om0 = TC.assemble(Deta, a, b, rho, TC.KAPPA0)[3]
            for c in (0.0, 0.05, 0.1, 0.2):
                Om = Om0 + c * (y + 0.5 * eta)[:, None] * (D['Osd'] * s2)[None, :] if c > 0 else Om0; cell = dict(Fm=Fm, Om=Om, y=y, D=Deta, rho=rho, f=0.5); PA, PB = blocks(Om); s1r = s1_test(PA, PB, y, s, 0, rho)
                out.append(dict(kind='a', rho=rho, c=c, seed=0, f=0.5, est={'A': arm_yield('A', cell, PA, PB)}, truth_eta=truth_eta, **cell_flags(cell, PA, PB, s1r)))
        g = np.random.default_rng([315, s, 0]).standard_normal(n); sg2 = np.random.default_rng([316, s, 0]).choice([-1.0, 1.0], 70); gv = np.random.default_rng([317, s, 0]).standard_normal(32)
        vg = (((gv * P['Sv'][20:52] / math.sqrt(P['n'])) @ P['Vt'][20:52]) * P['Fsd'])
        for rho in RHO2:
            cell0 = mk_cell(D, 0.5, rho); Fm_g = cell0['Fm'] + 0.45 * G(g)[:, None] * G(vg)[None, :]; Om_g = cell0['Om'] + 0.3 * g[:, None] * (D['Osd'] * sg2)[None, :]
            cell = dict(Fm=Fm_g, Om=Om_g, y=y, D=D, rho=rho, f=0.5); PA, PB = blocks(Om_g); s1r = s1_test(PA, PB, y, s, 0, rho)
            out.append(dict(kind='b', rho=rho, c=0.0, seed=0, f=0.5, est={'A': arm_yield('A', cell, PA, PB)}, **cell_flags(cell, PA, PB, s1r)))
        return out
    run_stage('p6', PL, fn)
    r = ev_p6(); jdump(RES / 'p6_result.json', r); log('P6', r); set_verdict('p6', not r['dead_p6'], 6 if r['dead_p6'] else None, r)
    return CALLS


def stage_test8():
    verify_stamp(); require_rows_1_5(); budget_guard('test8'); T8 = [5, 2] if SMOKE else C['test8']; T8 = sorted(T8); fams = ['R2', 'R4'] if SMOKE else ['R2', 'R3', 'R4']; seeds_of = lambda fam: ([100, 101] if fam in ('R2', 'R3') else [202])
    ci = {(0.25, 0.25): 2, (0.25, 0.5): 3, (0.25, 0.75): 4, (0.5, 0.25): 6, (0.5, 0.5): 7, (0.5, 0.75): 8}; cg = TC.grid_cells()
    for (rho, f), i in ci.items(): assert cg[i][1] == rho and abs(cg[i][4] - f) < 1e-12, (rho, f, i, cg[i])
    def shard(fam, s, sd): return np.load(ROOT / (f'results/td_pilot/cells/r4test/R4_sub{s:03d}_seed202.npz' if fam == 'R4' else f'results/td_pilot/cells/test/{fam}_sub{s:03d}_seed{sd}.npz'))
    fsub = FS3
    # ---- phase 1: regeneration check (neural-only read-out of each participant, seed, f vs the shard truth)
    rp = RES / 'test8_regen.json'
    if not rp.exists():
        rows = []
        for fam in fams:
            for s in T8:
                for sd in (seeds_of(fam)[:1] if SMOKE else seeds_of(fam)):
                    P, D = gen_D(s, sd, fam); Z = shard(fam, s, sd); Y = Z['Y']
                    for f in fsub:
                        a, b = PR.amp(f); Fn = TC.assemble(D, a, 0.0, 0.5, TC.KAPPA0)[0]; got = plain_yield(Fn, D['y'])
                        for rho in RHO2:
                            i = ci[(rho, f)]; rows.append(dict(fam=fam, s=s, seed=sd, f=f, rho=rho, ci=i, d64=abs(got['64'] - float(Y[i, 0])), dfull=abs(got['full'] - float(Y[i, 1]))))
                    log(f'test8 regen {fam} sub {s} seed {sd}')
        summ = {fam: dict(max_abs=float(max(max(r_['d64'], r_['dfull']) for r_ in rows if r_['fam'] == fam))) for fam in fams}
        for fam in fams: summ[fam]['pass'] = bool(summ[fam]['max_abs'] <= 1e-4)
        jdump(rp, dict(rows=rows, summary=summ)); log('TEST-8 regeneration', summ)
    regen = jload(rp); ok_f = [f for f in fams if regen['summary'][f]['pass']]
    def fn(s):
        out = []
        for fam in ok_f:
            for sd in (seeds_of(fam)[:1] if SMOKE else seeds_of(fam)):
                P, D = gen_D(s, sd, fam); Z = shard(fam, s, sd); Y = Z['Y']; BL = Z['BL']
                for rho in RHO2:
                    for f in fsub:
                        cell = mk_cell(D, f, rho); i = ci[(rho, f)]; PA, PB = blocks(cell['Om']); est = {'A': arm_yield('A', cell, PA, PB)}
                        if fam == 'R4': est['A4'] = arm_yield('A4', cell)
                        else: est['A4'] = {'64': float(BL[i, 2]), 'full': float(BL[i, 3])}
                        out.append(dict(fam=fam, rho=rho, seed=sd, f=f, est=est, truth={'64': float(Y[i, 0]), 'full': float(Y[i, 1])}))
        return out
    run_stage('test8', T8, fn); r = ev_test8(); jdump(RES / 'test8_result.json', r); log('TEST-8', r); set_verdict('test8', r.get('pass'), None, r)
    return CALLS


def guards_fn(s):
    out = []
    for rho in RHO2:
        for sd in SEEDS2:
            P, D = gen_D(s, sd)
            for f in FS3:
                cell = mk_cell(D, f, rho); e, ef = e_flag(cell['y'], cell['Om']); out.append(dict(rho=rho, seed=sd, f=f, eflag=ef, E=e, est={g_: arm_yield(g_, cell) for g_ in ('G1', 'G2', 'G3')}))
    return out


def stage_guards():
    verify_stamp(); require_rows_1_5(); budget_guard('guards'); run_stage('guards', PL, guards_fn); r = ev_guards(); jdump(RES / 'guards_result.json', r); log('GUARDS', r); set_verdict('guards', r['p5_pass'], None, r)
    return CALLS


def stage_perm():
    verify_stamp(); require_rows_1_5(); budget_guard('perm'); dropped = plan()['dropped'] if plan() else []; seeds = [0] if ('perm_seed1' in dropped or SMOKE) else SEEDS2
    def fn(s):
        out = []
        for sd in seeds:
            P, D = gen_D(s, sd); y = D['y']; n = len(y); perm = np.random.default_rng([777, s, sd]).permutation(n); cell0 = mk_cell(D, FS3[0], 0.5); PA, PB = blocks(cell0['Om']); PBp = PB[torch.as_tensor(perm, device=DEV)]; s1r = s1_test(PA, PBp, y, s, sd, 0.5)
            for f in FS3:
                cell = cell0 if f == FS3[0] else mk_cell(D, f, 0.5); out.append(dict(rho=0.5, seed=sd, f=f, est={'A_perm': arm_yield('A_perm', cell, PA, PB, perm)}, s1_flag=s1r['flag'], s1_flag_max=s1r['flag_max'], s1=s1r['obs'], s1_thr=s1r['thr'], **s23(cell['Fm'], PA, PBp, y)))
        return out
    run_stage('perm', PL, fn); r = ev_perm(); jdump(RES / 'perm_result.json', r); log('P2', r); set_verdict('perm', r['pass'], None, r)
    return CALLS


def stage_grid():
    verify_stamp(); require_rows_1_5(); budget_guard('grid'); dropped = plan()['dropped'] if plan() else []
    if 'grid_readouts' in dropped and not SMOKE: log('grid read-outs dropped by the drop order (E clause stays: flags_E.json)'); return CALLS
    def fn(s):
        out = []; P, D = gen_D(s, 0)
        for f in FS3:
            cell = mk_cell(D, f, 0.8); e, ef = e_flag(cell['y'], cell['Om']); out.append(dict(rho=0.8, seed=0, f=f, eflag=ef, E=e, est={'A': arm_yield('A', cell), 'A4': arm_yield('A4', cell)}))
        return out
    run_stage('grid', PL, fn); r = ev_grid(); jdump(RES / 'grid_result.json', r); log('RHO .8', r); set_verdict('grid', r['pass'], None, r)
    return CALLS


def stage_swap():
    verify_stamp(); require_rows_1_5(); budget_guard('swap')
    def fn(s):
        out = []
        for sd in SEEDS2:
            P, D = gen_D(s, sd)
            for f in FS3: cell = mk_cell(D, f, 0.5); out.append(dict(rho=0.5, seed=sd, f=f, est={'A_swap': arm_yield('A_swap', cell)}))
        return out
    run_stage('swap', PL, fn); r = ev_swap(); jdump(RES / 'swap_result.json', r); log('P3', r); set_verdict('swap', r['pass'], None, r)
    return CALLS


def stage_desc():
    verify_stamp(); require_rows_1_5(); budget_guard('desc')
    def fn(s):
        out = []; P, D = gen_D(s, 0)
        for rho in (0.6, 0.7):
            for f in FS3: cell = mk_cell(D, f, rho); out.append(dict(rho=rho, seed=0, f=f, est={'A': arm_yield('A', cell), 'A4': arm_yield('A4', cell)}))
        for rho in RHO2:
            for sd in SEEDS2:
                P, D = gen_D(s, sd)
                for f in FS3: cell = mk_cell(D, f, rho); out.append(dict(rho=rho, seed=sd, f=f, est={'A_r1': arm_yield('A_r1', cell)}))
        return out
    run_stage('desc', PL, fn); recs = load_recs('desc'); parts = sorted(recs); res = {}
    for rho in (0.6, 0.7):
        cs = [(rho, f) for f in FS3]; res[f'rho{rho:g}'] = dict(A_mae6=mae6(err_matrix(recs, 'A', cs, '64', SEED0)), A4_mae6=mae6(err_matrix(recs, 'A4', cs, '64', SEED0)))
    res['r1_mae6'] = mae6(err_matrix(recs, 'A_r1', six_cells(), '64', SEEDS2)); jdump(RES / 'desc_result.json', res); log('DESC', res)
    return CALLS


def stage_summary():
    verify_stamp(); out = dict(NOT_CITABLE='until a blind bug review', script_sha256=SCRIPT_SHA256, protocol_sha256=_hl.sha256(PROTOCOL.read_bytes()).hexdigest()); v = verdicts(); g = jload(RES / 'gates.json'); out['gates'] = dict(row=g['row'], fails=g['fails'], g_time=g['g_time']['s_per_call'], dropped=g['g_time']['dropped'])
    out['ledger_sum_h'] = ledger_sum_h(); out['p4a'] = ev_p4a(); unreached = []; row = None; why = ''
    if g['row'] == 'VOID_row1': row, why = 1, 'G-REGEN / G-LEAK / G-IDENT / frozen / E gate failed'
    elif g['row'] == 'VOID_row2': row, why = 2, 'G-RELIAB failed'
    elif 'p0' not in v: row, why = None, 'P0 not run'
    elif not v['p0']['pass']: row, why = 3, 'P0 failed'
    elif 'reach' not in v: row, why = None, 'REACH not run'
    elif not v['reach']['pass']: row, why = 4, 'REACH failed (UNREACHABLE)'
    elif 'p1' not in v: row, why = None, 'P1 not run'
    elif not v['p1']['pass']: row, why = 5, 'P1 core clause failed'
    if row is not None:
        out['row'] = row; out['why'] = why; out['outcome'] = {1: 'VOID', 2: 'VOID', 3: 'DEAD', 4: 'DEAD (UNREACHABLE)', 5: 'DEAD'}[row]
        if row == 5: out['p4a_E_counts_only'] = out['p4a']
    elif why: out['row'] = None; out['why'] = why
    else:
        r6dead = ('p6' in v and v['p6']['row'] == 6) or not out['p4a']['dead_class_pass']; out['p4a_dead_class'] = out['p4a']['dead_class_pass']
        if 'p6' not in v and out['p4a']['dead_class_pass']: unreached.append('p6')
        if r6dead: out['row'] = 6; out['outcome'] = 'DEAD'; out['why'] = 'P6 c=.05 / <5 of 6 failed, or P4a E clauses failed'
        else:
            piv = False; pivot_unreached = []
            for st in ('guards', 'perm'):
                if st not in v: pivot_unreached.append(st)
                elif not v[st]['pass']: piv = True
            if piv: out['row'] = 7; out['outcome'] = 'PIVOT'; out['why'] = 'P5 or P2 failed'
            else:
                ref = []; p1r = v['p1']['result']['itt']
                if not p1r['secondary_plmae']: ref.append('participant-level clause')
                if not p1r['full_pass']: ref.append('full-width clause')
                for st, nm in (('swap', 'P3'), ('grid', 'RHO .8 clause'), ('test8', 'TEST-8')):
                    if st not in v: unreached.append(st)
                    elif not v[st]['pass']: ref.append(nm)
                if 'p6' in v and v['p6']['result']['refine_p6']: ref.append('P6 c>=.1 or (b)')
                if 'p6' in v:
                    p4b = v['p6']['result']['p4b']['clause']
                    if p4b is False: ref.append('P4b clause')
                if not out['p4a']['answered_clause_rho9']: ref.append('P4a answered clause')
                ref += [f'unreached {u}' for u in unreached + pivot_unreached]
                if ref: out['row'] = 8; out['outcome'] = 'REFINE'; out['why'] = '; '.join(ref)
                else: out['row'] = 9; out['outcome'] = 'PROCEED'; out['why'] = 'everything passes'
    out['verdicts'] = {k: dict(pass_=x.get('pass'), row=x.get('row')) for k, x in v.items()}; jdump(RES / 'summary_A.json', out); log('SUMMARY', {k: out[k] for k in ('row', 'why') if k in out}, out.get('outcome'))
    return 0


STAGES = dict(stamp=stage_stamp, s0=stage_s0, p0=stage_p0, reach=stage_reach, p1=stage_p1, p6=stage_p6, test8=stage_test8, guards=stage_guards, perm=stage_perm, grid=stage_grid, swap=stage_swap, desc=stage_desc, summary=stage_summary)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=list(STAGES)); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); ledger_line(dict(step=STEP[a.stage], stage=a.stage, event='start', t_start_unix=T_START, smoke=SMOKE, script_sha256=SCRIPT_SHA256[:12])); err = None; calls = 0
    try: calls = STAGES[a.stage]() or 0
    except Unreached as e: err = f'UNREACHED: {e}'; log(err)
    except SystemExit as e: err = f'EXIT: {e}'; raise
    except BaseException as e: err = f'{type(e).__name__}: {e}'; raise
    finally: ledger_end(a.stage, CALLS, err)


if __name__ == '__main__':
    main()
