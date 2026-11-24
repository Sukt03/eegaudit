"""td_cells.py: GPU generator of planted cells with VARIED recipes for the pilot of the trained decomposer (runner, 23-24 Nov 2026).
Protocol: results/td_pilot/PREREG_pilot.json (written before any cell; this script stamps its hash + clock time in results/td_pilot/PREREG_STAMP.json before the first read-out call).
Cell = (participant background F of ds003458, n_use kept trials, target y, recipe theta). Read-out = registered D2.pe_gain (L 64 and full), yield = gain - mean of 3 shift floors, bits, unclipped.
Truth = finite-sample SINGLE-SOURCE yield of the same cell and read-out: Y_N (F + neural plant only), Y_O (F + ocular plant only); the joint yield Y_NO is the measured mixed cell.
Families: TRAIN prior (train + validation participants), R0 (stored planted_v2 recipe, stored rng stream), R2 (rank 3), R3 (structured own-leakage eye plant), R4 (window PCs 0-19).
Stages: stamp | gate | jitter | gen --kinds val,cal,train,test,r4test | check.  All numerics in torch on the GPU. Named CPU exceptions: numpy rng draws and the numpy SVD of the PC basis (bit-exact
regeneration of the stored cells), MNE inside the cached ocular arrays, np.lstsq of the own-leakage map (R3 only), JSON/IO.
Shard (np.savez): X1 (m, 58) D1 scalar statistics, PCX (m, 3, 64) per-PC channels, TOK (m, 470, 76) fp16 per-trial tokens, NUSE (m), Y (m, 6) truths, BL (m, 4) baselines (nan if not scored),
META (m, k) recipe parameters for AUDIT ONLY (never an input or a label), plus names.
"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "3")
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import d2_pe_tasks as D2
import pe_eogreg as PE
import pid_regen as PR
import td_screen as TS           # reuse: ocular cache patch (PE.ocular_v2), base(), e_scope(), stored(), scell(), G

DEV = 'cuda'; T64 = dict(dtype=torch.float64, device=DEV); t0 = time.time(); LS = (64, 'full')
log = lambda *m: print(f'[{time.time() - t0:8.1f}s]', *m, flush=True)
OUT = ROOT / 'results/td_pilot'; CELLS = OUT / 'cells'
G = TS.G
TRAIN = [1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 21, 22]; VAL = [4, 10, 16]; TEST = [2, 5, 8, 11, 14, 17, 20, 23]
assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST)) and not (set(VAL) & set(TEST)) and len(set(TRAIN + VAL + TEST)) == 23
A0_AMP, B0_AMP, KAPPA0 = 0.14, 0.45, 0.3; NMAX = 480; NTOK = 76; P_PERM = 19
FAMCODE = dict(TRAIN=1, R0=2, R2=3, R3=4, R4=5)
STORED_RHOS = (0.25, 0.5, 0.9)
META_NAMES = ['fam', 'participant', 'seed', 'cell', 'n_use', 'cls', 'rho', 'kappa', 'a', 'b', 'f', 'w0', 'wd', 'r_n', 'r_e', 'eye_class', 'cos', 'marg', 'has_A1A4']
X1_NAMES = (['joint64', 'jointfull'] + [f'spec{i}' for i in range(8)] + [f'tmap_energy{i}' for i in range(8)] + [f'tmap_rep{i}' for i in range(8)] + ['E'] + [f'sv{i}' for i in range(1, 6)] + ['theta', 'k', 'n']
            + [f'foldgain{i}' for i in range(10)] + [f'foldfloor{i}' for i in range(10)] + ['floor1', 'floor2', 'floor3'])
FORBIDDEN = {'rho', 'f', 'a', 'b', 'kappa', 'cos', 'neural_only', 'ocular_only', 'a1', 'a4', 'a3', 'YN', 'YO', 'w0', 'wd', 'fam', 'cls'}
assert len(X1_NAMES) == 58 and not (set(X1_NAMES) & FORBIDDEN)
EYE_CLASS = dict(random=0, own=1); MARG = dict(gauss=0, t3=1, gamma=2)


def jdump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f'.tmp{os.getpid()}'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def ledger(stage, extra=None):
    """GPU wall-time ledger: one line per step start/end (budget accounting: at most 2 A100 hours)."""
    with open(OUT / 'gpu_ledger.jsonl', 'a') as fh: fh.write(json.dumps(dict(stage=stage, unix=time.time(), clock=time.strftime('%Y-%m-%d %H:%M:%S'), code=CODE_SHA256[:12], **(extra or {}))) + '\n')


# ---------------------------------------------------------------- participant background (cached)
_PB = {}
def pb(s):
    if s not in _PB:
        F, O = TS.base(s); n = len(F); Fsd = F.std(0) + 1e-9; Osd = O.std(0) + 1e-9
        Fz = (F - F.mean(0)) / Fsd; _, Sv, Vt = np.linalg.svd(Fz, full_matrices=False)        # numpy SVD: bit-identical to pid_regen.load / gf_planted_calib_v2
        _PB[s] = dict(F=F, O=O, n=n, Fsd=Fsd, Osd=Osd, Sv=Sv, Vt=Vt, Fg=G(F), Og=O)
    return _PB[s]


def own_leak_map(P):
    if 'BO' not in P:
        O1 = np.c_[np.ones(P['n']), P['O']]; B = np.linalg.lstsq(O1, P['F'], rcond=None)[0]; P['BO'] = B[1:]            # (70, D) the participant's own fitted eye-to-EEG map
        rng = np.random.default_rng([7, 7, 7]); skd = P['Sv'][20:52] / np.sqrt(P['n']); Vk = P['Vt'][20:52]
        P['ref_norm'] = float(np.median([np.linalg.norm(((rng.standard_normal(32) * skd) @ Vk) * P['Fsd']) for _ in range(20)]))
    return P['BO'], P['ref_norm']


def struct_sg(k):
    """Structured regressor pattern over [VEOG 23 | Fp1-Fp2 23 | F7-F8 23 | blink flag]; k = 0 slow dipole shift, k = 1 saccadic spike + blink; normalised to norm^2 = 70."""
    t = np.arange(23.0)
    if k == 0: w = 0.5 * (1 - np.cos(np.pi * np.clip(t / 22.0, 0, 1))); wts = (1.0, 0.5, -0.3); bl = 0.0
    else: w = np.exp(-0.5 * ((t - 7.0) / 1.5) ** 2); wts = (0.4, 1.0, -1.0); bl = 1.0
    sg = np.concatenate([wts[0] * w, wts[1] * w, wts[2] * w, [bl]]); return sg * math.sqrt(70.0) / np.linalg.norm(sg)


# ---------------------------------------------------------------- recipes
def draw_theta_train(rng):
    cls = int(rng.choice(4, p=[.1, .1, .1, .7])); n_use = int(rng.choice([150, 300, 0], p=[.25, .25, .5]))
    wd = int(rng.choice([8, 16])); w0 = int(rng.integers(20, 49)); r_n = int(rng.choice([1, 2])); r_e = int(rng.choice([1, 2]))
    rho = float(rng.uniform(.05, .35) if rng.random() < .5 else rng.uniform(.65, .95)); kappa = float(math.exp(rng.uniform(math.log(.05), math.log(1.0))))
    a = float(math.exp(rng.uniform(math.log(.04), math.log(.32)))); b = float(math.exp(rng.uniform(math.log(.08), math.log(.9))))
    cos = float(rng.uniform(0, .9)) if rng.random() < .5 else -1.0; marg = ['gauss', 't3', 'gamma'][int(rng.choice(3, p=[.5, .25, .25]))]
    if cls in (0, 1): b = 0.0
    if cls in (0, 2): a = 0.0
    return dict(fam='TRAIN', cls=cls, n_use=n_use, wd=wd, w0=w0, r_n=r_n, r_e=r_e, rho=rho, kappa=kappa, a=a, b=b, cos=cos, marg=marg, eye_class='random', f=float('nan'))


def fam_theta(fam):
    base = dict(fam=fam, cls=3, n_use=0, wd=32, w0=20, r_n=1, r_e=1, kappa=KAPPA0, cos=-1.0, marg='gauss', eye_class='random', f=float('nan'), rho=float('nan'), a=0.0, b=0.0)
    if fam == 'R2': base.update(r_n=3, r_e=3)
    if fam == 'R3': base.update(r_e=2, eye_class='own')
    if fam == 'R4': base.update(w0=0, wd=20)
    return base


def draw_data(P, th, rng):
    """All random draws of one (participant, seed, family) in the stored order: y, g_u (per neural component), then per eye component (sg, g_v), then e per eye component.
    r = 1 reproduces gf_planted_calib_v2 / pid_regen.load exactly when called with the stored stream default_rng(910000 + 100 s + seed)."""
    n_full = P['n']; n = n_full if th['n_use'] == 0 else th['n_use']; wd, w0 = th['wd'], th['w0']
    skd = P['Sv'][w0:w0 + wd] / np.sqrt(n_full); Vk = P['Vt'][w0:w0 + wd]; scale = math.sqrt(32.0 / wd)
    pat = lambda g: (((g * skd) * scale) @ Vk) * P['Fsd']
    if th['marg'] == 'gauss': y = rng.standard_normal(n)
    elif th['marg'] == 't3': y = rng.standard_t(3, n) / math.sqrt(3.0)
    else: y = (rng.gamma(2.0, 1.0, n) - 2.0) / math.sqrt(2.0)
    gus = [rng.standard_normal(wd) for _ in range(th['r_n'])]; us = [pat(g) for g in gus]
    sgs, gvs, vs = [], [], []
    for k in range(th['r_e']):
        if th['eye_class'] == 'random': sgs.append(rng.choice([-1.0, 1.0], P['O'].shape[1])); gvs.append(rng.standard_normal(wd)); vs.append(pat(gvs[-1]))
        else: sgs.append(struct_sg(k))
    es = [rng.standard_normal(n) for _ in range(th['r_e'])]
    if th['eye_class'] == 'random' and th['cos'] >= 0:       # overlap: rotate the first eye coefficient vector relative to the first neural one (yield_projector.cell_data)
        gu, gv = gus[0], gvs[0]; uh = gu / np.linalg.norm(gu); vp = gv - (gv @ uh) * uh; vp = vp / np.linalg.norm(vp)
        gc = np.linalg.norm(gv) * (th['cos'] * uh + math.sqrt(1 - th['cos'] ** 2) * vp); gvs[0] = gc; vs[0] = pat(gc)
    if th['eye_class'] == 'own':
        BO, ref = own_leak_map(P); vs = []
        for sg in sgs:
            v = (P['Osd'] * sg) @ BO; vs.append(v * ref / np.linalg.norm(v))
    t = np.arange(n); ms = [np.ones(n)] + [math.sqrt(2.0) * np.cos(math.pi * (k) * t / n) for k in range(1, th['r_n'])]
    Nn = torch.zeros(n, P['F'].shape[1], **T64)
    for u, m in zip(us, ms): Nn += G(np.outer(y * m, u))
    return dict(n=n, y=y, es=es, sgs=np.array(sgs), vs=G(np.array(vs)), Nn=Nn, us=us, vraw=vs, F=P['F'][:n], Fg=P['Fg'][:n], O=P['O'][:n], Osd=P['Osd'], th=th)


def assemble(D, a, b, rho, kappa):
    th = D['th']; re_, rn = th['r_e'], th['r_n']; y = D['y']
    Fn = D['Fg'] + (a / math.sqrt(rn)) * D['Nn'] if a > 0 else D['Fg']
    S = np.stack([rho * y + math.sqrt(1 - rho ** 2) * e for e in D['es']], 1)                        # (n, r_e) latent eye signals
    if b > 0:
        Eye = (b / math.sqrt(re_)) * (G(S) @ D['vs']); Fo = D['Fg'] + Eye; Fm = Fn + Eye
        Om = D['O'] + kappa * (S @ (D['Osd'][None] * D['sgs'])) / math.sqrt(re_)
    else: Fo = D['Fg']; Fm = Fn; Om = D['O']
    return Fn, Fo, Fm, Om


# ---------------------------------------------------------------- read-out
def ycols(y): n = len(y); return np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
def readout(F, y, transform=None): return D2.pe_gain(F, ycols(y), LS, transform=transform)
def yld(sc): return {str(L): float(sc[L][:, 0].mean() - sc[L][:, 1:].mean()) for L in LS}


def a4_transform(Oc, ycol):
    """A4 = yield_projector.a4_transform (copied, same algebra): per fit LS of F[T] on [1, Oc, y_j][T] (ridge 1e-8), subtract Oc B_O from all rows."""
    X = torch.cat([torch.ones(len(Oc), 1, **T64), Oc, ycol[:, None]], 1); p = Oc.shape[1]
    def t(F, M):
        m = M.double()[:, :, None]; Xm = X[None] * m
        Bc = torch.linalg.solve(Xm.transpose(1, 2) @ Xm + 1e-8 * torch.eye(X.shape[1], **T64), Xm.transpose(1, 2) @ (F[None] * m))
        return F[None] - Oc[None] @ Bc[:, 1:1 + p]
    return t


def a4_yield(Fm, y, Om):
    Ys = ycols(y); Oc = G(Om); g = {str(L): [] for L in LS}
    for j in range(4):
        sc = D2.pe_gain(Fm, Ys[:, j:j + 1], LS, transform=a4_transform(Oc, G(Ys[:, j])))
        for L in LS: g[str(L)].append(float(sc[L][:, 0].mean()))
        torch.cuda.empty_cache()
    return {L: g[L][0] - float(np.mean(g[L][1:])) for L in g}


# ---------------------------------------------------------------- per-cell statistics and tokens
def cell_stats(Fm, y, Om, sc, pseed):
    """D1 statistics (58), per-PC channels (3, 64), tokens (n, 76). Functions of the mixed cell, y, the measured eye regressors and the joint read-out only."""
    n = len(y); X = Fm if torch.is_tensor(Fm) else G(Fm); X = (X - X.mean(0)) / (X.std(0, unbiased=False) + 1e-9)
    e, U = torch.linalg.eigh(X @ X.T); e = e.flip(-1).clamp_min(0); U = U.flip(-1); S = e.sqrt()                        # n x n Gram route (same singular values as the SVD)
    lam = S[:64] ** 2 / n; Z = U[:, :64] * math.sqrt(n)
    yc = G(y); yc = (yc - yc.mean()) / yc.std(unbiased=False)
    spec = torch.log(lam); fo = D2.folds(n); od = torch.as_tensor(fo % 2 == 1, device=DEV); ev = ~od
    c = Z.T @ yc / n; c1 = Z[od].T @ yc[od] / int(od.sum()); c2 = Z[ev].T @ yc[ev] / int(ev.sum())
    en = n * c ** 2; rep = n / 2 * c1 * c2
    E = TS.e_scope(y, Om)
    D = torch.stack([torch.ones(n, **T64), yc], 1); Dp = torch.linalg.pinv(D); Zp = Z - D @ (Dp @ Z); Ob = G(Om); Op = Ob - D @ (Dp @ Ob)
    Uo, So, _ = torch.linalg.svd(Op, full_matrices=False); r = int((So > 1e-8 * So[0]).sum()); Q = Uo[:, :r]
    top_sv = lambda C: torch.linalg.eigvalsh(C.transpose(-1, -2) @ C).flip(-1).clamp_min(0).sqrt()
    sv = top_sv(Q.T @ Zp); sv5 = torch.zeros(5, **T64); sv5[:min(5, len(sv))] = sv[:5]
    g = torch.Generator(device=DEV); g.manual_seed(int(pseed)); perms = torch.argsort(torch.rand(P_PERM, n, generator=g, **T64), dim=1)
    top = top_sv(Q[perms].transpose(1, 2) @ Zp[None])[:, 0]; theta = float(torch.sort(top).values[math.ceil(0.95 * (P_PERM + 1)) - 1]); k = int((sv5 > theta).sum())
    s64 = sc[64]; fon = fo
    jt = yld(sc); fg = [float(s64[fon == b_, 0].mean()) for b_ in range(10)]; ff = [float(s64[fon == b_, 1:].mean()) for b_ in range(10)]; fl = [float(s64[:, j].mean()) for j in (1, 2, 3)]
    x1 = [jt['64'], jt['full'], *spec.reshape(8, 8).mean(1).tolist(), *en.reshape(8, 8).sum(1).tolist(), *rep.reshape(8, 8).sum(1).tolist(), E, *sv5.tolist(), theta, float(k), float(n), *fg, *ff, *fl]
    assert len(x1) == 58 and np.isfinite(x1).all()
    pcx = torch.stack([spec, rep, en]).cpu().numpy()
    Ob_c = (Ob - Ob.mean(0)) / (Ob.std(0, unbiased=False) + 1e-9); Uo2, _, _ = torch.linalg.svd(Ob_c, full_matrices=False); Os = Uo2[:, :8] * math.sqrt(n)
    tok = torch.cat([Z, Os, yc[:, None], G(s64[:, 0:1]), G(s64[:, 1:].mean(1, keepdims=True)), torch.arange(n, **T64)[:, None] / n], 1)
    assert tok.shape == (n, NTOK) and bool(torch.isfinite(tok).all())
    return np.array(x1), pcx, tok.cpu().numpy()


def meta_row(th, s, seed, cell, a, b, rho, kappa, f, has):
    return [FAMCODE[th['fam']], s, seed, cell, th['n_use'], th['cls'], rho, kappa, a, b, f, th['w0'], th['wd'], th['r_n'], th['r_e'], EYE_CLASS[th['eye_class']], th['cos'], MARG[th['marg']], float(has)]


class Unit:
    """Accumulates the cells of one shard and the read-out cache of the cells that share a feature matrix."""
    def __init__(self): self.rows = dict(X1=[], PCX=[], TOK=[], NUSE=[], Y=[], BL=[], META=[]); self.cache = {}; self.calls = 0

    def sc(self, key, F, y, transform=None):
        if key not in self.cache: self.cache[key] = readout(F, y, transform); self.calls += 1; torch.cuda.empty_cache()
        return self.cache[key]

    def add(self, D, th, s, seed, cell, a, b, rho, kappa, f, want_base, pseed):
        y = D['y']; Fn, Fo, Fm, Om = assemble(D, a, b, rho, kappa)
        kN = ('N', 0.0) if a == 0 else ('N', a); kO = ('N', 0.0) if b == 0 else ('O', b, rho, kappa)
        kJ = kN if b == 0 else (kO if a == 0 else ('J', a, b, rho, kappa))
        scN = self.sc(kN, Fn, y); scO = self.sc(kO, Fo, y); scJ = scN if kJ == kN else (scO if kJ == kO else self.sc(kJ, Fm, y))
        yN, yO, yJ = yld(scN), yld(scO), yld(scJ)
        x1, pcx, tok = cell_stats(Fm, y, Om, scJ, pseed)
        bl = [np.nan] * 4
        if want_base:
            a1 = yld(readout(Fm, y, PE.eogreg(Om))); self.calls += 1; a4 = a4_yield(Fm, y, Om); self.calls += 4; bl = [a1['64'], a1['full'], a4['64'], a4['full']]
        tp = np.zeros((NMAX, NTOK), np.float16); tp[:len(y)] = tok.astype(np.float16)
        for k_, v in (('X1', x1), ('PCX', pcx), ('TOK', tp), ('NUSE', len(y)), ('Y', [yN['64'], yN['full'], yO['64'], yO['full'], yJ['64'], yJ['full']]), ('BL', bl),
                      ('META', meta_row(th, s, seed, cell, a, b, rho, kappa, f, want_base))): self.rows[k_].append(v)

    def save(self, path):
        arr = dict(X1=np.array(self.rows['X1'], np.float64), PCX=np.array(self.rows['PCX'], np.float32), TOK=np.array(self.rows['TOK'], np.float16), NUSE=np.array(self.rows['NUSE'], np.int32),
                   Y=np.array(self.rows['Y'], np.float64), BL=np.array(self.rows['BL'], np.float64), META=np.array(self.rows['META'], np.float64),
                   X1_names=np.array(X1_NAMES), META_names=np.array(META_NAMES), code=np.array(CODE_SHA256))
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f'tmp{os.getpid()}_{path.name}')
        with open(tmp, 'wb') as fh: np.savez(fh, **arr)
        os.replace(tmp, path)


# ---------------------------------------------------------------- units of work
def unit_train(kind, s, chunk, nper):
    path = CELLS / kind / f'{kind}_sub{s:03d}_c{chunk:02d}.npz'
    if path.exists(): return 0.0
    t1 = time.time(); U = Unit(); P = pb(s); fc = 100 if kind == 'train' else 101; rt = np.random.default_rng([fc, s, chunk, 0])
    for i in range(nper):
        th = draw_theta_train(rt); rd = np.random.default_rng([fc, s, chunk, 1 + i]); D = draw_data(P, th, rd); U.cache = {}      # every cell has its own y and plants: no read-out is shared between cells
        U.add(D, th, s, chunk, i, th['a'], th['b'], th['rho'], th['kappa'], float('nan'), False, (fc * 1000 + s) * 1000 + chunk * 40 + i)
        del D
    U.save(path); dt = time.time() - t1; log(f'{kind} sub {s} chunk {chunk}: {nper} cells, {U.calls} read-out calls, {dt:.0f}s'); return dt


def grid_cells():
    """The 14 cells of one (participant, seed, family): RHO-free null and f = 0, then RHO x f (f = 1 is ocular-only). F1 grid (A1/A4 scored) = RHO .25/.5 x f .25/.5/.75."""
    out = [('null', None, 0.0, 0.0, float('nan')), ('f0', None, A0_AMP, 0.0, 0.0)]
    for rho in STORED_RHOS:
        for f in (0.25, 0.5, 0.75, 1.0): out.append((f'f{f:g}', rho, A0_AMP * math.sqrt(1 - f), B0_AMP * math.sqrt(f), f))
    return out


def unit_family(kind, fam, s, seed, want_base=True):
    path = CELLS / kind / f'{fam}_sub{s:03d}_seed{seed}.npz'
    if path.exists(): return 0.0
    t1 = time.time(); U = Unit(); P = pb(s); th = fam_theta(fam)
    rng = np.random.default_rng(910000 + 100 * s + seed) if fam == 'R0' else np.random.default_rng([200 + FAMCODE[fam], s, seed])
    D = draw_data(P, th, rng)
    for ci, (name, rho, a, b, f) in enumerate(grid_cells()):
        r = 0.5 if rho is None else rho
        thc = dict(th); thc['rho'] = r; thc['a'] = a; thc['b'] = b; thc['f'] = f if f == f else -1.0
        base = want_base and rho in (0.25, 0.5) and name in ('f0.25', 'f0.5', 'f0.75')
        U.add(D, thc, s, seed, ci, a, b, r, KAPPA0, thc['f'], base, ((200 + FAMCODE[fam]) * 1000 + s) * 1000 + seed * 20 + ci)
    U.save(path); dt = time.time() - t1; log(f'{kind} {fam} sub {s} seed {seed}: 14 cells, {U.calls} read-out calls, {dt:.0f}s'); return dt


# ---------------------------------------------------------------- stages
def stage_stamp():
    pr = OUT / 'PREREG_pilot.json'; st = OUT / 'PREREG_STAMP.json'
    if st.exists(): log('stamp exists', st.read_text()[:300]); return
    jdump(st, dict(prereg_sha256=_hl.sha256(pr.read_bytes()).hexdigest(), stamped=time.strftime('%Y-%m-%d %H:%M:%S %Z'), unix=time.time(), td_cells_sha256=CODE_SHA256,
                   note='no pilot cell had been generated or scored when this stamp was written (results/td_pilot/cells absent)', cells_dir_exists=CELLS.exists()))
    log('stamped', pr)


def stage_gate(smoke):
    """G1: the stored planted_v2 recipe regenerated by draw_data / assemble reproduces the stored cells to 1e-4 (L 64 and full), regenerated y, u, v, sig equal pid_regen.load exactly,
    A4 equals the stored A4 (runner_T9R arms_rho0.5, where present)."""
    pairs = [(2, 0.5), (5, 0.25)] if smoke else [(2, 0.5), (5, 0.25), (11, 0.9), (14, 0.5), (17, 0.25), (23, 0.9)]       # stored RHO 0.25 / 0.9 files exist for odd participants only (first attempt used (8, 0.9): no stored file, see ATTEMPTS)
    res = dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, rows=[]); worst = 0.0; worst4 = 0.0; nA4 = 0
    for s, rho in pairs:
        St = TS.stored(rho, s); sd = 0; P = pb(s); th = fam_theta('R0'); D = draw_data(P, th, np.random.default_rng(910000 + 100 * s + sd)); L = PR.load(s, rho, sd)
        sig = rho * D['y'] + np.sqrt(1 - rho ** 2) * D['es'][0]
        assert np.array_equal(D['y'], L['y']) and np.array_equal(D['us'][0], L['u']) and np.array_equal(D['vraw'][0], L['v']) and np.array_equal(sig, L['sig']), (s, rho)
        p4 = ROOT / f'results/yield_projector/runner_T9R/arms_rho0.5/sub-{s:03d}.json'; A4 = {}
        if rho == 0.5 and p4.exists():
            for c in json.loads(p4.read_text())['cells']: A4[(c['seed'], float(c['f']))] = c['arms']['A4']['yield']
        for f in ('null', 0.25, 0.75):
            a, b = PR.amp(f); Fn, Fo, Fm, Om = assemble(D, a, b, rho, KAPPA0); Sc = TS.scell(St, sd, f); y = D['y']; got = {}
            got['mix_all'] = yld(readout(Fm, y))
            if f != 'null':
                got['neural_only'] = yld(readout(Fn, y)); got['ocular_only'] = yld(readout(Fo, y)); got['mix_eogreg'] = yld(readout(Fm, y, PE.eogreg(Om)))
            for key, val in got.items():
                for w in ('64', 'full'):
                    dd = abs(val[w] - Sc[key][w]); worst = max(worst, dd); res['rows'].append(dict(s=s, rho=rho, f=f, key=key, width=w, regenerated=val[w], stored=Sc[key][w], absdiff=dd))
            if f != 'null' and (sd, float(f)) in A4:
                a4 = a4_yield(Fm, y, Om)
                for w in ('64', 'full'):
                    dd = abs(a4[w] - A4[(sd, float(f))][w]); worst4 = max(worst4, dd); nA4 += 1; res['rows'].append(dict(s=s, rho=rho, f=f, key='A4', width=w, regenerated=a4[w], stored=A4[(sd, float(f))][w], absdiff=dd))
        log(f'gate sub {s} rho {rho}: worst {worst:.2e} worstA4 {worst4:.2e}')
    res.update(max_absdiff_cells=worst, max_absdiff_A4=worst4, n_A4_compared=nA4, pass_1e_4=bool(worst <= 1e-4 and worst4 <= 1e-4))
    jdump(OUT / ('gate_smoke.json' if smoke else 'gate.json'), res); log('GATE', {k: v for k, v in res.items() if k != 'rows'})
    if not res['pass_1e_4']: raise SystemExit('GATE FAIL (G1): stored planted_v2 cells are not reproduced to 1e-4')


def stage_jitter(smoke, subs_override=None, fname=None, seed=1):
    subs = subs_override or ([1, 3] if smoke else [1, 3, 6, 7, 9, 12, 13, 15]); rows = []; det = []
    for s in subs:
        P = pb(s); D = draw_data(P, fam_theta('R0'), np.random.default_rng(910000 + 100 * s + seed)); n = D['n']; h = n // 20; y = D['y']
        for f in (0.25, 0.5, 0.75):
            a, b = PR.amp(f); Fn, _, _, _ = assemble(D, a, 0.0, 0.5, KAPPA0); y0 = yld(readout(Fn, y))
            y1 = yld(readout(Fn[h:], y[h:])); y2 = yld(readout(Fn[:n - h], y[:n - h])); yr = yld(readout(torch.roll(Fn, h, 0), np.roll(y, h)))
            rows.append(dict(s=s, f=f, Y64=y0['64'], Yfull=y0['full'], drop_first_d64=y1['64'] - y0['64'], drop_last_d64=y2['64'] - y0['64'], drop_first_dfull=y1['full'] - y0['full'], drop_last_dfull=y2['full'] - y0['full'],
                             roll_d64=yr['64'] - y0['64'], roll_dfull=yr['full'] - y0['full']))
            if f == 0.5: det.append(max(abs(yld(readout(Fn, y))[w] - y0[w]) for w in ('64', 'full')))
        log(f'jitter sub {s} done')
    dd = np.array([[r['drop_first_d64'], r['drop_last_d64']] for r in rows]); ddf = np.array([[r['drop_first_dfull'], r['drop_last_dfull']] for r in rows]); dr = np.array([r['roll_d64'] for r in rows]); drf = np.array([r['roll_dfull'] for r in rows])
    fl = float(np.sqrt(np.mean(dd ** 2) / 2)); res = dict(NOT_CITABLE='until a blind bug review', code_sha256=CODE_SHA256, participants=subs, n_cells=len(rows), rule='floor = sqrt(mean(d^2)/2), d = Y(rows dropped) - Y, drop first / last n//20 rows, neural-only R0 cells, seed 1, f .25/.5/.75, L 64',
                                               ROWDROP_floor64=fl, ROWDROP_floorfull=float(np.sqrt(np.mean(ddf ** 2) / 2)), ROWDROP_mean_d=[float(dd[:, 0].mean()), float(dd[:, 1].mean())], ROWDROP_sd_d=[float(dd[:, 0].std(ddof=1)), float(dd[:, 1].std(ddof=1))],
                                               ROLL_floor64=float(np.sqrt(np.mean(dr ** 2) / 2)), ROLL_floorfull=float(np.sqrt(np.mean(drf ** 2) / 2)), ROLL_mean_d=float(dr.mean()), ROLL_max_abs_d=float(np.abs(dr).max()),
                                               repeat_identical_call_max_absdiff=float(max(det)), void_if_ROWDROP_floor64_gt=0.02, VOID_by_rowdrop_rule=bool(fl > 0.02), VOID_by_T0_roll_rule=bool(np.sqrt(np.mean(dr ** 2) / 2) > 0.02), rows=rows)
    jdump(OUT / (fname or ('jitter_smoke.json' if smoke else 'jitter.json')), res); log('JITTER', {k: v for k, v in res.items() if k != 'rows'})


def stage_gen(kinds, ntrain_chunks, nper, nval_chunks, smoke):
    V, TE, TR = (VAL[:1], TEST[:1], TRAIN[:2]) if smoke else (VAL, TEST, TRAIN)
    for kind in kinds:
        if kind == 'val':
            for s in V:
                for c in range(nval_chunks): unit_train('val', s, c, nper)
        elif kind == 'cal':
            for s in V:
                for sd in ((200,) if smoke else (200, 201)): unit_family('cal', 'R4', s, sd, want_base=False)
        elif kind == 'train':
            for c in range(ntrain_chunks):
                for s in TR: unit_train('train', s, c, nper)               # chunk-major: an early stop leaves every participant represented
        elif kind == 'test':
            for fam, seeds in ((('R0', (0,)), ('R3', (100,))) if smoke else (('R0', (0, 1)), ('R2', (100, 101)), ('R3', (100, 101)))):
                for sd in seeds:
                    for s in TE: unit_family('test', fam, s, sd, want_base=True)
        elif kind == 'r4test':
            for s in TE: unit_family('r4test', 'R4', s, 202, want_base=False)
        else: raise SystemExit('unknown kind ' + kind)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--kinds', default='val,cal,train,test,r4test'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-chunks', type=int, default=5); ap.add_argument('--val-chunks', type=int, default=4); ap.add_argument('--nper', type=int, default=25); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); ledger('start:' + a.stage + (':smoke' if a.smoke else ''))
    if a.smoke: globals()['CELLS'] = OUT / 'cells_smoke'
    try:
        if a.stage == 'stamp': stage_stamp()
        elif a.stage == 'gate': stage_gate(a.smoke)
        elif a.stage == 'jitter': stage_jitter(a.smoke)
        elif a.stage == 'jitter_test': stage_jitter(False, TEST, 'jitter_test_participants.json')   # DESCRIPTIVE extra (not the pre-registered void rule): same measure on the test participants, run after the models are fixed
        elif a.stage == 'jitter_t0': stage_jitter(False, list(range(1, 9)), 'jitter_t0_configuration.json', seed=0)   # DESCRIPTIVE: the exact participants and seed of the T0 screen (1-8, seed 0), to see which cells drive the roll floor
        elif a.stage == 'gen':
            if not (OUT / 'PREREG_STAMP.json').exists(): raise SystemExit('PREREG_STAMP.json missing: run --stage stamp before generating')
            gj = OUT / ('gate_smoke.json' if a.smoke else 'gate.json')
            if not gj.exists() or not json.loads(gj.read_text())['pass_1e_4']: raise SystemExit('gate json missing or failed')
            stage_gen(a.kinds.split(','), a.train_chunks, a.nper, a.val_chunks, a.smoke)
        else: raise SystemExit('unknown stage')
    finally: ledger('end:' + a.stage)


if __name__ == '__main__':
    main()
