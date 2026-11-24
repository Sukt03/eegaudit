"""Step 3 of IDEATOR_MIXED_ATTRIBUTION (, 23 Nov 2026). GPU torch.
Oracle atoms of the planted_v2 generator + matched-swap dose-response, on the stored planted_v2 cells (regenerated exactly by pid_regen.load).

GENERATOR (gf_planted_calib_v2.py; pid_regen.py): F' = F + a*y*u + b*sig*v, y ~ N(0,1), sig = RHO*y + sqrt(1-RHO^2)*e, a = a0*sqrt(1-f), b = b0*sqrt(f);
u, v = random patterns in PCs 20-51 of the participant's z-scored features, so all planted signal lies inside the top-64 PC subspace that the L=64 read-out sees.

ORACLE A (closed form, PRIMARY; declared before any run). In the top-64 PC coordinates of the z-scored real features the noise covariance is S0 = diag(lam)
(participant's own spectrum, treated as Gaussian), signal directions uz, vz (pid_regen). With y unit variance, m = a*uz + b*RHO*vz:
   neural only : I_N  = 0.5*log2(1 + a^2 uz' S0^-1 uz)
   ocular only : I_O  = -0.5*log2(1 - c'S^-1 c),  c = b*RHO*vz,  S = S0 + b^2 vz vz'
   both        : I_NO = -0.5*log2(1 - m'S^-1 m),  S = S0 + m m' + b^2 (1-RHO^2) vz vz'
(Gaussian MI = population held-out log-score gain of the optimal calibrated linear read-out at L=64, infinite training data.)
Atoms by MMI (Barrett 2015): R = min(I_N, I_O), U_N = I_N - R, U_O = I_O - R, S = I_NO - I_N - I_O + R. Atoms are formed on cell means (as step 1).
ORACLE B (secondary, estimate): the SAME read-out family (ridge LAM*n on the 64 PC coordinates, OLS calibration, Gaussian log score) on a synthetic Gaussian
sample from the generator, NBIG train + NBIG test rows per participant x seed (shows the ridge-shrinkage part of the gap; LAM=0 variant = code gate vs ORACLE A).
Shares: ocular-atom share PRIMARY = (U_O + R)/Y_NO (= Y_O/Y_NO under MMI; what the ocular source carries); ALT = U_O/Y_NO. Oracle share = same formula on oracle atoms.
Current share = 1 - kept, kept = mean(mix_eogreg)/mean(mix_permnull) (stored cells).

DOSE-RESPONSE (matched swap, fixed read-out). X(d) = X - d*b*sig*v + d*b*sig_swap*v on HELD-OUT rows only; the read-out and its calibration are the ones fit on
the unswapped training rows (exactly pe_gain; the ridge prediction is affine in the input, so pred(X + t*b*v) = pred(X) + t*(pred(b*v) - pred(0)): two extra
never-trained rows give every d exactly). sig_swap = sig of another trial in the same target class (4 quantile bins of y; random cyclic shift inside the bin,
no fixed points). PRIMARY curve: amplitude blend d in {0,.25,.5,.75,1} (ideator formula), yield = gain - mean of the 3 circular-shift floors, slope at d=0 =
linear coefficient of the exact quadratic fit. Secondary: unmatched swap (any trial), trial-fraction swap (fraction d of test rows fully swapped), swap of the neural source.
PLANTED slope (closed form, oracle read-out w = S^-1 m, rv = 1 - m'w, c = b*w'vz, eta = between-bin variance share of y, 4 Gaussian quantile bins):
   Y(d) = I_NO - (2 d c K + d^2 c^2 V)/(2 rv ln2);   K = Cov(y - yhat, sig - sig_swap) = (RHO - w'(a RHO uz + b vz)) - eta*(RHO - w'(a RHO uz + b RHO^2 vz)),
   V = 2 - 2 RHO^2 eta (matched);  unmatched: K = RHO - w'(a RHO uz + b vz), V = 2.   slope0 = -c K/(rv ln2).
GATES (fixed before the run): G_repro  max |Y(d=0) - stored mix_all L64| <= 1e-6 on every participant x seed x f;  G_oracle  max |ORACLE B(LAM=0) - ORACLE A| on I_NO <= 0.01 bit.
Usage: pid_oracle_swap.py OUT run RHO[,RHO] [smoke]   |   pid_oracle_swap.py OUT summ"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import pid_regen as G
D2 = G.D2; DEV = 'cuda'; LN2 = float(np.log(2)); t0 = time.time(); DS = [0.0, 0.25, 0.5, 0.75, 1.0]; FS = [0.0, 0.25, 0.5, 0.75]; NBIG = 200000
RES = f'{G.ROOT}/results/gpu_fill_20261123'
log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
T = lambda x: torch.as_tensor(x, dtype=torch.float64, device=DEV)
_nd = torch.distributions.Normal(T(0.0), T(1.0)); _q = _nd.icdf(T([0.25, 0.5, 0.75])); _pdf = torch.exp(_nd.log_prob(_q))
_ed = torch.cat([T([0.0]), _pdf, T([0.0])]); ETA = float((((_ed[:-1] - _ed[1:]) / 0.25) ** 2).mean())     # between-bin variance share of N(0,1), 4 quantile bins


def gmi(c, S): return float(-0.5 * torch.log2(1 - c @ torch.linalg.solve(S, c)))


def oracle_closed(lam, uz, vz, a, b, rho):
    S0 = torch.diag(lam); m = a * uz + b * rho * vz; SM = S0 + torch.outer(m, m) + b * b * (1 - rho ** 2) * torch.outer(vz, vz)
    o = dict(I_N=gmi(a * uz, S0 + a * a * torch.outer(uz, uz)), I_O=gmi(b * rho * vz, S0 + b * b * torch.outer(vz, vz)) if b > 0 else 0.0, I_NO=gmi(m, SM))
    w = torch.linalg.solve(SM, m); rv = float(1 - m @ w); cO = float(b * (w @ vz)); cN = float(a * (w @ uz))
    Ksig = rho - float(w @ (a * rho * uz + b * vz)); Ksw = ETA * (rho - float(w @ (a * rho * uz + b * rho * rho * vz)))
    def curve(c, K, V): return dict(slope0=-c * K / (rv * LN2), Y=[o['I_NO'] - (2 * d * c * K + d * d * c * c * V) / (2 * rv * LN2) for d in DS])
    o['swapO_matched'] = curve(cO, Ksig - Ksw, 2 - 2 * rho * rho * ETA); o['swapO_unmatched'] = curve(cO, Ksig, 2.0)
    o['swapN_matched'] = curve(cN, (1 - ETA) * rv, 2 - 2 * ETA); o['rv'] = rv
    return o


def cyc(idx, rng):      # random cyclic shift inside an index set: every trial gets a different trial of the same set
    p = rng.permutation(idx); src = dict(zip(p, np.roll(p, 1))); return np.array([src[i] for i in idx])


def partner(y, rng, matched):
    n = len(y); pr = np.arange(n)
    if not matched: return cyc(np.arange(n), rng)
    r = np.argsort(np.argsort(y)); bins = r * 4 // n
    for k in range(4): ix = np.where(bins == k)[0]; pr[ix] = cyc(ix, rng)
    return pr


def oracle_synth(lam, uz, vz, rho, seed):
    """ORACLE B: {f: {...}} from one synthetic Gaussian sample (common random numbers across f and arms)."""
    g = torch.Generator(device=DEV); g.manual_seed(int(seed)); rn = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.float64)
    sq = lam.sqrt(); E = [rn(NBIG, 64) * sq for _ in range(2)]; Y = [rn(NBIG) for _ in range(2)]; SG = [rho * y + np.sqrt(1 - rho ** 2) * rn(NBIG) for y in Y]
    rng = np.random.default_rng(int(seed)); yte = Y[1].cpu().numpy(); pm = torch.as_tensor(partner(yte, rng, True), device=DEV); pu = torch.as_tensor(partner(yte, rng, False), device=DEV)
    def fit(Xtr, Xte, lamr):
        mu = Xtr.mean(0); A = Xtr - mu; yc = Y[0] - Y[0].mean(); w = torch.linalg.solve(A.T @ A + lamr * NBIG * torch.eye(64, dtype=torch.float64, device=DEV), A.T @ yc)
        return A @ w, (Xte - mu) @ w, w
    sc = lambda ptr, pte: float(D2.gain_t(Y[0][None], ptr[None], Y[1][None], pte[None]).mean())
    out = {}
    for f in FS:
        a, b = G.amp(f); N = [a * torch.outer(y, uz) for y in Y]; O = [b * torch.outer(s, vz) for s in SG]; o = {}
        for nm, X in (('I_N', [E[i] + N[i] for i in (0, 1)]), ('I_O', [E[i] + O[i] for i in (0, 1)]), ('I_NO', [E[i] + N[i] + O[i] for i in (0, 1)])):
            ptr, pte, w = fit(X[0], X[1], D2.LAM); o[nm] = sc(ptr, pte)
            if nm == 'I_NO':
                gO = float(b * (w @ vz)); gN = float(a * (w @ uz))
                for key, z, p_, gg in (('swapO_matched', SG[1], pm, gO), ('swapO_unmatched', SG[1], pu, gO), ('swapN_matched', Y[1], pm, gN)):
                    Yd = [sc(ptr, pte + d * gg * (z[p_] - z)) for d in DS]; o[key] = dict(Y=Yd, slope0=float(np.polyfit(DS, Yd, 2)[1]))
                p0tr, p0te, _ = fit(X[0], X[1], 0.0); o['I_NO_lam0'] = sc(p0tr, p0te)
        out[f] = o
    return out


def swap_cell(Fm, y, sig, u, v, a, b, seed):
    """Real-cell dose-response with the frozen pe_gain read-out. Returns d=0 yield and Y(d) per swap kind."""
    n = len(y); rng = np.random.default_rng(int(seed)); pm = partner(y, rng, True); pu = partner(y, rng, False)
    Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
    Fa = T(np.vstack([Fm, np.zeros((1, Fm.shape[1])), b * v[None], a * u[None]])); Ya = T(np.vstack([Ys, np.zeros((3, 4))])); Y = Ya[:n]
    M, tests, trains = D2.nested_masks(n); Ma = torch.cat([M, torch.zeros((M.shape[0], 3), dtype=torch.bool, device=DEV)], 1)
    P = D2.ridge_batch(Fa, Ya, Ma, [64])[:, 0]                                  # (100, n+3, 4)
    gO = P[:, n + 1] - P[:, n]; gN = P[:, n + 2] - P[:, n]                         # (100, 4): prediction change per unit of the source
    kinds = [('swapO_matched', sig[pm] - sig, gO), ('swapO_unmatched', sig[pu] - sig, gO), ('swapN_matched', y[pm] - y, gN)]
    Tm = T(np.stack([d * dz for _, dz, _ in kinds for d in DS])); V = Tm.shape[0]     # (15, n)
    sc = torch.full((V, n, 4), float('nan'), dtype=torch.float64, device=DEV)
    for bb in range(10):
        tr = trains[bb]; te = torch.as_tensor(tests[bb], device=DEV); trg = torch.as_tensor(tr, device=DEV)
        inner = torch.zeros((len(tr), 4), dtype=torch.float64, device=DEV); pos = {int(r): i for i, r in enumerate(tr)}
        for c in range(9):
            k = 10 + 9 * bb + c; idx = torch.as_tensor(tests[k], device=DEV); inner[torch.as_tensor([pos[int(r)] for r in tests[k]], device=DEV)] = P[k][idx]
        gk = torch.stack([g[bb] for _, _, g in kinds]).repeat_interleave(len(DS), 0)                    # (15, 4)
        pred = P[bb][te][None] + Tm[:, te][:, :, None] * gk[:, None, :]                                 # (15, nte, 4)
        sc[:, te] = D2.gain_t(Y[trg].T[None], inner.T[None], Y[te].T[None], pred.transpose(1, 2)).transpose(1, 2)
    assert torch.isfinite(sc).all()
    yl = lambda s: float(s[:, 0].mean() - s[:, 1:].mean())
    out = dict(Y0=yl(sc[0])); order = torch.as_tensor(rng.permutation(n), device=DEV); rank = torch.empty(n, device=DEV); rank[order] = torch.arange(n, device=DEV, dtype=torch.float32)
    for i, (nm, _, _) in enumerate(kinds):
        s = sc[i * len(DS):(i + 1) * len(DS)]; Yd = [yl(x) for x in s]
        Yf = [yl(torch.where((rank < d * n)[:, None], s[-1], s[0])) for d in DS]                        # trial-fraction swap (nested random subsets)
        out[nm] = dict(Y=Yd, slope0=float(np.polyfit(DS, Yd, 2)[1]), Y_trialfrac=Yf, slope_trialfrac=float(np.polyfit(DS, Yf, 1)[0]))
    return out


def run(out, rhos, smoke):
    for rho in rhos:
        subs = [s for s in range(1, 24) if os.path.exists(f'{RES}/planted_v2_rho{rho}/sub-{s:03d}.json')]
        if smoke: subs = subs[:1]
        rows = []
        for s in subs:
            st = json.load(open(f'{RES}/planted_v2_rho{rho}/sub-{s:03d}.json'))['cells']
            for sd in (0, 1):
                d = G.load(s, float(rho), sd); lam, uz, vz = T(d['lam']), T(d['uz']), T(d['vz']); syn = oracle_synth(lam, uz, vz, float(rho), 5000000 + 1000 * s + sd)
                for f in FS:
                    a, b = G.amp(f); stc = [c for c in st if c['seed'] == sd and c['f'] == f][0]
                    row = dict(participant=s, seed=sd, f=f, n=d['n'], stored={k: stc[k]['64'] for k in ('neural_only', 'ocular_only', 'mix_all', 'mix_eogreg', 'mix_permnull')},
                               oracle_closed=oracle_closed(lam, uz, vz, a, b, float(rho)), oracle_synth=syn[f])
                    if f > 0:
                        Fm = d['F'] + a * np.outer(d['y'], d['u']) + b * np.outer(d['sig'], d['v'])
                        row['swap'] = swap_cell(Fm, d['y'], d['sig'], d['u'], d['v'], a, b, 6000000 + 1000 * s + sd); row['repro_abs_diff'] = abs(row['swap']['Y0'] - stc['mix_all']['64'])
                        if row['repro_abs_diff'] > 1e-6: raise SystemExit(f'G_repro FAIL rho{rho} sub{s} seed{sd} f{f}: {row["repro_abs_diff"]}')
                    rows.append(row)
                if smoke: break
            log(f'rho{rho} sub{s} done')
        json.dump(dict(NOT_CITABLE='until independent review', rho=rho, eta=ETA, NBIG=NBIG, LAM=D2.LAM, rows=rows), open(f'{out}/raw_rho{rho}{"_smoke" if smoke else ""}.json', 'w'))
        if smoke:
            for r in rows:
                oc, os_ = r['oracle_closed'], r['oracle_synth']
                print('f', r['f'], 'stored', {k: round(v, 3) for k, v in r['stored'].items()}, '\n  closed', {k: round(oc[k], 3) for k in ('I_N', 'I_O', 'I_NO')}, 'synth', {k: round(os_[k], 3) for k in ('I_N', 'I_O', 'I_NO', 'I_NO_lam0')},
                      '\n  slope closed/synth/real', {k: (round(oc[k]['slope0'], 4), round(os_[k]['slope0'], 4), round(r['swap'][k]['slope0'], 4) if 'swap' in r else None) for k in ('swapO_matched', 'swapO_unmatched', 'swapN_matched')},
                      '\n  Y(d) real', [round(x, 3) for x in r['swap']['swapO_matched']['Y']] if 'swap' in r else None, 'closed', [round(x, 3) for x in oc['swapO_matched']['Y']], 'repro', r.get('repro_abs_diff'), flush=True)


def boot(A, fn, B=10000):   # A (P, k) participant rows; fn maps (..., k) means -> (..., j)
    A = T(A); g = torch.Generator(device=DEV); g.manual_seed(0); bs = fn(A[torch.randint(len(A), (B, len(A)), generator=g, device=DEV)].mean(1))
    return fn(A.mean(0)).tolist(), bs.quantile(0.025, 0).tolist(), bs.quantile(0.975, 0).tolist()


def atoms(X):
    yn, yo, yno = X[..., 0], X[..., 1], X[..., 2]; r = torch.minimum(yn, yo); return torch.stack([r, yn - r, yo - r, yno - yn - yo + r], -1)


def summ(out):
    S = dict(NOT_CITABLE='until independent review', eta=ETA, cells={}); RHOS = ('0.25', '0.5', '0.9'); raw = {r: json.load(open(f'{out}/raw_rho{r}.json'))['rows'] for r in RHOS}
    sub_sets = {r: sorted({x['participant'] for x in raw[r]}) for r in RHOS}; common = sorted(set.intersection(*[set(v) for v in sub_sets.values()])); S['participants'] = {r: len(v) for r, v in sub_sets.items()}; S['common_participants'] = len(common)
    S['G_repro_max_abs_diff'] = max(x['repro_abs_diff'] for r in RHOS for x in raw[r] if 'swap' in x); S['G_repro_pass'] = bool(S['G_repro_max_abs_diff'] <= 1e-6)
    S['G_oracle_max_abs_diff'] = max(abs(np.mean([x['oracle_synth']['I_NO_lam0'] - x['oracle_closed']['I_NO'] for x in raw[r] if x['f'] == f])) for r in RHOS for f in FS)
    S['G_oracle_max_abs_diff_single_draw'] = max(abs(x['oracle_synth']['I_NO_lam0'] - x['oracle_closed']['I_NO']) for r in RHOS for x in raw[r]); S['G_oracle_pass'] = bool(S['G_oracle_max_abs_diff_single_draw'] <= 0.01)
    def pm(rows, subs, get): return np.array([np.mean([get(x) for x in rows if x['participant'] == s], 0) for s in subs])     # participant means over seeds
    for r in RHOS:
        for f in FS:
            rows = [x for x in raw[r] if x['f'] == f]; subs = sub_sets[r]; c = {}
            for tag, ss in (('hat', subs), ('hat_common', common)):
                A = pm(rows, ss, lambda x: [x['stored'][k] for k in ('neural_only', 'ocular_only', 'mix_all')]); pt, lo, hi = boot(A, atoms)
                c[tag] = dict(Y_N=float(A[:, 0].mean()), Y_O=float(A[:, 1].mean()), Y_NO=float(A[:, 2].mean()), **{k: dict(mean=pt[i], ci95=[lo[i], hi[i]]) for i, k in enumerate(('R', 'U_N', 'U_O', 'S'))})
                c[tag]['undefined_floor'] = bool(min(c[tag]['Y_N'], c[tag]['Y_O']) <= 0.02)
            for tag in ('oracle_closed', 'oracle_synth'):
                A = pm(rows, subs, lambda x: [x[tag][k] for k in ('I_N', 'I_O', 'I_NO')]); pt = atoms(T(A).mean(0)).tolist()
                c[tag] = dict(I_N=float(A[:, 0].mean()), I_O=float(A[:, 1].mean()), I_NO=float(A[:, 2].mean()), R=pt[0], U_N=pt[1], U_O=pt[2], S=pt[3])
            K = pm(rows, subs, lambda x: [x['stored']['mix_eogreg'], x['stored']['mix_permnull']]); c['current_share_1_minus_kept'] = float(1 - K[:, 0].mean() / K[:, 1].mean())
            h = c['hat']; c['atom_share_UO_plus_R'] = h['Y_O'] / h['Y_NO']; c['atom_share_UO_only'] = h['U_O']['mean'] / h['Y_NO']
            for tag in ('oracle_closed', 'oracle_synth'): c[tag]['share_UO_plus_R'] = c[tag]['I_O'] / c[tag]['I_NO']; c[tag]['share_UO_only'] = c[tag]['U_O'] / c[tag]['I_NO']
            if f > 0:
                c['dose'] = {}
                for kind in ('swapO_matched', 'swapO_unmatched', 'swapN_matched'):
                    Yd = pm(rows, subs, lambda x: x['swap'][kind]['Y']); sl = pm(rows, subs, lambda x: [x['swap'][kind]['slope0'], x['swap'][kind]['slope_trialfrac']]); pt, lo, hi = boot(sl, lambda z: z)
                    dd = dict(Y_hat=Yd.mean(0).tolist(), slope_hat=pt[0], slope_hat_ci95=[lo[0], hi[0]], slope_trialfrac_hat=pt[1], slope_trialfrac_ci95=[lo[1], hi[1]])
                    for tag in ('oracle_closed', 'oracle_synth'):
                        dd['Y_' + tag] = pm(rows, subs, lambda x: x[tag][kind]['Y']).mean(0).tolist(); dd['slope_' + tag] = float(pm(rows, subs, lambda x: [x[tag][kind]['slope0']]).mean())
                        dd['ratio_hat_over_' + tag] = dd['slope_hat'] / dd['slope_' + tag]; dd['within25_' + tag] = bool(abs(dd['ratio_hat_over_' + tag] - 1) <= 0.25)
                        dd['trialfrac_planted_' + tag] = dd['Y_' + tag][-1] - dd['Y_' + tag][0]; dd['trialfrac_ratio_' + tag] = dd['slope_trialfrac_hat'] / dd['trialfrac_planted_' + tag]; dd['trialfrac_within25_' + tag] = bool(abs(dd['trialfrac_ratio_' + tag] - 1) <= 0.25)
                    c['dose'][kind] = dd
            S['cells'][f'rho{r}|f={f}'] = c
    C = S['cells']; nine = [f'rho{r}|f={f}' for r in RHOS for f in FS if f > 0]; crit = {}
    for tag in ('oracle_closed', 'oracle_synth'):
        d = {k: dict(dU_O=C[k]['hat']['U_O']['mean'] - C[k][tag]['U_O'], dR=C[k]['hat']['R']['mean'] - C[k][tag]['R'], undefined_floor=C[k]['hat']['undefined_floor']) for k in nine}
        nu = sum(abs(v['dU_O']) <= 0.03 for v in d.values()); nr = sum(abs(v['dR']) <= 0.03 for v in d.values())
        nus = sum(abs(v['dU_O']) <= 0.03 and not v['undefined_floor'] for v in d.values()); nrs = sum(abs(v['dR']) <= 0.03 and not v['undefined_floor'] for v in d.values())
        crit['C1_' + tag] = dict(cells=d, n_UO_within_003=nu, n_R_within_003=nr, n_UO_within_003_defined_only=nus, n_R_within_003_defined_only=nrs, PASS_numeric=bool(nu >= 7 and nr >= 7), PASS=bool(nus >= 7 and nrs >= 7))
    c2 = {r: {a: C[f'rho{r}|f=0.0']['hat'][a] for a in ('U_O', 'R')} for r in RHOS}
    crit['C2'] = dict(cells=c2, PASS=bool(all(v['ci95'][0] <= 0 <= v['ci95'][1] for x in c2.values() for v in x.values())), note='U_O is identically 0 whenever Y_O < Y_N (R = min), so its CI is degenerate [0,0]')
    for tag in ('hat', 'hat_common'):
        m = {str(f): [C[f'rho{r}|f={f}'][tag]['R']['mean'] for r in RHOS] for f in FS}; ok = {f: bool(v[0] < v[1] < v[2]) for f, v in m.items()}
        crit['C3' + ('' if tag == 'hat' else '_common_participants')] = dict(R_hat_by_rho=m, monotone=ok, n_monotone=sum(ok.values()), PASS=bool(sum(ok.values()) >= 3))
    for kind in ('swapO_matched', 'swapO_unmatched'):
        for tag in ('oracle_closed', 'oracle_synth'):
            d = {k: dict(hat=C[k]['dose'][kind]['slope_hat'], planted=C[k]['dose'][kind]['slope_' + tag], ratio=C[k]['dose'][kind]['ratio_hat_over_' + tag]) for k in nine}; nn = sum(C[k]['dose'][kind]['within25_' + tag] for k in nine)
            dt = {k: dict(hat=C[k]['dose'][kind]['slope_trialfrac_hat'], planted=C[k]['dose'][kind]['trialfrac_planted_' + tag], ratio=C[k]['dose'][kind]['trialfrac_ratio_' + tag]) for k in nine}; nt = sum(C[k]['dose'][kind]['trialfrac_within25_' + tag] for k in nine)
            crit[f'C4_{kind}_{tag}'] = dict(cells=d, n_within25=nn, PASS=bool(nn >= 7), trialfrac_cells=dt, trialfrac_n_within25=nt, trialfrac_PASS=bool(nt >= 7))
    for sh in ('UO_plus_R', 'UO_only'):
        for tag in ('oracle_closed', 'oracle_synth'):
            d = {k: dict(atom=C[k]['atom_share_' + sh], current=C[k]['current_share_1_minus_kept'], oracle=C[k][tag]['share_' + sh]) for k in nine}
            for v in d.values(): v['err_atom'] = abs(v['atom'] - v['oracle']); v['err_current'] = abs(v['current'] - v['oracle']); v['atom_better'] = bool(v['err_atom'] < v['err_current'])
            nb = sum(v['atom_better'] for v in d.values()); ma = float(np.mean([v['err_atom'] for v in d.values()])); mc = float(np.mean([v['err_current'] for v in d.values()]))
            crit[f'C5_{sh}_{tag}'] = dict(cells=d, n_atom_better=nb, mean_err_atom=ma, mean_err_current=mc, PASS=bool(nb >= 7 and ma < mc))
    S['criteria'] = crit; S['primary'] = dict(C1='C1_oracle_closed', C2='C2', C3='C3', C4='C4_swapO_matched_oracle_closed', C5='C5_UO_plus_R_oracle_closed')
    json.dump(S, open(f'{out}/summary.json', 'w'), indent=1)
    print('gates', {k: S[k] for k in S if k.startswith('G_')})
    for k in [f'rho{r}|f={f}' for r in RHOS for f in FS]:
        c = C[k]; print(k, 'hat', {a: round(c['hat'][a]['mean'], 3) for a in ('R', 'U_N', 'U_O', 'S')}, 'floor' if c['hat']['undefined_floor'] else '', '| closed', {a: round(c['oracle_closed'][a], 3) for a in ('I_N', 'I_O', 'I_NO', 'R', 'U_N', 'U_O', 'S')},
                        '| synth', {a: round(c['oracle_synth'][a], 3) for a in ('I_N', 'I_O', 'I_NO', 'R', 'U_O')}, '| Y', round(c['hat']['Y_N'], 3), round(c['hat']['Y_O'], 3), round(c['hat']['Y_NO'], 3))
    for k, v in crit.items():
        print(k, {a: b for a, b in v.items() if a not in ('cells', 'trialfrac_cells')})
        if 'cells' in v and k[:2] in ('C4', 'C5'): print('   ', {a: {x: round(y, 3) if isinstance(y, float) else y for x, y in b.items()} for a, b in v['cells'].items()})


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    if sys.argv[2] == 'summ': summ(out)
    else: run(out, sys.argv[3].split(','), len(sys.argv) > 4 and sys.argv[4] == 'smoke')
