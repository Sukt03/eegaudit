"""pilot_cmi_B.py: Pilot B (runner, 24 Nov 2026), ONE hash-pinned script.
Protocol: outputs/FINAL/PROTOCOL_cmi_B.md (constants read from its CONSTANTS_JSON block). Stages: stamp | s0 | q0 | q1 | post | summary.
E1 = ridge-CCA nuisance proxy Z (6 columns, label free) -> A1-type in-fold regression of Z out of the features -> yield Y_A1q -> Y_B = Y_A1q / (1 - r_hat2)^2.
Reads td_cells, td_screen, pid_regen, d2_pe_tasks, pe_eogreg (not the pilot A script). --smoke writes only to results/pilot_proximal_cmi/B_smoke.
"""
import argparse, hashlib, json, math, os, re, sys, time
T_START = time.time()
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "3")
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'experiments/ext'), str(ROOT)]
import td_cells as TC
import td_screen as TS
import pid_regen as PR
import d2_pe_tasks as D2
import pe_eogreg as PE

SCRIPT = Path(__file__).resolve(); SCRIPT_SHA = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
PROTO = ROOT / 'outputs/FINAL/PROTOCOL_cmi_B.md'; PTXT = PROTO.read_text()
CBLOCK = re.search(r'```json CONSTANTS_JSON\n(.*?)\n```', PTXT, re.S).group(1); C = json.loads(CBLOCK)
DEV = 'cuda'; T64 = dict(dtype=torch.float64, device=DEV); G = TS.G
OUT = ROOT / 'results/pilot_proximal_cmi/B'; SMOKE_OUT = ROOT / 'results/pilot_proximal_cmi/B_smoke'
PARTS = C['participants']; SEEDS = C['seeds_primary']; LW = ('64', 'full')
CELLS6 = [(r, f) for r in (0.25, 0.5) for f in (0.25, 0.5, 0.75)]
STORED_PATH = {0.5: ROOT / 'results/gpu_fill_20261123/planted_v2_rho0.5', 0.25: ROOT / 'results/gpu_fill_20261123/planted_v2_rho0.25'}
ADD11_025 = ROOT / 'results/yield_projector/planted_v2_rho0.25_add11'
log = lambda *m: print(f'[{time.time() - T_START:8.1f}s]', *m, flush=True)


def jdump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f'.tmp{os.getpid()}'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def jload(path): return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- E1 estimator
def isq(S):
    e, V = torch.linalg.eigh(S); return (V / e.clamp_min(1e-12).sqrt()) @ V.T


def cca_Z(Om, q=3, lam=None):
    """Label-free ridge CCA between PA (VEOG lags + blink flag, 24 cols) and PB (Fp1-Fp2, F7-F8, 46 cols) of the z-scored 70 columns; Z = [ZA wa, ZB wb] (2q columns)."""
    lam = C['lam_cca'] if lam is None else lam
    O = G(Om); n = len(O); Zs = (O - O.mean(0)) / (O.std(0, unbiased=False) + 1e-9)
    A = Zs[:, list(range(23)) + [69]]; B = Zs[:, list(range(23, 69))]
    Saa = A.T @ A / n + lam * torch.eye(A.shape[1], **T64); Sbb = B.T @ B / n + lam * torch.eye(B.shape[1], **T64); Sab = A.T @ B / n
    Ia, Ib = isq(Saa), isq(Sbb); U, S, Vh = torch.linalg.svd(Ia @ Sab @ Ib, full_matrices=False)
    wa = Ia @ U[:, :q]; wb = Ib @ Vh.T[:, :q]
    return torch.cat([A @ wa, B @ wb], 1)


def rhat2(y, Znp): return max(0.0, TS.e_scope(y, Znp))


def e1_cell(Fm, y, Z):
    """One read-out call: A1 transform with Z (y not in the regression) -> Y_A1q (both widths); r_hat2 (cross-fitted OLS); Y_B."""
    Znp = Z.cpu().numpy() if torch.is_tensor(Z) else np.asarray(Z, float)
    r2 = rhat2(y, Znp); a1q = TC.yld(TC.readout(Fm, y, PE.eogreg(Znp))); torch.cuda.empty_cache()
    return dict(r2=r2, a1q=a1q, YB={L: a1q[L] / (1 - r2) ** 2 for L in LW})


def gen(s, seed):
    P = TC.pb(s); D = TC.draw_data(P, TC.fam_theta('R0'), np.random.default_rng(910000 + 100 * s + seed)); return D, D['y']


def slat(D, y, rho): return rho * y + math.sqrt(1 - rho ** 2) * D['es'][0]


def stored_cell(s, seed, f):
    return TS.scell(TS.stored(0.5, s), seed, f)


# ---------------------------------------------------------------- comparator A4 and metrics
def load_a4(subs):
    """A4[rho][s][seed][f][L] from results/yield_projector/runner_T9R/arms_rho{rho}/sub-XXX.json."""
    A4 = {}
    for rho in (0.25, 0.5):
        A4[rho] = {}
        for s in subs:
            J = jload(ROOT / f'results/yield_projector/runner_T9R/arms_rho{rho}/sub-{s:03d}.json'); d = {}
            for c in J['cells']: d.setdefault(c['seed'], {})[float(c['f'])] = {L: c['arms']['A4']['yield'][L] for L in LW}
            A4[rho][s] = d
    return A4


def truth_N(s, seed, f): c = stored_cell(s, seed, f)['neural_only']; return {L: c[L] for L in LW}


def tmat(fn, subs, rho, L):
    """(n_part, 6) participant (mean of seeds) value of fn(s, seed, rho, f, L) over CELLS6 restricted to the given rho list order."""
    return np.array([[np.mean([fn(s, sd, r, f, L) for sd in SEEDS]) for (r, f) in CELLS6 if r in rho] for s in subs])


def metrics(err):
    """err (n_part, k cells) participant-level signed errors -> dict(cell cohort error, MAE over cells, participant-level MAE)."""
    cell = err.mean(0); return dict(cell_err=cell.tolist(), mae=float(np.abs(cell).mean()), pl_mae=float(np.abs(err).mean()))


def boot_ix(n):
    g = torch.Generator(device=DEV); g.manual_seed(C['boot_seed']); return torch.randint(0, n, (C['boot_B'], n), generator=g, device=DEV)


def boot_ratio(eb, ea, IX):
    """per resample MAE_b(B)/MAE_b(A4) over cells; returns point, UB97.5, UB98.75."""
    B_, A_ = torch.as_tensor(eb, **T64), torch.as_tensor(ea, **T64)
    mb = B_[IX].mean(1).abs().mean(1); ma = A_[IX].mean(1).abs().mean(1); r = mb / ma
    q = lambda p: float(torch.quantile(r, torch.tensor(p, **T64)))
    return dict(point=float(B_.mean(0).abs().mean() / A_.mean(0).abs().mean()), ub975=q(C['ub_q']), ub9875=q(C['ub_q_descriptive']))


def sd_ratio_ub(eb, ea, IX):
    B_, A_ = torch.as_tensor(eb, **T64), torch.as_tensor(ea, **T64)
    r = B_[IX].std(1) / A_[IX].std(1)            # (boot, cells)
    return dict(point=(B_.std(0) / A_.std(0)).tolist(), ub975=torch.quantile(r, torch.tensor(C['ub_q'], **T64), dim=0).tolist())


# ---------------------------------------------------------------- PREREG, ledger, budget
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_stamp(smoke):
    if smoke: return
    pr = OUT / 'PREREG.json'
    if not pr.exists(): raise SystemExit('PREREG.json missing: run --stage stamp first')
    if jload(pr)['script_sha256'] != SCRIPT_SHA: raise SystemExit('script sha256 differs from PREREG (bug after stamp = new hash, new PREREG): abort')


def ledger_sum_h(out):
    tot = 0.0
    for p in {out / 'ledger.jsonl', SMOKE_OUT / 'ledger.jsonl', SMOKE_OUT / 'ledger_diag.jsonl'}:
        if p.exists(): tot += sum(json.loads(l)['dur_s'] for l in p.read_text().splitlines() if l.strip()) / 3600
    return tot


def write_ledger(out, stage, hold, calls, smoke, extra=None):
    t1 = time.time(); dur = t1 - T_START; led = out / 'ledger.jsonl'; step = (len(led.read_text().splitlines()) + 1) if led.exists() else 1
    line = dict(step=step, stage=stage, hold=hold, t_start_unix=T_START, t_end_unix=t1, dur_s=dur, calls=calls, s_per_call=(dur / calls if calls else None), script_sha256=SCRIPT_SHA[:12], **(extra or {}))
    line['sum_dur_h_so_far'] = ledger_sum_h(out) + dur / 3600
    with open(led, 'a') as fh: fh.write(json.dumps(line) + '\n')


def budget_ok(out, calls, smoke):
    if smoke: return
    sp = jload(out / 's0' / 'gtime.json')['s_per_call']; proj = calls * 1.25 * sp / 3600; so = ledger_sum_h(out)
    log(f'budget: summed so far {so:.4f} h (main + smoke + diagnostic ledgers), projected {proj:.4f} h, cap {C["ledger_cap_h"]}')
    if so + proj > C['ledger_cap_h']: raise SystemExit(f'BUDGET: {so:.4f} + {proj:.4f} > {C["ledger_cap_h"]} h: step not started')


# ---------------------------------------------------------------- stage: stamp
def stage_stamp(hold):
    if OUT.exists() and any(OUT.iterdir()): raise SystemExit(f'stamp refuses: {OUT} is not empty: {[p.name for p in OUT.iterdir()]}')
    sec4 = PTXT[PTXT.index('## 4. Pre-registered predictions'):PTXT.index('## 5. Gates')]
    mods = {m.__name__ if hasattr(m, '__name__') else str(m): sha_file(m.__file__) for m in (TC, TS, PR, D2, PE)}
    pr = dict(NOT_CITABLE='until a blind bug review', protocol_path=str(PROTO), protocol_sha256=sha_file(PROTO), script_path=str(SCRIPT), script_sha256=SCRIPT_SHA, imported_module_sha256=mods,
              constants_sha256=hashlib.sha256(CBLOCK.encode()).hexdigest(), constants=C, bootstrap_seed=C['boot_seed'], stamped_clock=time.strftime('%Y-%m-%d %H:%M:%S %Z'), stamped_unix=time.time(),
              hold=f'cluster (protocol); this step ran on {hold}', python=sys.executable, scored_outputs_exist=False, prediction_section4_sha256=hashlib.sha256(sec4.encode()).hexdigest(),
              note='smoke runs (results/pilot_proximal_cmi/B_smoke) and one diagnostic JSON-structure read happened before the stamp; they produced no scored output and no stage reads them. Per-step holds are in ledger.jsonl.')
    jdump(OUT / 'PREREG.json', pr); log('stamped', OUT / 'PREREG.json')


# ---------------------------------------------------------------- stage: s0
def regen_gate(smoke):
    pairs = [(2, 0.5, 'main'), (5, 0.25, 'main')] if smoke else [(2, 0.5, 'main'), (14, 0.5, 'main'), (5, 0.25, 'main'), (17, 0.25, 'main'), (2, 0.25, 'add11')]
    fs = ('null', 0.25) if smoke else ('null', 0.25, 0.75); rows = []; worst = 0.0; ident = True
    for s, rho, src in pairs:
        St = TS.stored(rho, s) if src == 'main' else jload(ADD11_025 / f'sub-{s:03d}.json'); assert St is not None, (s, rho)
        D, y = gen(s, 0); L_ = PR.load(s, rho, 0); sig = slat(D, y, rho)
        same = bool(np.array_equal(y, L_['y']) and np.array_equal(D['us'][0], L_['u']) and np.array_equal(D['vraw'][0], L_['v']) and np.array_equal(sig, L_['sig'])); ident &= same
        for f in fs:
            a, b = PR.amp(f); Fn, Fo, Fm, Om = TC.assemble(D, a, b, rho, TC.KAPPA0); Sc = TS.scell(St, 0, f); got = {'mix_all': TC.yld(TC.readout(Fm, y))}
            if f != 'null':
                got['neural_only'] = TC.yld(TC.readout(Fn, y)); got['ocular_only'] = TC.yld(TC.readout(Fo, y)); got['mix_eogreg'] = TC.yld(TC.readout(Fm, y, PE.eogreg(Om)))
            for k, v in got.items():
                for w in LW:
                    d = abs(v[w] - Sc[k][w]); worst = max(worst, d); rows.append(dict(s=s, rho=rho, src=src, f=f, key=k, width=w, regenerated=v[w], stored=Sc[k][w], absdiff=d))
        log(f'G-REGEN sub {s} rho {rho} ({src}): worst {worst:.2e} exact-identical-pieces {same}')
    return dict(max_absdiff=worst, pieces_exact=ident, passed=bool(worst <= 1e-4 and ident), rows=rows)


def ident_gate(smoke):
    res = []
    for s in ([1, 2] if smoke else [1, 2, 3, 4, 5, 6]):
        D, y = gen(s, 0); a, b = PR.amp(0.5); Fn, Fo, Fm, Om = TC.assemble(D, a, b, 0.5, TC.KAPPA0); Z = cca_Z(Om)
        X = Fm - Fm.mean(0); X = X / (X.std(0, unbiased=False) + 1e-9); U, S_, _ = torch.linalg.svd(X, full_matrices=False); X8 = U[:, :8] * S_[:8]; X8 = (X8 - X8.mean(0)) / X8.std(0, unbiased=False)
        yt = G(y); n = len(yt); one = torch.ones(n, 1, **T64)
        def ols_res(Xc, M): Dm = torch.cat([one, Xc], 1) if Xc is not None else one; return M - Dm @ torch.linalg.lstsq(Dm, M).solution
        sse = lambda r: float((r ** 2).sum()); sst = sse(yt - yt.mean())
        rz = ols_res(Z, yt[:, None])[:, 0]; rj = ols_res(torch.cat([Z, X8], 1), yt[:, None])[:, 0]
        n1 = 1 - sse(rj) / sse(rz); Xp = ols_res(Z, X8); yp = ols_res(Z, yt[:, None])[:, 0]
        r2_yp = 1 - sse(ols_res(Xp, yp[:, None])[:, 0]) / sse(yp - yp.mean()); r2_yXp = 1 - sse(ols_res(Xp, yt[:, None])[:, 0]) / sst; r2_yZ = 1 - sse(rz) / sst; n3 = r2_yXp / (1 - r2_yZ)
        res.append(dict(s=s, joint_partial=n1, r2_yperp_Xperp=r2_yp, r2_y_Xperp_over=n3, maxdiff=max(abs(n1 - r2_yp), abs(n1 - n3), abs(r2_yp - n3))))
    return dict(rows=res, max_diff=max(r['maxdiff'] for r in res), passed=bool(max(r['maxdiff'] for r in res) <= 1e-9))


def a4_gate(subs):
    A4 = load_a4(subs); out = dict(passed=None)
    fn = lambda s, sd, r, f, L: A4[r][s][sd][f][L] - truth_N(s, sd, f)[L]
    m = {}
    for L in LW:
        E = np.array([[np.mean([fn(s, sd, r, f, L) for sd in SEEDS]) for (r, f) in CELLS6] for s in subs]); mm = metrics(E); m[L] = mm
        m[L]['mae_rho25'] = float(np.abs(np.array(mm['cell_err'][:3])).mean()); m[L]['mae_rho5'] = float(np.abs(np.array(mm['cell_err'][3:])).mean())
    fz = C['a4_frozen_L64']; ff = C['a4_frozen_full']
    chk = dict(mae6=abs(m['64']['mae'] - fz['mae6']), mae_rho25=abs(m['64']['mae_rho25'] - fz['mae_rho25']), mae_rho5=abs(m['64']['mae_rho5'] - fz['mae_rho5']), participant_mae=abs(m['64']['pl_mae'] - fz['participant_mae']), full_mae6=abs(m['full']['mae'] - ff['mae6']))
    sav = jload(ROOT / 'results/pilot_proximal_cmi/A/a4_stored.json'); dsav = 0.0
    for r in ('0.25', '0.5'):
        for s in subs:
            for sd in SEEDS:
                for f in (0.25, 0.5, 0.75):
                    for L in LW: dsav = max(dsav, abs(sav[r][str(s)][str(sd)][f'f{f:g}'][L] - A4[float(r)][s][sd][f][L]))
    return dict(measured=m, abs_dev_from_frozen=chk, max_dev_vs_A_a4_stored=dsav, passed=bool(max(chk.values()) <= 5.1e-7 and dsav <= 1e-9))


def curve_pred(s, seed, ratio_c):
    """Stored NEURAL curve (runner_T9R base_rho0.5) at x = a^2/a0^2 = (null 0, f.75 .25, f.5 .5, f.25 .75, f0 1), piecewise linear; returns {f: Y_B prediction} for factor c = (1 - r^2)^2 per f."""
    J = jload(ROOT / f'results/yield_projector/runner_T9R/base_rho0.5/sub-{s:03d}.json'); pts = {}
    for c in J['cells']:
        if c['seed'] == seed: pts[0.0 if c['f'] == 'null' else 1.0 - float(c['f'])] = c['arms']['NEURAL']['yield']['64']
    xs = sorted(pts); ys = [pts[x] for x in xs]
    return {f: float(np.interp(ratio_c[f] * (1 - f), xs, ys) / ratio_c[f]) for f in (0.25, 0.5, 0.75)}


def stage_s0(smoke, out, hold):
    subs = [1, 2] if smoke else PARTS; gate = dict(attempts=[])
    gp = out / 's0' / 'gates.json'
    if gp.exists() and not smoke:
        prev = jload(gp)
        if prev.get('final_status') == 'VOID': raise SystemExit('G gates VOID already')
        gate['attempts'] = prev['attempts']
    if len(gate['attempts']) >= 2: raise SystemExit('gate attempts exhausted: VOID')
    t1 = time.time(); g1 = regen_gate(smoke); g2 = ident_gate(smoke); g3 = a4_gate(PARTS)
    # G-TIME: E1 on 2 participants, one cell (rho .5, f .5, seed 0): one untimed warm-up call, then two timed
    ts = []; D0, y0 = gen(1, 0); a, b = PR.amp(0.5); _, _, Fm0, Om0 = TC.assemble(D0, a, b, 0.5, TC.KAPPA0); e1_cell(Fm0, y0, cca_Z(Om0)); torch.cuda.synchronize()
    for s in (1, 2):
        D, y = gen(s, 0); _, _, Fm, Om = TC.assemble(D, a, b, 0.5, TC.KAPPA0); torch.cuda.synchronize(); t2 = time.time(); e1_cell(Fm, y, cca_Z(Om)); torch.cuda.synchronize(); ts.append(time.time() - t2)
    gt = dict(seconds=ts, s_per_call=float(np.mean(ts)), note='one untimed warm-up call first; each timed call = CCA + e_scope + one 4-column pe_gain read-out')
    jdump(out / 's0' / 'gtime.json', gt)
    att = dict(attempt=len(gate['attempts']) + 1, G_REGEN=dict(passed=g1['passed'], max_absdiff=g1['max_absdiff'], pieces_exact=g1['pieces_exact']), G_IDENT=dict(passed=g2['passed'], max_diff=g2['max_diff']),
               G_A4_FROZEN=dict(passed=g3['passed'], abs_dev_from_frozen=g3['abs_dev_from_frozen'], max_dev_vs_A_a4_stored=g3['max_dev_vs_A_a4_stored']), seconds=time.time() - t1)
    gate['attempts'].append(att); ok = g1['passed'] and g2['passed'] and g3['passed']
    gate['final_status'] = 'PASS' if ok else ('FAIL_rerun_allowed' if len(gate['attempts']) < 2 else 'VOID')
    jdump(out / 's0' / 'gates_detail.json', dict(regen=g1, ident=g2, a4=g3)); jdump(gp, gate); log('GATES', json.dumps(att)); log('G-TIME', gt)
    if not ok: raise SystemExit('S0 gate failed: ' + gate['final_status'])
    # ---- B0 prints (stored arrays; r_hat2 and R2(s | Z) of the oracle-Z and the CCA-Z; E vs flags_E.json of pilot A; no read-out call here)
    fl = jload(ROOT / 'results/pilot_proximal_cmi/A/flags_E.json'); Eg = fl['E']; rows = {}; dE = 0.0
    for s in subs:
        for rho in (0.5, 0.25):
            for sd in SEEDS:
                D, y = gen(s, sd); a, b = PR.amp(0.5); _, _, _, Om = TC.assemble(D, a, b, rho, TC.KAPPA0); sl = slat(D, y, rho); Z = cca_Z(Om); Znp = Z.cpu().numpy()
                E = TS.e_scope(y, Om); ref = Eg.get(str(rho), {}).get(str(s)); dE = max(dE, abs(E - ref[sd])) if ref else dE
                rg = np.random.default_rng(777 + 10 * s + sd)       # descriptive nulls of r_hat2: shuffled labels, and a noise Z of the same width
                rows[f'{s}_{rho}_{sd}'] = dict(E=E, r2_s_given_Z=TS.e_scope(sl, Znp), r2hat_cca=rhat2(y, Znp), r2hat_oracle=rhat2(y, sl[:, None]),
                                               r2hat_shuffled_y=TS.e_scope(y[rg.permutation(len(y))], Znp), r2hat_noiseZ=TS.e_scope(y, rg.standard_normal(Znp.shape)))
    pred = {}
    for rho in (0.5, 0.25):
        for nm, key in (('theory', None), ('cca', 'r2hat_cca'), ('oracle', 'r2hat_oracle')):
            bias = {f: [] for f in (0.25, 0.5, 0.75)}
            for s in subs:
                per = {f: [] for f in bias}
                for sd in SEEDS:
                    r2 = rho ** 2 if key is None else rows[f'{s}_{rho}_{sd}'][key]; c = (1 - r2) ** 2; p = curve_pred(s, sd, {f: c for f in bias})
                    for f in bias: per[f].append(p[f] - stored_cell(s, sd, f)['neural_only']['64'])
                for f in bias: bias[f].append(float(np.mean(per[f])))
            pred[f'rho{rho}_{nm}'] = dict(cohort_bias={str(f): float(np.mean(v)) for f, v in bias.items()}, participant_sd={str(f): float(np.std(v, ddof=1)) for f, v in bias.items()}, mae3=float(np.mean([abs(np.mean(v)) for v in bias.values()])))
    r2s = {str(rho): [float(np.mean([rows[f'{s}_{rho}_{sd}']['r2_s_given_Z'] for sd in SEEDS])) for s in subs] for rho in (0.5, 0.25)}
    summ = {k: dict(median=float(np.median(v)), min=float(np.min(v))) for k, v in r2s.items()}
    b0 = dict(NOT_CITABLE='until a blind bug review', predictions=pred, r2_s_given_Z_summary=summ, r2hat_median={str(rho): {k: float(np.median([rows[f'{s}_{rho}_{sd}'][k] for s in subs for sd in SEEDS])) for k in ('r2hat_cca', 'r2hat_oracle', 'r2hat_shuffled_y', 'r2hat_noiseZ')} for rho in (0.5, 0.25)},
              E_vs_flags_E_max_absdiff=dE, rows=rows)
    jdump(out / 's0' / 'b0.json', b0); log('B0 predictions', json.dumps(pred)); log('B0 R2(s|Z)', json.dumps(summ)); log('B0 r2hat medians', json.dumps(b0['r2hat_median'])); log('E vs flags_E max absdiff', dE)
    return 0


# ---------------------------------------------------------------- stage: q0
def q0_participant(s, out):
    p = out / 'q0' / f'sub-{s:03d}.json'
    if p.exists(): return 0
    rows = []; calls = 0; st = TS.stored(0.5, s)
    for sd in SEEDS:
        D, y = gen(s, sd); tn = {L: TS.scell(st, sd, 'null')['mix_all'][L] for L in LW}
        a, b = PR.amp('null'); _, _, Fm, Om = TC.assemble(D, a, b, 0.5, TC.KAPPA0); r = e1_cell(Fm, y, cca_Z(Om)); calls += 1
        rows.append(dict(cell='null', rho=None, seed=sd, E=TS.e_scope(y, Om), truth_null=tn, **r))
        for rho in (0.5, 0.25):
            a, b = PR.amp(1.0); _, _, Fm, Om = TC.assemble(D, a, b, rho, TC.KAPPA0); sl = slat(D, y, rho); Z = cca_Z(Om); r = e1_cell(Fm, y, Z); calls += 1
            rows.append(dict(cell=f'oc{rho}', rho=rho, seed=sd, E=TS.e_scope(y, Om), r2_s_given_Z=TS.e_scope(sl, Z.cpu().numpy()), truth_null=tn, **r))
            if rho == 0.5:
                r = e1_cell(Fm, y, sl[:, None]); calls += 1; rows.append(dict(cell='oracle0.5', rho=rho, seed=sd, truth_null=tn, **r))
    jdump(p, dict(NOT_CITABLE='until a blind bug review', participant=s, rows=rows)); return calls


def q0_eval(subs, out):
    R = {s: jload(out / 'q0' / f'sub-{s:03d}.json')['rows'] for s in subs}; res = {}
    get = lambda cell: [r for s in subs for r in R[s] if r['cell'] == cell]
    for cell in ('null', 'oc0.5', 'oc0.25', 'oracle0.5'):
        rr = get(cell); res[cell] = {L: dict(cohort_mean_YB_minus_truth_null=float(np.mean([r['YB'][L] - r['truth_null'][L] for r in rr])), cohort_mean_A1q_minus_truth_null=float(np.mean([r['a1q'][L] - r['truth_null'][L] for r in rr])),
                                           sd=float(np.std([r['YB'][L] - r['truth_null'][L] for r in rr], ddof=1)), n=len(rr)) for L in LW}
        res[cell]['r2hat_mean'] = float(np.mean([r['r2'] for r in rr]))
    r2 = {}
    for cell in ('oc0.5', 'oc0.25'):
        v = [r['r2_s_given_Z'] for s in subs for r in R[s] if r['cell'] == cell and r['seed'] == 0]; r2[cell] = dict(median=float(np.median(v)), min=float(np.min(v)), values=v)
    m = lambda c: abs(res[c]['64']['cohort_mean_YB_minus_truth_null'])
    cl = dict(i_r2s=bool(all(r2[c]['median'] >= C['q0_r2s_median'] and r2[c]['min'] >= C['q0_r2s_min'] for c in r2)), ii_oc5=bool(m('oc0.5') <= C['q0_ocular']), ii_oc25=bool(m('oc0.25') <= C['q0_ocular']),
              ii_null=bool(m('null') <= C['q0_null']), iii_oracle=bool(m('oracle0.5') <= C['oracle_gate']))
    return dict(NOT_CITABLE='until a blind bug review', cells=res, r2_s_given_Z=r2, clauses=cl, passed=bool(all(cl.values())))


def stage_q0(smoke, out, hold):
    subs = [1, 2] if smoke else PARTS
    if not smoke:
        g = jload(out / 's0' / 'gates.json')
        if g['final_status'] != 'PASS': raise SystemExit('S0 not passed')
    budget_ok(out, 8 * len(subs), smoke); calls = 0
    for s in subs: calls += q0_participant(s, out); log(f'q0 sub {s} done')
    res = q0_eval(subs, out); jdump(out / 'q0_result.json', res); log('Q0', json.dumps({k: res[k] for k in ('clauses', 'passed')}), json.dumps({c: res['cells'][c]['64'] for c in res['cells']}))
    return calls


# ---------------------------------------------------------------- stage: q1
def q1_participant(s, rho, out):
    p = out / 'q1' / f'rho{rho}' / f'sub-{s:03d}.json'
    if p.exists(): return 0
    rows = []; calls = 0
    for sd in SEEDS:
        D, y = gen(s, sd); a, b = PR.amp(0.5); _, _, _, Om = TC.assemble(D, a, b, rho, TC.KAPPA0); Z = cca_Z(Om); E = TS.e_scope(y, Om); sl = slat(D, y, rho)
        for f in (0.25, 0.5, 0.75):
            a, b = PR.amp(f); Fn, Fo, Fm, Om_f = TC.assemble(D, a, b, rho, TC.KAPPA0); assert np.array_equal(Om_f, Om)
            r = e1_cell(Fm, y, Z); calls += 1
            Sr = TS.stored(rho, s); mix = TS.scell(Sr, sd, f)['mix_all'] if Sr is not None else None
            rows.append(dict(rho=rho, f=f, seed=sd, E=E, flagE=bool(E > C['e_thr']), flagR1=bool(r['r2'] > C['r2_cap_flag']), truth=truth_N(s, sd, f), mix_all=mix, **r))
    jdump(p, dict(NOT_CITABLE='until a blind bug review', participant=s, rho=rho, rows=rows)); return calls


def q1_collect(subs, rhos, out):
    """-> dict with (n_part, k) arrays for variants pure / itt, truth, A4, I_c; cells ordered as CELLS6 restricted to rhos."""
    A4 = load_a4(subs); cells = [(r, f) for (r, f) in CELLS6 if r in rhos]; R = {}
    for rho in rhos:
        for s in subs:
            for r in jload(out / 'q1' / f'rho{rho}' / f'sub-{s:03d}.json')['rows']: R[(rho, s, r['seed'], r['f'])] = r
    arr = lambda fn: np.array([[np.mean([fn(R[(r, s, sd, f)], r, s, sd, f) for sd in SEEDS]) for (r, f) in cells] for s in subs])
    D = {}
    for L in LW:
        D[L] = dict(pure=arr(lambda x, r, s, sd, f: x['YB'][L]), itt=arr(lambda x, r, s, sd, f: A4[r][s][sd][f][L] if x['flagE'] else x['YB'][L]), truth=arr(lambda x, r, s, sd, f: x['truth'][L]),
                    a4=arr(lambda x, r, s, sd, f: A4[r][s][sd][f][L]), a1q=arr(lambda x, r, s, sd, f: x['a1q'][L]), ic=arr(lambda x, r, s, sd, f: x['a1q'][L] / (1 - x['r2'])))
    D['r2'] = arr(lambda x, r, s, sd, f: x['r2']); D['flagE'] = arr(lambda x, r, s, sd, f: float(x['flagE'])); D['flagR1'] = arr(lambda x, r, s, sd, f: float(x['flagR1']))
    D['mix'] = arr(lambda x, r, s, sd, f: x['mix_all']['64'] if x['mix_all'] is not None else float('nan'))
    D['cells'] = cells; return D


def q1_eval(subs, rhos, out, IX):
    D = q1_collect(subs, rhos, out); res = dict(cells=[list(c) for c in D['cells']]); k = len(D['cells'])
    for L in LW:
        d = D[L]; res[L] = {}
        for var in ('pure', 'itt'):
            eb = d[var] - d['truth']; ea = d['a4'] - d['truth']; mb = metrics(eb); ma = metrics(ea)
            res[L][var] = dict(B=mb, A4=ma, ratio=boot_ratio(eb, ea, IX), sd_ratio=sd_ratio_ub(eb, ea, IX))
        res[L]['descriptive'] = dict(ic_over_YN=(d['ic'].mean(0) / d['truth'].mean(0)).tolist(), one_minus_rho2=[1 - c[0] ** 2 for c in D['cells']], ic_over_mix_all=(d['ic'].mean(0) / np.nanmean(D['mix'], 0)).tolist() if L == '64' else None)
    res['r2hat_cohort_mean_per_cell'] = D['r2'].mean(0).tolist(); res['flagE_rate'] = D['flagE'].mean(0).tolist(); res['flagR1_rate'] = D['flagR1'].mean(0).tolist(); return res


def rho5_clauses(ev):
    out = {}
    for var in ('pure', 'itt'):
        b = ev['64'][var]['B']; out[var] = dict(c_rho5_mae=b['mae'], c_ok=bool(b['mae'] <= C['bar_rho5']), d_max_cell_err=float(np.max(np.abs(b['cell_err']))), d_ok=bool(np.max(np.abs(b['cell_err'])) <= C['cell_cap']))
    return out


def stage_q1(smoke, out, hold):
    subs = [1, 2] if smoke else PARTS
    if not smoke:
        if not jload(out / 'q0_result.json')['passed']: raise SystemExit('Q0 failed: DEAD, q1 not run')
    budget_ok(out, 6 * len(subs) * 2, smoke); calls = 0; IX = boot_ix(len(subs))
    for s in subs: calls += q1_participant(s, 0.5, out); log(f'q1 rho .5 sub {s} done')
    e5 = q1_eval(subs, (0.5,), out, IX); r5 = rho5_clauses(e5); jdump(out / 'q1_rho0.5_result.json', dict(NOT_CITABLE='until a blind bug review', eval=e5, clauses=r5)); log('Q1 RHO .5 clauses c,d', json.dumps(r5)); log('RHO .5 cell errors', json.dumps({v: e5['64'][v]['B']['cell_err'] for v in ('pure', 'itt')}))
    stop = not all(r5[v]['c_ok'] and r5[v]['d_ok'] for v in ('pure', 'itt'))
    if stop and not smoke:
        jdump(out / 'q1_result.json', dict(NOT_CITABLE='until a blind bug review', early_stop_after_rho5=True, rho5_clauses=r5, core_passed=False, reason='clause c or d failed at RHO .5 (conservative reading: failing in either the pure or the ITT variant)')); log('EARLY STOP after RHO .5: Q1 core FAILED'); return calls
    for s in subs: calls += q1_participant(s, 0.25, out); log(f'q1 rho .25 sub {s} done')
    e = q1_eval(subs, (0.25, 0.5), out, IX); cl = {}
    for var in ('pure', 'itt'):
        b = e['64'][var]['B']; rt = e['64'][var]['ratio']; rr = rho5_clauses(e5)[var]
        cl[var] = dict(a_mae6=b['mae'], a_ok=bool(b['mae'] <= C['bar_q1']), b_ub975=rt['ub975'], b_ok=bool(rt['ub975'] < 1), c_ok=rr['c_ok'], c_val=rr['c_rho5_mae'], d_ok=rr['d_ok'], d_val=rr['d_max_cell_err'],
                       d_all_cells_max=float(np.max(np.abs(b['cell_err']))), secondary_pl_ok=bool(b['pl_mae'] <= C['pl_limit']), pl_val=b['pl_mae'],
                       full_mae_ok=bool(e['full'][var]['B']['mae'] <= C['full_mae']), full_ub_ok=bool(e['full'][var]['ratio']['ub975'] < C['full_ub']), full_val=e['full'][var]['B']['mae'], full_ub=e['full'][var]['ratio']['ub975'])
        cl[var]['core_passed'] = bool(cl[var]['a_ok'] and cl[var]['b_ok'] and cl[var]['c_ok'] and (np.max(np.abs(b['cell_err'])) <= C['cell_cap']))
        cl[var]['secondary_passed'] = bool(cl[var]['secondary_pl_ok'] and cl[var]['full_mae_ok'] and cl[var]['full_ub_ok'])
    core = bool(all(cl[v]['core_passed'] for v in cl)); sec = bool(all(cl[v]['secondary_passed'] for v in cl))
    jdump(out / 'q1_result.json', dict(NOT_CITABLE='until a blind bug review', eval=e, clauses=cl, core_passed=core, secondary_passed=sec, early_stop_after_rho5=False,
                                       reading='conservative: a clause counts only if it holds in BOTH the pure Y_B variant and the ITT variant (flagged participant-seeds carry A4)')); log('Q1', json.dumps(cl)); return calls


# ---------------------------------------------------------------- stage: post (only if Q1 core passed)
def stage_post(smoke, out, hold):
    q1 = jload(out / 'q1_result.json')
    if not q1['core_passed'] and not smoke: raise SystemExit('Q1 core not passed: post items UNREACHED')
    sp = ROOT / 'results/td_pilot/cells'; subs = C['test8'] if not smoke else C['test8'][:1]; res = dict(NOT_CITABLE='until a blind bug review', families={}); calls = 0
    budget_ok(out, 360 + 276, smoke); g8 = torch.Generator(device=DEV); g8.manual_seed(C['test8_boot_seed']); IX = torch.randint(0, len(subs), (C['boot_B'], len(subs)), generator=g8, device=DEV)
    num = {}; den = {}
    for fam, seeds, kind in (('R2', (100, 101), 'test'), ('R3', (100, 101), 'test'), ('R4', (202,), 'r4test')):
        EB = np.zeros((len(subs), 6)); EA = np.zeros((len(subs), 6)); regen_max = 0.0
        for i, s in enumerate(subs):
            eb = []; ea = []
            for sd in seeds:
                Z_ = np.load(sp / kind / f'{fam}_sub{s:03d}_seed{sd}.npz'); Y = Z_['Y']; BL = Z_['BL']; M = Z_['META']; P = TC.pb(s); th = TC.fam_theta(fam)
                D = TC.draw_data(P, th, np.random.default_rng([200 + TC.FAMCODE[fam], s, sd])); y = D['y']; ev = {}
                for ci in range(len(M)):
                    rho, f = M[ci, 6], M[ci, 10]
                    if rho not in (0.25, 0.5) or f not in (0.25, 0.5, 0.75): continue
                    a, b = PR.amp(float(f)); Fn, Fo, Fm, Om = TC.assemble(D, a, b, float(rho), TC.KAPPA0)
                    if rho == 0.5 and f == 0.5:
                        j = TC.yld(TC.readout(Fm, y)); calls += 1; regen_max = max(regen_max, abs(j['64'] - Y[ci, 4]), abs(j['full'] - Y[ci, 5]))
                    r = e1_cell(Fm, y, cca_Z(Om)); calls += 1
                    a4 = float(BL[ci, 2]) if np.isfinite(BL[ci, 2]) else TC.a4_yield(Fm, y, Om)['64']; calls += 0 if np.isfinite(BL[ci, 2]) else 4
                    ev[(float(rho), float(f))] = (r['YB']['64'] - float(Y[ci, 0]), a4 - float(Y[ci, 0]))
                eb.append([ev[c][0] for c in CELLS6]); ea.append([ev[c][1] for c in CELLS6])
            EB[i] = np.mean(eb, 0); EA[i] = np.mean(ea, 0)
        mb, ma = metrics(EB), metrics(EA); res['families'][fam] = dict(mae6_B=mb['mae'], mae6_A4=ma['mae'], regen_max_absdiff=regen_max, ok=bool(mb['mae'] <= C['test8_family_factor'] * ma['mae'] and mb['mae'] <= C['test8_family_abs'] and regen_max <= 1e-4)); num[fam] = EB; den[fam] = EA
    nb = sum(torch.as_tensor(num[f], **T64)[IX].mean(1).abs().mean(1) for f in num); da = sum(torch.as_tensor(den[f], **T64)[IX].mean(1).abs().mean(1) for f in den)
    ub = float(torch.quantile(nb / da, torch.tensor(C['ub_q'], **T64))); res['pooled_ub975'] = ub; res['pooled_ok'] = bool(ub < C['test8_pooled_ub']); res['test8_passed'] = bool(all(v['ok'] for v in res['families'].values()) and res['pooled_ok'])
    # q = 5 variant on the six primary cells of all participants (descriptive flag rate)
    flags = []
    for s in (subs if smoke else PARTS):
        for rho in (0.25, 0.5):
            q3 = {(r['seed'], r['f']): r['YB']['64'] for r in jload(out / 'q1' / f'rho{rho}' / f'sub-{s:03d}.json')['rows']}
            for sd in SEEDS:
                D, y = gen(s, sd); a, b = PR.amp(0.5); _, _, _, Om = TC.assemble(D, a, b, rho, TC.KAPPA0); Z5 = cca_Z(Om, q=5)
                for f in (0.25, 0.5, 0.75):
                    a, b = PR.amp(f); _, _, Fm, _ = TC.assemble(D, a, b, rho, TC.KAPPA0); r = e1_cell(Fm, y, Z5); calls += 1; flags.append(abs(r['YB']['64'] - q3[(sd, f)]) > C['q5_diff'])
    res['q5'] = dict(flag_rate=float(np.mean(flags)), n=len(flags), done=True); jdump(out / 'post_result.json', res); log('POST', json.dumps({k: v for k, v in res.items()})); return calls


# ---------------------------------------------------------------- stage: summary
def stage_summary(smoke, out, hold):
    g = jload(out / 's0' / 'gates.json'); S = dict(NOT_CITABLE='until a blind bug review', script_sha256=SCRIPT_SHA, gates=g['final_status'])
    if g['final_status'] != 'PASS': S.update(row=1, outcome='VOID')
    else:
        q0 = jload(out / 'q0_result.json'); S['q0'] = q0['clauses']
        if not q0['passed']: S.update(row=2, outcome='DEAD')
        else:
            q1 = jload(out / 'q1_result.json'); S['q1_core_passed'] = q1['core_passed']
            if not q1['core_passed']: S.update(row=3, outcome='DEAD', early_stop_after_rho5=q1['early_stop_after_rho5'], post='UNREACHED')
            else:
                pp = (out / 'post_result.json'); post = jload(pp) if pp.exists() else None
                ok = bool(q1['secondary_passed'] and post and post['test8_passed'] and post['q5'].get('done'))
                S.update(row=5 if ok else 4, outcome='PROCEED' if ok else 'REFINE', secondary_passed=q1['secondary_passed'], post=post)
    S['ledger_sum_h_incl_smoke_and_diag'] = ledger_sum_h(out); jdump(out / 'summary_B.json', S); log('SUMMARY', json.dumps(S)); return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--hold', default='node'); a = ap.parse_args()
    out = SMOKE_OUT if a.smoke else OUT
    if a.stage == 'stamp':
        stage_stamp(a.hold); (OUT / 'ledger.jsonl').write_text(''); write_ledger(OUT, 'stamp', a.hold, 0, False); return
    check_stamp(a.smoke); out.mkdir(parents=True, exist_ok=True)
    fn = dict(s0=stage_s0, q0=stage_q0, q1=stage_q1, post=stage_post, summary=stage_summary)[a.stage]
    calls = 0
    try: calls = fn(a.smoke, out, a.hold)
    finally: write_ledger(out, a.stage, a.hold, calls or 0, a.smoke, extra=dict(smoke=a.smoke))


if __name__ == '__main__':
    main()
