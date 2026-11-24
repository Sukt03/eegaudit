"""RF7b (24 Nov 2026) = rf7_mi_width.py with ONE change: the ridge-penalty grid can be extended DOWN.
  --lams old : M.LAMS (1e-4 ... 1e4, 17 points), the RF7 grid. Gate mode: every arm's per-participant bits must equal the stored rf7 arms.json to 1e-6.
  --lams new : [1e-7, 1e-6, 1e-5] + M.LAMS (20 points; 1e-4 is already the old minimum, so the requested 1e-4..1e-7 adds three new points). Joint (lambda, T) selection unchanged.
  The mi_card_v3 L=256 gate (1e-4) applies to the old grid only. Everything else (bases, folds, nulls, bootstrap, L90) is the rf7 code, text below unchanged.
---- original rf7 header ----
RUNNER RF7 (C4, 24 Nov 2026): MI D5 input-sufficiency width curve on EEGMMIDB.
Question: yield of the mi_card_v3 raw-voltage read-out as a function of the retained width L of the input, for the label-free PCA basis and for an
equal-width RANDOM orthonormal projection, L in {8, 16, 32, 64, 128, 256}. Read-out family, features, per-participant z-scoring, leave-participant-group-out
(10 groups, inner group (g+1) mod 10, joint (lambda, T) on the inner CE), exact-Newton multinomial logistic and bits scorer are IMPORTED UNCHANGED from
mi_card_v3.py (M.newton_logistic, M.run_design, M.select, M.bits, M.LAMS, M.TGRID). Only the basis differs: PCA = first L columns of the SAME 256-component
eigenbasis that mi_card_v3 fits on the outer-training rows (so L = 256 IS the stored primary raw row); random = QR of a Gaussian [4480, L] matrix (same 1/sqrt(4480)
column scale as the PCA basis, same centring on the outer-training mean, one basis per (width, draw), shared by all 10 groups).
Conditions per basis and width: eeg | noise input (v3 draw: seed 20261121 Gaussian, 1-40 Hz, per-participant per-channel SD) | shuffled labels (v3 permutation, seed 7).
Planted positive control: the rf2 class pattern (3 Hz, 5 channels C3 C4 CZ CP3 CP4, amp 0.5 x per-channel SD, seed 5) added to the v3 noise input; PCA and random curves.
PRE-DECLARED (before any run; the protocol numbers of the task and card C4 / T8):
  GATE  L = 256 PCA eeg / noise / shuffle cohort means equal the stored mi_card_v3 raw row (0.30163, -0.00026, -0.00002) to 1e-4 (and per participant, full cohort only).
  NULL  every cohort mean of noise-input and shuffled-label bits (every width, PCA and each random draw) within +-0.02 bits (also counts of participants within 0.02).
  L90   smallest grid L whose yield >= 0.9 x yield(L=256) (rubric / pe_width / l90_ubp definition); "defined" iff some L < 256 satisfies it (point estimate) AND the
        participant-bootstrap lower bound of the yield at that L is > 0; per-resample L90 distribution reported; L90 = 8 is a grid minimum (upper bound only).
  PARTNER the random projection counts as a partner at L90 iff its yield there is below the PCA yield with the 95% participant interval of the paired difference
        excluding 0 (also reported: non-overlap of the two marginal intervals).
  PLANT  planted yield at L = 256 > 0.5 bits (recovered); curves reported descriptively.
Usage: python rf7_mi_width.py --out DIR [--subjects 1,2,..] [--draws 3] [--widths 8,16,...]"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext'))
import mi_card_v3 as M
DEV = 'cuda'; CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
PLANT_CH = ['C3', 'C4', 'CZ', 'CP3', 'CP4']; WIDTHS = [8, 16, 32, 64, 128, 256]


def raw_feat(Z, sid, nP):                                 # = M.features()['raw'] + zpp (per-participant column z-score), as rf2
    N = len(Z); Xf = M.bandpass(Z, 1, 40)[:, :, M.W0:M.W1].reshape(N, 64, 70, 8).mean(-1).reshape(N, -1).double()
    for k in range(nP):
        r = sid == k; Xf[r] = (Xf[r] - Xf[r].mean(0)) / (Xf[r].std(0) + 1e-8)
    return Xf


def eig_bases(Xf, pca_rows, ncomp=256):
    """Per group: (mu, V256 [Fd, 256] descending-variance columns, scaled 1/sqrt(Fd)) = the basis of M.primal_fits."""
    out = {}; Fd = Xf.shape[1]
    for key, rows in pca_rows.items():
        A = Xf[rows]; mu = A.mean(0); A = A - mu; Cv = A.T @ A / len(A); w, V = torch.linalg.eigh(Cv)
        out[key] = (mu, V[:, -ncomp:].flip(1) / math.sqrt(Fd)); del A, Cv
    return out


def rand_basis(Fd, L, seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(Fd, L, generator=g, device=DEV, dtype=torch.float64)); return Q / math.sqrt(Fd)


def fits_basis(Xf, fits, ylab, lams, basis, lchunk=4):
    """= M.primal_fits with the basis given: basis[key] = (mu, V[Fd, L]). Returns eval logits per fit [len(lams), n_eval, 4], max|grad|."""
    out = [None] * len(fits); gm = 0.0
    for i, (tr, ev, key) in enumerate(fits):
        mu, V = basis[key]; Ptr = (Xf[tr] - mu) @ V; Pev = (Xf[ev] - mu) @ V
        Pe1 = torch.cat([Pev, torch.ones(len(ev), 1, dtype=Pev.dtype, device=DEV)], -1); Zs = []
        for l0 in range(0, len(lams), lchunk):
            lc = lams[l0:l0 + lchunk]; L = len(lc)
            W, g = M.newton_logistic(Ptr[None].expand(L, -1, -1).contiguous(), ylab[i][None].expand(L, -1).contiguous(),
                                     torch.ones(L, len(tr), dtype=torch.float64, device=DEV), torch.tensor(lc, dtype=torch.float64, device=DEV))
            gm = max(gm, g); Zs.append(torch.einsum('nd,ldk->lnk', Pe1, W))
        out[i] = torch.cat(Zs)
    return out, gm, {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=''); ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--widths', default=','.join(map(str, WIDTHS))); ap.add_argument('--lchunk', type=int, default=4); ap.add_argument('--no_plant', action='store_true'); ap.add_argument('--lams', choices=['old', 'new'], required=True)
    ap.add_argument('--resume_any', action='store_true', help='SMOKE ONLY: reuse arms.json written by an earlier revision of this script (the arm arithmetic is unchanged)')
    a = ap.parse_args(); LAMS_USE = M.LAMS if a.lams == 'old' else np.concatenate([[1e-7, 1e-6, 1e-5], M.LAMS]); assert np.all(np.diff(LAMS_USE) > 0); log('lams', a.lams, len(LAMS_USE), LAMS_USE.tolist())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); widths = [int(x) for x in a.widths.split(',')]; assert widths[-1] == 256
    cache = ROOT / 'results/ext/mi_card_cache'; D_, excl = {}, {}
    for s in ([int(x) for x in a.subjects.split(',')] if a.subjects else range(1, 110)):
        d, why = M.load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d['y'], minlength=4)
        if c.max() - c.min() > 2: excl[s] = 'class counts'; continue
        D_[s] = d
    P = sorted(D_); nP = len(P); ch = D_[P[0]]['ch']; assert all(D_[s]['ch'] == ch for s in P); log(f'{nP} participants, excluded {len(excl)}')
    X = torch.tensor(np.concatenate([D_[s]['X'] for s in P]), dtype=torch.float32, device=DEV); y = torch.tensor(np.concatenate([D_[s]['y'] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D_[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y); del D_
    g = torch.Generator(device=DEV); g.manual_seed(20261121); Xb = M.bandpass(X, 1, 40); Nz = M.bandpass(torch.randn(X.shape, generator=g, device=DEV), 1, 40)   # = v3 noise draw
    for k in range(nP):
        r = sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * Xb[r].std((0, 2), keepdim=True)
    del Xb; torch.cuda.empty_cache()
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7); ysh = y.clone()
    for k in range(nP):
        r = torch.where(sid == k)[0]; ysh[r] = y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
    G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]; fits, units, pca_rows = [], [], {}
    for gg in range(G):
        v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
        fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg)); units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
    yl = {c: [yy[f[0]] for f in fits] for c, yy in (('eeg', y), ('noise', y), ('shuffle', ysh))}
    # feature matrices (v3 raw row), once; the plant sits in the v3 noise input
    FX = {'eeg': raw_feat(X, sid, nP), 'noise': raw_feat(Nz, sid, nP)}; del X
    if not a.no_plant:
        ix = [ch.index(c) for c in PLANT_CH]; gp = torch.Generator(device=DEV); gp.manual_seed(5); pat = torch.randint(0, 2, (4, 5), generator=gp, device=DEV).float() * 2 - 1
        tt = torch.arange(Nz.shape[-1], device=DEV) / M.SF; wave = torch.sin(2 * math.pi * 3 * tt); sdc = Nz[:, ix].std((0, 2))
        for pname, amp in (('planted', 0.5), ('planted_weak', 0.1)):          # amp 0.5 = rf2 plant (saturates near 2 bits); 0.1 added to keep the control off the ceiling
            Zp = Nz.clone(); Zp[:, ix] += amp * sdc[None, :, None] * pat[y][:, :, None] * wave[None, None]; FX[pname] = raw_feat(Zp, sid, nP); del Zp
    del Nz; torch.cuda.empty_cache(); Fd = FX['eeg'].shape[1]; assert Fd == 4480
    EIG = {k: eig_bases(v, pca_rows) for k, v in FX.items()}; log('eigenbases done', {k: len(v) for k, v in EIG.items()})
    res = dict(code_sha256=CODE_SHA256, participants=P, n=nP, widths=widths, arms={}); dest = out / 'arms.json'
    if dest.exists():
        res = json.loads(dest.read_text()); assert res['participants'] == P and res['widths'] == widths and (a.resume_any or res['code_sha256'] == CODE_SHA256), 'stale arms.json (other code/cohort)'
        res['code_sha256'] = CODE_SHA256
    def run(name, data, cond, kind, L, draw=0):
        if name in res['arms']: return
        if kind == 'pca': basis = {k: (mu, V[:, :L].contiguous()) for k, (mu, V) in EIG[data].items()}
        else:
            Vr = rand_basis(Fd, L, 1000 * L + draw); basis = {k: (mu, Vr) for k, (mu, V) in EIG[data].items()}
        eng = lambda Xf, fi, ylab: fits_basis(Xf, fi, ylab, LAMS_USE, basis, a.lchunk)
        R, diag = M.run_design('rf7', units, fits, {cond: {'x': FX[data]}}, y, {cond: yl[cond]}, sid, nP, eng, [], [], None, None, None)
        b = [R[k]['x'][cond]['bits'] for k in range(nP)]; acc = [R[k]['x'][cond]['acc'] for k in range(nP)]
        res['arms'][name] = dict(bits=b, acc=acc, mean=float(np.mean(b)), data=data, cond=cond, kind=kind, L=L, draw=draw, edges=diag['edges'][f'{cond}|x'], max_grad=diag['grad'][f'{cond}|x'])
        res['runtime_s'] = round(time.time() - t0); dest.write_text(json.dumps(res)); log(name, f'mean bits {np.mean(b):+.4f}  edges {diag["edges"][f"{cond}|x"]}')
    # order: the L = 256 PCA row first (gate), then the rest
    res['lams'] = a.lams; res['lams_values'] = LAMS_USE.tolist()
    for cond in ('eeg', 'noise', 'shuffle'): run(f'pca|{cond}|256', 'noise' if cond == 'noise' else 'eeg', cond, 'pca', 256)
    sv = json.loads((ROOT / 'results/ext/mi_card_v3/primary_lpgo/summary_mi.json').read_text()); gate = None
    if sv['participants'] == P and a.lams == 'old':
        gate = {c: dict(this=res['arms'][f'pca|{c}|256']['mean'], saved=sv['rows']['raw'][cs]['mean']) for c, cs in (('eeg', 'eeg'), ('noise', 'noise'), ('shuffle', 'shuffle'))}
        for c in gate: gate[c]['abs_diff'] = abs(gate[c]['this'] - gate[c]['saved'])
        pp = []
        for k, s in enumerate(P):
            r = json.loads((ROOT / f'results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json').read_text())['raw']
            pp += [abs(res['arms']['pca|eeg|256']['bits'][k] - r['eeg']['bits']), abs(res['arms']['pca|noise|256']['bits'][k] - r['noise']['bits']), abs(res['arms']['pca|shuffle|256']['bits'][k] - r['shuffle']['bits'])]
        gate['max_abs_diff_per_participant'] = float(max(pp)); res['gate'] = gate; res['gate_pass_1e-4'] = bool(max(v['abs_diff'] for c, v in gate.items() if c in ('eeg', 'noise', 'shuffle')) < 1e-4 and max(pp) < 1e-4)
        log('GATE', json.dumps(gate)); assert res['gate_pass_1e-4'], 'GATE FAIL vs mi_card_v3'
    else: log('gate vs mi_card_v3 skipped (subset run or extended grid)')
    for L in widths:
        for cond in ('eeg', 'noise', 'shuffle'): run(f'pca|{cond}|{L}', 'noise' if cond == 'noise' else 'eeg', cond, 'pca', L)
    for d in range(a.draws):
        for L in widths:
            for cond in ('eeg', 'noise', 'shuffle'): run(f'rand|{cond}|{L}|d{d}', 'noise' if cond == 'noise' else 'eeg', cond, 'rand', L, d)
    if not a.no_plant:
        for pname in ('planted', 'planted_weak'):
            for L in widths: run(f'pca|{pname}|{L}', pname, 'eeg', 'pca', L); run(f'rand|{pname}|{L}|d0', pname, 'eeg', 'rand', L, 0)
    # ---------------- summary (participant bootstrap in torch, shared resamples)
    A = res['arms']; T = lambda v: torch.tensor(np.asarray(v, np.float64), device=DEV); gb = torch.Generator(device=DEV); gb.manual_seed(0); IX = torch.randint(0, nP, (10000, nP), generator=gb, device=DEV)
    ci = lambda m: [float(m.quantile(0.025)), float(m.quantile(0.975))]; bm = lambda v: T(v)[IX].mean(1)
    def curve(kind, cond):                                # per-participant bits [W, nP] (random: mean over draws)
        rows = []
        for L in widths:
            if kind == 'pca': rows.append(T(A[f'pca|{cond}|{L}']['bits']))
            else: rows.append(torch.stack([T(A[f'rand|{cond}|{L}|d{d}']['bits']) for d in range(a.draws)]).mean(0))
        return torch.stack(rows)
    S = dict(NOT_CITABLE='until a blind review', code_sha256=CODE_SHA256, n=nP, widths=widths, draws=a.draws, gate=gate, gate_pass=res.get('gate_pass_1e-4'),
             stored_raw_row=dict(eeg=sv['rows']['raw']['eeg']['mean'], noise=sv['rows']['raw']['noise']['mean'], shuffle=sv['rows']['raw']['shuffle']['mean']),
             definition='yield = mean bits per trial over participants (log2 p(true) - log2 train prior); CI = 95% participant bootstrap, 10000 resamples, GPU, seed 0 (shared resamples)')
    cur = {k: curve(k, 'eeg') for k in ('pca', 'rand')}; boot = {k: cur[k][:, IX].mean(2) for k in cur}              # [W, B]
    for kind in ('pca', 'rand'):
        m = cur[kind].mean(1); full = m[-1]
        S[kind] = dict(yield_mean=m.tolist(), yield_ci95=[ci(boot[kind][w]) for w in range(len(widths))], share_of_L256=(m / full).tolist(),
                       share_of_L256_ci95=[ci(boot[kind][w] / boot[kind][-1]) for w in range(len(widths))], n_participants_positive=[int((cur[kind][w] > 0).sum()) for w in range(len(widths))])
    S['rand']['per_draw_yield'] = {f'd{d}': [float(T(A[f'rand|eeg|{L}|d{d}']['bits']).mean()) for L in widths] for d in range(a.draws)}
    # L90 (point and per-resample)
    def l90(mean_curve, full):
        ok = [i for i, v in enumerate(mean_curve) if v >= 0.9 * full]; return widths[ok[0]] if ok else None
    for kind in ('pca', 'rand'):
        pt = l90(cur[kind].mean(1).tolist(), float(cur[kind].mean(1)[-1])); thr = (boot[kind] >= 0.9 * boot[kind][-1:]); first = thr.double().argmax(0)          # [B] index of first width reaching 90%
        dist = torch.tensor(widths, device=DEV, dtype=torch.float64)[first]
        S[kind]['L90'] = dict(point=pt, boot_median=float(dist.median()), boot_p2_5=float(dist.quantile(0.025)), boot_p97_5=float(dist.quantile(0.975)),
                              boot_prob_L90_le={str(L): float((dist <= L).double().mean()) for L in widths})
    i90 = widths.index(S['pca']['L90']['point']) if S['pca']['L90']['point'] else None
    if i90 is not None and widths[i90] < 256:
        lo = S['pca']['yield_ci95'][i90][0]; d = boot['pca'][i90] - boot['rand'][i90]
        S['L90_defined'] = bool(lo > 0); S['L90_at_grid_minimum'] = bool(widths[i90] == widths[0])
        S['partner_at_L90'] = dict(L=widths[i90], pca=S['pca']['yield_mean'][i90], pca_ci95=S['pca']['yield_ci95'][i90], rand=S['rand']['yield_mean'][i90], rand_ci95=S['rand']['yield_ci95'][i90],
                                   diff_mean=float(d.mean()), diff_ci95=ci(d), partner_by_paired_diff=bool(float(d.mean()) > 0 and ci(d)[0] > 0),
                                   partner_by_nonoverlap=bool(S['rand']['yield_ci95'][i90][1] < S['pca']['yield_ci95'][i90][0]))
    else: S['L90_defined'] = False; S['L90_at_grid_minimum'] = False; S['partner_at_L90'] = None; S['L90_note'] = 'no width below 256 reaches 0.9 x yield(256): L90 = 256 (no compression)'
    S['pca_minus_rand'] = [dict(L=L, d=float((cur['pca'][w] - cur['rand'][w]).mean()), ci95=ci(boot['pca'][w] - boot['rand'][w]), n_pos=int(((cur['pca'][w] - cur['rand'][w]) > 0).sum())) for w, L in enumerate(widths)]
    # nulls per width (PCA and every random draw)
    nul = {}
    for cond in ('noise', 'shuffle'):
        for kind in ('pca', 'rand'):
            for d in (range(a.draws) if kind == 'rand' else [None]):
                key = f'{cond}|{kind}' + (f'|d{d}' if d is not None else '')
                vals = [A[(f'pca|{cond}|{L}' if kind == 'pca' else f'rand|{cond}|{L}|d{d}')]['bits'] for L in widths]
                nul[key] = dict(mean=[float(np.mean(v)) for v in vals], n_abs_le_0p02=[int(sum(abs(x) <= 0.02 for x in v)) for v in vals], max_abs_participant=[float(np.max(np.abs(v))) for v in vals])
    S['nulls'] = nul; S['nulls_max_abs_mean'] = float(max(abs(x) for v in nul.values() for x in v['mean'])); S['nulls_ok_abs_le_0.02'] = bool(S['nulls_max_abs_mean'] <= 0.02)
    S['edges'] = {k: v['edges'] for k, v in A.items()}; S['max_grad'] = float(max(v['max_grad'] for v in A.values()))
    S['lams'] = a.lams; S['lams_values'] = LAMS_USE.tolist()
    st = json.loads((ROOT / 'results/runner_fixes_20261123/rf7_mi_width/arms.json').read_text()); rf7_gate_ok = None
    if st['participants'] == P:                           # stored rf7 (old grid, 105 participants): per-arm comparison for every arm that exists in both
        cmp_ = {n: dict(old_mean=st['arms'][n]['mean'], new_mean=A[n]['mean'], max_abs_participant_diff=float(np.max(np.abs(np.asarray(st['arms'][n]['bits']) - np.asarray(A[n]['bits'])))),
                        old_edges=st['arms'][n]['edges']) for n in A if n in st['arms']}
        S['vs_stored_rf7'] = cmp_
        if a.lams == 'old':
            S['gate_rf7_oldgrid_max_abs_participant_diff'] = float(max(v['max_abs_participant_diff'] for v in cmp_.values())); S['gate_rf7_oldgrid_n_arms'] = len(cmp_)
            rf7_gate_ok = bool(S['gate_rf7_oldgrid_max_abs_participant_diff'] < 1e-6 and len(cmp_) == len(A)); S['gate_rf7_oldgrid_pass_1e-6'] = rf7_gate_ok
    else: S['vs_stored_rf7'] = 'cohort differs (subset run), no comparison'
    if not a.no_plant:
        S['planted_control'] = {}
        for pname in ('planted', 'planted_weak'):
            pl = {}
            for kind in ('pca', 'rand'):
                v = [A[f'{kind}|{pname}|{L}' + ('|d0' if kind == 'rand' else '')] for L in widths]; pl[kind] = dict(yield_mean=[x['mean'] for x in v], yield_ci95=[ci(bm(x['bits'])) for x in v])
            pl['recovered_L256_gt_0.5'] = bool(pl['pca']['yield_mean'][-1] > 0.5); S['planted_control'][pname] = pl
    for k, s in enumerate(P):                                # per-participant file
        (out / f'sub-{s:03d}.json').write_text(json.dumps(dict(participant=s, widths=widths, **{kk: {n: A[n]['bits'][k] for n in A if n.startswith(kk + '|')} for kk in ('pca', 'rand')}), indent=1))
    S['runtime_s'] = round(time.time() - t0); (out / 'summary.json').write_text(json.dumps(S, indent=1)); log('SUMMARY', json.dumps({k: S[k] for k in ('pca', 'rand', 'L90_defined', 'partner_at_L90', 'nulls_max_abs_mean', 'gate_pass')}, indent=1)[:6000])
    if a.lams == 'old' and rf7_gate_ok is not None: log('RF7 OLD-GRID GATE (1e-6)', rf7_gate_ok, S['gate_rf7_oldgrid_max_abs_participant_diff']); assert rf7_gate_ok, 'GATE FAIL vs stored rf7 arms.json'


if __name__ == '__main__': main()
