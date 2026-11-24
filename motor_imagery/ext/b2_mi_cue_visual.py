"""B2 (IDEATOR_LIFT_ABOVE_5_20261124 section 4): MI cue-locked visual control.
Question: does the cue-locked class yield of the registered raw-voltage MI read-out (EEGMMIDB, 105 participants, mi_card_v3) ride on occipital channels (visual cue reaction) or on sensorimotor channels?
Everything (groups, windows, thresholds, plant, reading rule, gates) is fixed in results/ext/b2_mi_cue_visual/PREREG.json (sha256 stamped before any scored output); this script reads its parameters from it.
Pipeline = mi_card_v3 imported unchanged (primal_fits / run_design / bandpass / eog_regress), scoring harness copied from rf2_mi_matched_rank.py (which reproduced mi_card_v3 to 1e-14).
Usage: python -I -u b2_mi_cue_visual.py --out DIR [--subjects 1,2] [--smoke]   (smoke: no real-data bits are printed, no verdict)"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext'))
import mi_card_v3 as M
DEV = 'cuda'; CODE = os.path.abspath(__file__); CODE_SHA256 = hashlib.sha256(open(CODE, 'rb').read()).hexdigest()
V3_SHA256 = hashlib.sha256(open(ROOT / 'experiments/ext/mi_card_v3.py', 'rb').read()).hexdigest()
t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
PRE_DIR = ROOT / 'results/ext/b2_mi_cue_visual'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=''); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--lchunk', type=int, default=4)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    PR = json.loads((PRE_DIR / 'PREREG.json').read_text()); ST = json.loads((PRE_DIR / 'PREREG_STAMP.json').read_text()); PM = PR['params']
    assert sha(PRE_DIR / 'PREREG.json') == ST['prereg_sha256'], 'PREREG changed after stamp'; assert ST['extra_file_sha256'][CODE] == CODE_SHA256, 'script changed after stamp'
    WIN = {k: tuple(v) for k, v in PM['windows_samples_in_epoch'].items()}; assert WIN['t05_40'] == (M.W0, M.W1)
    cache = ROOT / 'results/ext/mi_card_cache'; D_, excl = {}, {}
    for s in ([int(x) for x in a.subjects.split(',')] if a.subjects else range(1, 110)):
        d, why = M.load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d['y'], minlength=4)
        if c.max() - c.min() > 2: excl[s] = 'class counts'; continue
        D_[s] = d
    P = sorted(D_); nP = len(P); ch = D_[P[0]]['ch']; assert all(D_[s]['ch'] == ch for s in P); log(f'{nP} participants, excluded {sorted(excl)}')
    G = {k: [ch.index(c) for c in v] for k, v in PM['groups'].items() if isinstance(v, list)}; G['all'] = list(range(64))
    assert (len(G['occ']), len(G['sm']), len(G['fo'])) == (9, 21, 12) and len(set(G['occ']) | set(G['sm']) | set(G['fo'])) == 42
    X = torch.tensor(np.concatenate([D_[s]['X'] for s in P]), dtype=torch.float32, device=DEV); y = torch.tensor(np.concatenate([D_[s]['y'] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D_[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y); del D_
    g = torch.Generator(device=DEV); g.manual_seed(20261121); R0 = M.bandpass(X, 1, 40); Nz = M.bandpass(torch.randn(X.shape, generator=g, device=DEV), 1, 40)       # = v3 noise draw
    for k in range(nP):
        r = sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * R0[r].std((0, 2), keepdim=True)
    del X; Nz0 = M.bandpass(Nz, 1, 40); del Nz; torch.cuda.empty_cache()                 # v3 noise features = bandpass(Nz, 1, 40)
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7); ysh = y.clone()
    for k in range(nP):
        r = torch.where(sid == k)[0]; ysh[r] = y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
    Gn = min(10, nP); grp = torch.tensor([k % Gn for k in range(nP)], device=DEV)[sid]; fits, units, pca_rows = [], [], {}
    for gg in range(Gn):
        v = (gg + 1) % Gn; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
        fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg)); units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
    yl = {c: [yy[f[0]] for f in fits] for c, yy in (('eeg', y), ('shuffle', ysh))}
    eng = lambda Xf, fi, ylab: M.primal_fits(Xf, fi, ylab, M.LAMS, pca_rows, lchunk=a.lchunk)

    def feat(Z, idx, win):
        lo, hi = WIN[win]; Xf = Z[:, idx, lo:hi].reshape(N, len(idx), (hi - lo) // 8, 8).mean(-1).reshape(N, -1).double()
        for k in range(nP):
            r = sid == k; Xf[r] = (Xf[r] - Xf[r].mean(0)) / (Xf[r].std(0) + 1e-8)
        return Xf

    def score(Xf, cond):
        R, _ = M.run_design('b2', units, fits, {cond: {'x': Xf}}, y, {cond: yl['shuffle' if cond == 'shuffle' else 'eeg']}, sid, nP, eng, [], [], None, None, None)
        return [R[k]['x'][cond]['bits'] for k in range(nP)]
    res = dict(participants=P, arms={}); dest = out / 'arms.json'; tl0 = time.time(); capped = []
    sv = json.loads((ROOT / 'results/ext/mi_card_v3/primary_lpgo/summary_mi.json').read_text()); full = (sv['participants'] == P)
    STORED = {'t05_40': 'raw', 't05_15': 'raw_t05_15', 't15_40': 'raw_t15_40', 'eogreg|t05_40': 'raw_eogreg'}
    cap = PM['time_cap_s']; gate = {}

    def run(name, Z, grpname, win, cond):
        if capped: return
        if time.time() - tl0 > cap: capped.append(name); log('TIME CAP hit before', name); return
        if name in res['arms']: return
        ts = time.time(); b = score(feat(Z, G[grpname], win), cond)
        res['arms'][name] = dict(bits=b, mean=float(np.mean(b)), group=grpname, window=win, cond=cond, secs=round(time.time() - ts, 1)); dest.write_text(json.dumps(res))
        if not a.smoke or name.startswith('plant'): log(name, f"mean bits {np.mean(b):+.4f} ({time.time()-ts:.0f}s)")
        else: log(name, f"done ({time.time()-ts:.0f}s)")
    def gate_check(name, row, cond):
        if full: gate[name] = dict(this=res['arms'][name]['mean'], saved=sv['rows'][row][cond]['mean'], abs_diff=abs(res['arms'][name]['mean'] - sv['rows'][row][cond]['mean']))

    # ---- 1. gate arms: all-channel read-outs in the three post-cue windows (eeg, noise, shuffle) and the proxy-regressed 0.5-4.0 s read-out
    for w in ('t05_40', 't05_15', 't15_40'):
        for cond, Z in (('eeg', R0), ('noise', Nz0), ('shuffle', R0)):
            run(f'all|{w}|{cond}', Z, 'all', w, cond); gate_check(f'all|{w}|{cond}', STORED[w], cond)
    Rr, shr = M.eog_regress(R0, sid, nP, ch); Rr = M.bandpass(Rr, 1, 40); Rn, _ = M.eog_regress(Nz0, sid, nP, ch); Rn = M.bandpass(Rn, 1, 40)
    for cond, Z in (('eeg', Rr), ('noise', Rn), ('shuffle', Rr)):
        run(f'eogreg|t05_40|{cond}', Z, 'all', 't05_40', cond); gate_check(f'eogreg|t05_40|{cond}', STORED['eogreg|t05_40'], cond)
    del Rr, Rn; torch.cuda.empty_cache()
    res['gate'] = gate; res['gate_pass'] = bool(full and gate and all(v['abs_diff'] < PM['gate_tol'] for v in gate.values())); res['gate_n'] = len(gate)
    if full: log('GATE', json.dumps({k: round(v['abs_diff'], 12) for k, v in gate.items()})); assert res['gate_pass'], 'GATE FAIL vs mi_card_v3'
    else: log('gate skipped (subset run)')
    # ---- 2. decision arms: pre-cue leak check first, then the groups in the decision windows, then the descriptive 1.5-4.0 s window
    for cond, Z in (('eeg', R0), ('noise', Nz0)):
        for gname in ('all', 'occ', 'sm', 'fo'): run(f'{gname}|precue|{cond}', Z, gname, 'precue', cond)
    run('all|precue|shuffle', R0, 'all', 'precue', 'shuffle')
    for w in ('t05_40', 't05_15'):
        for gname in ('occ', 'sm', 'fo'):
            for cond, Z in (('eeg', R0), ('noise', Nz0)): run(f'{gname}|{w}|{cond}', Z, gname, w, cond)
    for gname in ('occ', 'sm', 'fo'): run(f'{gname}|t05_40|shuffle', R0, gname, 't05_40', 'shuffle')
    for gname in ('occ', 'sm', 'fo'):
        for cond, Z in (('eeg', R0), ('noise', Nz0)): run(f'{gname}|t15_40|{cond}', Z, gname, 't15_40', cond)
    # ---- 3. planted positive control: class-specific 3 Hz pattern from the cue onward in the noise input, in occ (resp. sm) channels only
    PL = PM['plant']; tt = torch.arange(Nz0.shape[-1], device=DEV) / M.SF - 1.0; wave = torch.sin(2 * math.pi * PL['frequency_hz'] * tt) * (tt >= 0).float()
    for pname in PL['plants']:
        gp = torch.Generator(device=DEV); gp.manual_seed(5); ix = G[pname]; pat = torch.randint(0, 2, (4, len(ix)), generator=gp, device=DEV).float() * 2 - 1
        Zp = Nz0.clone(); sdc = Nz0[:, ix].std((0, 2)); Zp[:, ix] += PL['amplitude_x_per_channel_SD'] * sdc[None, :, None] * pat[y][:, :, None] * wave[None, None]
        for w in PL['windows']:
            for gname in ('all', 'occ', 'sm', 'fo'): run(f'plant_{pname}|{gname}|{w}|eeg', Zp, gname, w, 'eeg')
        del Zp; torch.cuda.empty_cache()
    # ---- 4. count-matched random 9-channel draws (descriptive)
    rest = [i for i in range(64) if i not in set(G['occ']) | set(G['sm']) | set(G['fo'])]; assert len(rest) == 22; rng = np.random.default_rng(20261124)
    for i in range(2):
        G[f'rand9_{i}'] = sorted(int(c) for c in rng.choice(rest, 9, replace=False))
        for w in ('t05_40', 't05_15'): run(f'rand9_{i}|{w}|eeg', R0, f'rand9_{i}', w, 'eeg')
    res['runtime_s'] = round(time.time() - t0); res['capped_at'] = capped; res['rand9_channels'] = {k: [ch[i] for i in v] for k, v in G.items() if k.startswith('rand9')}
    dest.write_text(json.dumps(res))
    if a.smoke:
        pa = {k: v['mean'] for k, v in res['arms'].items() if k.startswith('plant') or '|noise' in k}
        (out / 'smoke_ok.json').write_text(json.dumps(dict(ok=True, n_arms=len(res['arms']), runtime_s=res['runtime_s'], secs_per_arm_mean=float(np.mean([v['secs'] for v in res['arms'].values()])), capped=capped,
                                                           plant_and_noise_means=pa, code_sha256=CODE_SHA256), indent=1)); log('smoke ok', len(res['arms']), 'arms'); return
    # ---------------- summary
    A = res['arms']; T = lambda v: torch.tensor(np.asarray(v, np.float64), device=DEV); gb = torch.Generator(device=DEV); gb.manual_seed(0); IX = torch.randint(0, nP, (10000, nP), generator=gb, device=DEV)
    have = lambda *ks: all(k in A for k in ks)
    mean = lambda k: A[k]['mean']
    def keep(num, den):
        n, d = T(A[num]['bits']), T(A[den]['bits']); r = n[IX].mean(1) / d[IX].mean(1); return dict(value=float(n.mean() / d.mean()), ci95=[float(r.quantile(.025)), float(r.quantile(.975))], bits=float(n.mean()), denominator_bits=float(d.mean()))
    def read(pref_all, pref_occ, pref_sm, w, tag=''):
        ya = mean(f'{pref_all}|{w}|eeg') if f'{pref_all}|{w}|eeg' in A else None
        if ya is None or not have(pref_occ, pref_sm): return dict(window=w, reading=None, why='arms missing')
        ko, ks = keep(pref_occ, f'{pref_all}|{w}|eeg'), keep(pref_sm, f'{pref_all}|{w}|eeg')
        grad = ya >= PM['min_all_channel_yield_bits']; o, s = ko['value'], ks['value']
        rd = 'visual' if (o >= PM['thr_visual_occ_min'] and s < PM['thr_visual_sm_max']) else ('sensorimotor' if (s >= PM['thr_sm_sm_min'] and o < PM['thr_sm_occ_max']) else 'mixed')
        return dict(window=w, all_channel_bits=ya, gradable=bool(grad), keep_occ=ko, keep_sm=ks, reading=rd if grad else 'not gradable')
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, v3_code_sha256=V3_SHA256, prereg_sha256=ST['prereg_sha256'], n=nP, excluded={str(k): v for k, v in excl.items()}, gate=gate, gate_pass=res['gate_pass'],
             runtime_s=res['runtime_s'], capped_at=capped, arms_scored=len(A), definition=PM['keep_definition'], per_arm_means={k: v['mean'] for k, v in A.items()})
    wins = PM['decision_windows']; S['readings'] = {w: read('all', f'occ|{w}|eeg', f'sm|{w}|eeg', w) for w in wins}
    S['descriptive_t15_40'] = read('all', 'occ|t15_40|eeg', 'sm|t15_40|eeg', 't15_40')
    S['keep_fo_descriptive'] = {w: keep(f'fo|{w}|eeg', f'all|{w}|eeg') for w in wins + ['t15_40'] if have(f'fo|{w}|eeg', f'all|{w}|eeg')}
    S['keep_rand9_descriptive'] = {f'rand9_{i}|{w}': keep(f'rand9_{i}|{w}|eeg', f'all|{w}|eeg') for i in range(2) for w in ('t05_40', 't05_15') if have(f'rand9_{i}|{w}|eeg')}
    S['rand9_channels'] = res['rand9_channels']
    # leak check and floors
    S['leak_precue'] = {g_: mean(f'{g_}|precue|eeg') for g_ in ('all', 'occ', 'sm', 'fo') if f'{g_}|precue|eeg' in A}; S['leak_pass'] = bool(len(S['leak_precue']) == 4 and all(v <= PM['leak_max_bits'] for v in S['leak_precue'].values()))
    S['leak_precue_ci95'] = {g_: M.boot(A[f'{g_}|precue|eeg']['bits']) for g_ in ('all', 'occ', 'sm', 'fo') if f'{g_}|precue|eeg' in A}
    used = [k for k in A if k.endswith('|noise') or k.endswith('|shuffle')]
    S['floors'] = {k: A[k]['mean'] for k in used}; S['floors_pass'] = bool(all(abs(v) <= PM['floor_max_abs_bits'] for v in S['floors'].values()))
    # planted positive control
    pc = {}
    for pn, want in (('occ', 'visual'), ('sm', 'sensorimotor')):
        pc[pn] = {w: read(f'plant_{pn}|all', f'plant_{pn}|occ|{w}|eeg', f'plant_{pn}|sm|{w}|eeg', w) for w in wins}
        for w in wins: pc[pn][w]['want'] = want
    S['planted_control'] = pc; S['planted_pass'] = bool(all(pc[pn][w].get('reading') == want and pc[pn][w].get('gradable') for pn, want in (('occ', 'visual'), ('sm', 'sensorimotor')) for w in wins))
    # verdict
    rds = [S['readings'][w] for w in wins]; reasons = []
    if capped: reasons.append('time cap hit')
    if not res['gate_pass']: reasons.append('gate failed')
    if not S['floors_pass']: reasons.append('a noise or shuffle floor exceeds 0.02')
    if not S['leak_pass']: reasons.append('pre-cue leak check exceeds 0.02 bits')
    if any(r['reading'] in (None, 'not gradable') for r in rds): reasons.append('a decision window is not gradable or missing')
    if reasons: overall = 'VOID'
    else: overall = rds[0]['reading'] if rds[0]['reading'] == rds[1]['reading'] else 'mixed'
    S['verdict'] = dict(overall=overall, void_reasons=reasons, label=PR['reading_rule']['labels'].get(overall), planted_control_pass=S['planted_pass'], validated=bool(S['planted_pass'] and overall != 'VOID'),
                        per_window={w: S['readings'][w]['reading'] for w in wins})
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    pdir = out / 'per_participant'; pdir.mkdir(exist_ok=True)
    for k, s in enumerate(P): (pdir / f'sub-{s:03d}.json').write_text(json.dumps({n: v['bits'][k] for n, v in A.items()}))
    log('VERDICT', json.dumps(S['verdict'])); log('leak', json.dumps(S['leak_precue'])); log('readings', json.dumps({w: (S['readings'][w]['reading'], S['readings'][w].get('keep_occ', {}).get('value'), S['readings'][w].get('keep_sm', {}).get('value')) for w in wins}))


if __name__ == '__main__':
    main()
