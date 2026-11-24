"""A3 (IDEATOR_LIFT_ABOVE_5_20261124, action A3): participant-level detection rule on STORED blank values. No refit, no EEG.

Rule (fixed in results/ext/a3_participant_detection/PREREG.json before any scored output):
  participant j is DETECTED if its score (the yield's un-floored score of the same read-out) is strictly above the 95th percentile
  (linear interpolation, float64) of the POOLED stored blank values of the same read-out and cell (all blank arms listed for the cell,
  all participants). k of n detected; exact one-sided binomial P(X >= k | n, 0.05); "detected at the participant level" only if p < 0.01.
  A cell with fewer than 60 stored pooled blank values is reported, not graded.
The 0.02 tolerance is reported as written (|B_j| <= 0.02), every breach listed. The pooled blank is NOT a permutation null.
Usage: python -I -u a3_participant_detection.py --out <dir> [--smoke]   (smoke: first 3 participants per cell, no aggregate gates)
"""
import argparse, glob, hashlib, json, math, os, sys, time
import torch

ROOT = '/path/to/eegaudit'
R = ROOT + '/results'
CODE = os.path.abspath(__file__)
P0, ALPHA, NB_MIN, TOL_GATE, TOL_BLANK, Q = 0.05, 0.01, 60, 1e-4, 0.02, 0.95
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
J = lambda p: json.load(open(p))


def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()


def binom_sf(k, n, p=P0):  # exact P(X >= k)
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


# ---------------------------------------------------------------- loaders: each returns dict(score={pid:v}, arms={name:{pid:v}}, groups={name:[arms]}, gates=[(name,computed,stored)], align=[(name,maxabsdiff)])
def load_pe_reg(ds):
    rep = J(ROOT + '/outputs/AGENT_LOOP_20261120/results/log_yield_replication.json')['datasets'][ds]
    sc = {str(x['subject']): x['gain'] for x in rep}
    arms = {f'shift{i}': {str(x['subject']): x['shifted'][i] for x in rep} for i in range(3)}
    st = J(R + '/standard/summary.json')['prediction_error'][f'yield_{ds}']
    return dict(score=sc, arms=arms, groups={'shift_mean': ['shift0', 'shift1', 'shift2']},
                gates=[('registered_mean_yield', None, st['mean']), ('registered_mean_shift_floor', None, st['floor']),
                       ('n_positive', None, st['n_positive']), ('n_above_floor', None, st['n_above_floor'])], gate_fn='pe_reg')


def load_pe_refit():
    d = R + '/gpu_fill_20261122/pe_eegnet_floor'
    cond = lambda c: {k: v['gain'] for k, v in J(f'{d}/cond_{c}.json')['rows'].items()}
    real = J(f'{d}/cond_real.json')['rows']
    arms = {c: cond(c) for c in ['phase0', 'phase1', 'phase2', 'shuffle0', 'shuffle1', 'shuffle2']}
    rep = {str(x['subject']): x['gain'] for x in J(ROOT + '/outputs/AGENT_LOOP_20261120/results/log_yield_replication.json')['datasets']['ds003458']}
    s = J(d + '/summary.json')
    return dict(score={k: v['gain'] for k, v in real.items()}, arms=arms,
                groups={'phase_seedmean': ['phase0', 'phase1', 'phase2'], 'shuffle_seedmean': ['shuffle0', 'shuffle1', 'shuffle2']},
                align=[('refit_saved_gain_vs_registered_gain', max(abs(real[k]['saved_gain'] - rep[k]) for k in real))],
                gates=[('refit_mean_yield', None, s['bits']['real']['mean']),
                       ('phase_max_abs_per_participant_seedmean', None, s['phase_max_abs_per_participant_seedmean']),
                       ('shuffle_max_abs_per_participant_seedmean', None, s['shuffle_max_abs_per_participant_seedmean'])], gate_fn='refit')


def load_ridge():
    st = J(R + '/ext/d2_ds007647_v2/summary.json')['cells']['all|pe_resid|L64']
    sc, a = {}, {'shift0': {}, 'shift1': {}, 'shift2': {}, 'noise': {}}
    al = 0.0
    for f in sorted(glob.glob(R + '/ext/d2_ds007647_full/ds007647_sub-*.json')):
        o = J(f); x = o['windows']['w0600']['all']; p = o['participant']
        sc[p] = x['gain']
        for i in range(3): a[f'shift{i}'][p] = x['shifted'][i]
        a['noise'][p] = x['noise']
        v2 = J(R + f'/ext/d2_ds007647_v2/{p}.json')['subsets']['all']['pe_resid']['64']
        al = max(al, abs(v2['gain'] - x['gain']), abs(v2['shift_mean'] - sum(x['shifted']) / 3), abs(v2['noise'] - x['noise']))
    return dict(score=sc, arms=a, groups={'shift_mean': ['shift0', 'shift1', 'shift2'], 'noise': ['noise']},
                align=[('d2_full_vs_d2_v2_gain_shiftmean_noise', al)], gates=[('v2_mean_gain_all_pe_resid_L64', None, st['gain']['mean'])], gate_fn='ridge')


def mi_ids():
    return [f'{p:03d}' for p in J(R + '/ext/mi_card_v3/primary_lpgo/meta.json')['participants']]


def load_mi(key):  # raw or raw_eogreg from mi_card_v3 (+ phase_control_mi for raw)
    ids = mi_ids(); base = R + '/ext/mi_card_v3/primary_lpgo'
    c = {i: J(f'{base}/sub-{i}.json')[key] for i in ids}
    sc = {i: c[i]['eeg']['bits'] for i in ids}
    arms = {'noise': {i: c[i]['noise']['bits'] for i in ids}, 'shuffle': {i: c[i]['shuffle']['bits'] for i in ids}}
    groups = {'noise': ['noise'], 'shuffle': ['shuffle']}
    al = []
    if key == 'raw':
        pc = {i: J(f'{R}/ext/phase_control_mi/sub{i}.json')['bits'] for i in ids}
        al.append(('phase_control_real_vs_mi_card_raw', max(abs(pc[i]['real'] - sc[i]) for i in ids)))
        for n in ['phase0', 'phase1', 'phase2', 'shuffle0', 'shuffle1', 'shuffle2']:
            arms['pc_' + n] = {i: pc[i][n] for i in ids}
        groups['phase_seedmean'] = ['pc_phase0', 'pc_phase1', 'pc_phase2']; groups['pc_shuffle_seedmean'] = ['pc_shuffle0', 'pc_shuffle1', 'pc_shuffle2']
    s = J(base + '/summary_mi.json')['rows'][key]
    g = [(f'{key}_mean_yield', None, s['eeg']['mean']), (f'{key}_mean_noise', None, s['noise']['mean']), (f'{key}_mean_shuffle', None, s['shuffle']['mean'])]
    return dict(score=sc, arms=arms, groups=groups, align=al, gates=g, gate_fn='mi_' + key)


def load_fm(m):
    ids = mi_ids(); e = R + f'/ext/mi_fm_eogreg/{m}'; ph = R + f'/gpu_fill_20261122/mi_fm_phase/{m}'
    ge = J(R + '/ext/mi_fm_eogreg/gate.json')['models'][m]
    eo = {i: J(f'{e}/sub-{i}.json')['fm'] for i in ids}; po = {i: J(f'{ph}/sub-{i}.json')['fm'] for i in ids}
    sc = {i: eo[i]['eeg']['bits'] for i in ids}
    arms = {'noise': {i: eo[i]['noise']['bits'] for i in ids}, 'phase0': {i: po[i]['phase0']['bits'] for i in ids}, 'phase1': {i: po[i]['phase1']['bits'] for i in ids}}
    return dict(score=sc, arms=arms, groups={'noise': ['noise'], 'phase_seedmean': ['phase0', 'phase1']},
                align=[('mi_fm_phase_real_vs_eogreg_eeg', max(abs(po[i]['real']['bits'] - sc[i]) for i in ids))],
                gates=[(f'{m}_fm_eeg_mean', None, ge['fm_eeg']), (f'{m}_fm_noise_mean', None, ge['fm_noise'])], gate_fn='fm', sens={'noise_only': ['noise']})


def load_mi_eegnet():
    d = R + '/gpu_fill_20261122/mi_eegnet_floor'
    cond = lambda c: {k: v['bits'] for k, v in J(f'{d}/cond_{c}.json')['rows'].items()}
    arms = {c: cond(c) for c in ['phase0', 'phase1', 'phase2', 'shuffle0', 'shuffle1']}  # this run stored only 2 shuffle seeds
    s = J(d + '/summary.json')
    return dict(score=cond('real'), arms=arms, groups={'phase_seedmean': ['phase0', 'phase1', 'phase2'], 'shuffle_seedmean': ['shuffle0', 'shuffle1']},
                gates=[('refit_mean_yield', None, s['bits']['real']['mean']), ('phase_max_abs_per_participant_seedmean', None, s['phase_max_abs_per_participant_seedmean']),
                       ('shuffle_max_abs_per_participant_seedmean', None, s['shuffle_max_abs_per_participant_seedmean'])], gate_fn='refit')


CELLS = [  # (id, role, loader)
    ('PE_ds003458_registered_EEGNet', 'primary', lambda: load_pe_reg('ds003458')),
    ('PE_ds004595_registered_EEGNet', 'primary', lambda: load_pe_reg('ds004595')),
    ('PE_ds007647_ridge_resid_L64', 'primary', load_ridge),
    ('MI_raw_voltage', 'primary', lambda: load_mi('raw')),
    ('MI_raw_voltage_eogreg', 'primary', lambda: load_mi('raw_eogreg')),
    ('MI_FM_eogreg_cbramod', 'primary', lambda: load_fm('cbramod')),
    ('MI_FM_eogreg_reve', 'primary', lambda: load_fm('reve')),
    ('MI_FM_eogreg_labram', 'primary', lambda: load_fm('labram')),
    ('MI_FM_eogreg_eegpt', 'primary', lambda: load_fm('eegpt')),
    ('PE_ds003458_refit_EEGNet', 'secondary', load_pe_refit),
    ('MI_EEGNet_refit', 'secondary', load_mi_eegnet),
]


def T(x): return torch.tensor(x, dtype=torch.float64, device=dev)


def run_cell(cid, role, c, smoke):
    pids = sorted(c['score'], key=lambda s: (len(s), s))
    if smoke: pids = pids[:3]
    n = len(pids)
    for a in c['arms'].values(): assert set(pids) <= set(a), (cid, 'arm missing participants')
    y = T([c['score'][p] for p in pids])
    arm_names = list(c['arms'])
    A = torch.stack([T([c['arms'][a][p] for p in pids]) for a in arm_names])  # arms x n
    pool = A.reshape(-1); owner = torch.arange(n, device=dev).repeat(len(arm_names))
    nb = int(pool.numel()); out = dict(cell=cid, role=role, n=n, n_blank_values=nb, arms=arm_names, groups=c['groups'])
    out['align'] = {k: float(v) for k, v in c.get('align', [])}
    out['align_ok'] = all(v <= TOL_GATE for v in out['align'].values())
    thr = torch.quantile(pool, Q).item()
    det = y > thr
    k = int(det.sum()); out.update(threshold_p95=thr, k=k, frac=k / n, binom_p=binom_sf(k, n), mean_score=float(y.mean()), median_score=float(y.median()))
    out['graded'] = nb >= NB_MIN and not smoke
    out['claim_detected_p_lt_0.01'] = bool(out['graded'] and out['binom_p'] < ALPHA)
    # participant bootstrap of the detection fraction (threshold fixed), 10000, seed 0
    g = torch.Generator(device=dev); g.manual_seed(0)
    ix = torch.randint(0, n, (10000, n), generator=g, device=dev)
    bs = det.double()[ix].mean(1); out['frac_ci95'] = [float(torch.quantile(bs, 0.025)), float(torch.quantile(bs, 0.975))]
    # leave-participant-out threshold (descriptive)
    thr_loo = torch.stack([torch.quantile(pool[owner != j], Q) for j in range(n)])
    kl = int((y > thr_loo).sum()); out['loo'] = dict(k=kl, binom_p=binom_sf(kl, n), thr_min=float(thr_loo.min()), thr_max=float(thr_loo.max()))
    # the 0.02 tolerance, as written: |B_j| <= 0.02 per participant, B_j = mean over the arms of each group (seed mean), and every single arm value
    br = {}
    for gname, members in c['groups'].items():
        Bj = torch.stack([A[arm_names.index(m)] for m in members]).mean(0)
        bad = (Bj.abs() > TOL_BLANK).nonzero().flatten().tolist()
        br[gname] = dict(n_breach=len(bad), max_abs=float(Bj.abs().max()), participants=[pids[i] for i in bad], values=[float(Bj[i]) for i in bad])
    single = (A.abs() > TOL_BLANK)
    br['_single_arm_values'] = dict(n_breach=int(single.sum()), n_values=int(single.numel()), max_abs=float(A.abs().max()))
    out['gate_0.02_breaches'] = br
    # sensitivity: pooled blank from named arms only (descriptive)
    if c.get('sens'):
        for sn, members in c['sens'].items():
            pl = torch.stack([A[arm_names.index(m)] for m in members]).reshape(-1); t2 = torch.quantile(pl, Q).item(); k2 = int((y > t2).sum())
            out.setdefault('sensitivity', {})[sn] = dict(threshold_p95=t2, n_blank_values=int(pl.numel()), k=k2, binom_p=binom_sf(k2, n))
    # null diagnostic: each blank arm as the "yield", the other arms pooled (calibration; expect about 5% above)
    nd = {}
    if len(arm_names) >= 2:
        for ai, a in enumerate(arm_names):
            rest = torch.stack([A[j] for j in range(len(arm_names)) if j != ai]).reshape(-1)
            if rest.numel() < NB_MIN: nd[a] = 'skipped: other-arm pool < 60'; continue
            t2 = torch.quantile(rest, Q).item(); k2 = int((A[ai] > t2).sum()); nd[a] = dict(k=k2, n=n, binom_p=binom_sf(k2, n))
    out['null_arm_as_yield'] = nd
    # planted positive control: arm0 + 5 SD of the pooled blank must be detected in (almost) all
    sd = float(pool.std()); pl_ = A[0] + 5 * sd; kp = int((pl_ > thr).sum()); out['planted_5sd'] = dict(k=kp, n=n, sd_pooled=sd, ok=kp >= n - 1)
    return out, dict(participants=pids, score=y.tolist(), detected=det.tolist(), detected_loo=(y > thr_loo).tolist(),
                     arms={a: A[i].tolist() for i, a in enumerate(arm_names)})


def gate_values(c, pids_all):
    """recompute the stored aggregates from the per-participant values"""
    sc = torch.tensor([c['score'][p] for p in pids_all], dtype=torch.float64)
    m = lambda arms: float(torch.stack([torch.tensor([c['arms'][a][p] for p in pids_all], dtype=torch.float64) for a in arms]).mean())
    fn = c['gate_fn']; g = {}
    if fn == 'pe_reg':
        g = {'registered_mean_yield': float(sc.mean()), 'registered_mean_shift_floor': m(['shift0', 'shift1', 'shift2']), 'n_positive': int((sc > 0).sum()),
             'n_above_floor': int((sc > torch.stack([torch.tensor([c['arms'][a][p] for p in pids_all], dtype=torch.float64) for a in ['shift0', 'shift1', 'shift2']]).mean(0)).sum())}
    elif fn in ('refit',):
        ph = torch.stack([torch.tensor([c['arms'][a][p] for p in pids_all], dtype=torch.float64) for a in ['phase0', 'phase1', 'phase2']]).mean(0)
        sh = torch.stack([torch.tensor([c['arms'][a][p] for p in pids_all], dtype=torch.float64) for a in c['groups']['shuffle_seedmean']]).mean(0)
        g = {'refit_mean_yield': float(sc.mean()), 'phase_max_abs_per_participant_seedmean': float(ph.abs().max()), 'shuffle_max_abs_per_participant_seedmean': float(sh.abs().max())}
    elif fn == 'ridge':
        g = {'v2_mean_gain_all_pe_resid_L64': float(sc.mean())}
    elif fn.startswith('mi_'):
        k = fn[3:]; g = {f'{k}_mean_yield': float(sc.mean()), f'{k}_mean_noise': m(['noise']), f'{k}_mean_shuffle': m(['shuffle'])}
    elif fn == 'fm':
        g = None  # filled by caller (needs model name)
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.smoke:
        assert os.path.exists(a.out + '/PREREG_STAMP.json'), 'stamp PREREG first'
        assert not os.path.exists(a.out + '/summary.json')
    t0 = time.time(); res = []; gate_fail = []
    for cid, role, ld in CELLS:
        c = ld()
        out, pp = run_cell(cid, role, c, a.smoke)
        # aggregate gates (full runs only)
        gates = {}
        if not a.smoke:
            pids = sorted(c['score'], key=lambda s: (len(s), s))
            gv = gate_values(c, pids)
            if gv is None:
                m = cid.split('_')[-1]; sc = torch.tensor([c['score'][p] for p in pids], dtype=torch.float64); nz = torch.tensor([c['arms']['noise'][p] for p in pids], dtype=torch.float64)
                gv = {f'{m}_fm_eeg_mean': float(sc.mean()), f'{m}_fm_noise_mean': float(nz.mean())}
            for name, _, stored in c['gates']:
                comp = gv[name]; ok = abs(comp - stored) <= TOL_GATE
                gates[name] = dict(computed=comp, stored=stored, abs_diff=abs(comp - stored), ok=bool(ok))
                if not ok: gate_fail.append((cid, name))
        out['gates_stored_vs_recomputed'] = gates
        out['gates_ok'] = all(v['ok'] for v in gates.values()) and out['align_ok']
        if not out['gates_ok'] and not a.smoke: gate_fail.append((cid, 'align' if not out['align_ok'] else 'gates'))
        out['void'] = bool((not out['gates_ok']) and not a.smoke)
        if out['void']: out['claim_detected_p_lt_0.01'] = False
        res.append(out)
        json.dump(pp, open(f'{a.out}/per_participant_{cid}.json', 'w'))
        print(cid, role, 'n', out['n'], 'n_b', out['n_blank_values'], 'k', out['k'], 'p95 %.5f' % out['threshold_p95'], 'binom_p %.3g' % out['binom_p'],
              'align_ok', out['align_ok'], 'gates_ok', out['gates_ok'], flush=True)
    summ = dict(NOT_CITABLE='until independent blind review', code_sha256=sha(CODE), prereg_sha256=sha(a.out + '/PREREG.json') if os.path.exists(a.out + '/PREREG.json') else None,
                smoke=a.smoke, device=dev, constants=dict(percentile=Q, p0=P0, alpha=ALPHA, n_blank_min=NB_MIN, gate_tol=TOL_GATE, blank_tol_as_written=TOL_BLANK),
                cells=res, gate_failures=gate_fail, runtime_s=time.time() - t0,
                not_run=['text DeWave-style (no per-participant blank arms inspected; not attempted)'])
    json.dump(summ, open(f'{a.out}/summary.json', 'w'), indent=1)
    print('done', 'gate_failures', gate_fail, 'runtime %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
