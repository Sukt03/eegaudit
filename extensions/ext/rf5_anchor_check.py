"""RUNNER FIX 5 (round_20261123_auto; METHODS B4, EXPERIMENTS G3): redefined anchor check (iii) on the MI run-pair retest rows.
Old rule (PDF): lower 95% bound of the Spearman-Brown split-half correlation > 0.1 -> noise-input and shuffled-label rows pass it (it certifies stable
participant-level calibration offsets, not signal).
New rule, fixed in this file BEFORE running (two parts, both required):
  (iii-a) lower 95% participant-bootstrap bound of SB(EEG - noise-input, per trial) > 0;
  (iii-b) SB(EEG) exceeds SB(noise-input) AND SB(shuffled-label) by a margin >= 0.2 (point estimates; the reviewer's wording).
  Sensitivity 'strict': the lower 95% bound of each margin SB(EEG) - SB(null) is > 0 (paired bootstrap, same resampled participants for both rows).
  A row without a shuffled arm (EEGNet) is judged on the noise margin only and flagged.
Split-half as mi_retest_agg.py: halves = alternating test trials within participant, Pearson r of the two half means across participants, SB = 2r/(1+r).
Input: results/gpu_fill_20261123/mi_retest_*/pertrial.npz (per-trial bits of the eeg / noise / shuffle arms).
GATE: SB point estimates == saved mi_retest_summary.json check_iii sb (1e-4).
Controls on the same participant/trial layout (synthetic per-trial bits): planted participant-level signal in the EEG arm only -> must PASS; no signal anywhere -> must FAIL;
a calibration offset shared by all three arms (what makes null rows pass the old rule) -> must FAIL the new rule while passing the old one.
All numerics in torch on the GPU. Usage: --out DIR"""
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); G = ROOT / 'results/gpu_fill_20261123'; DEV = 'cuda'; B = 10000; MARGIN = 0.2
CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()

def halves(v, sid, nP):                                   # per-participant means of alternating trials -> [nP, 2]
    v = torch.as_tensor(v, dtype=torch.float64, device=DEV); sid = torch.as_tensor(sid, device=DEV); out = torch.zeros(nP, 2, dtype=torch.float64, device=DEV)
    for k in range(nP):
        x = v[sid == k]; out[k, 0] = x[0::2].mean(); out[k, 1] = x[1::2].mean()
    return out

def sb(H, ix=None):                                       # H [n, 2]; ix [B, n] resampled participants or None -> SB (scalar or [B])
    a, c = (H[:, 0], H[:, 1]) if ix is None else (H[:, 0][ix], H[:, 1][ix]); a = a - a.mean(-1, keepdim=True); c = c - c.mean(-1, keepdim=True)
    r = (a * c).sum(-1) / ((a * a).sum(-1) * (c * c).sum(-1)).sqrt(); return 2 * r / (1 + r)

def q(x): x = x[torch.isfinite(x)]; return [float(x.quantile(.025)), float(x.quantile(.975))]

def judge(name, arms, sid, nP, ix):
    """arms: dict eeg / noise / shuffle(optional) of per-trial bits."""
    H = {k: halves(v, sid, nP) for k, v in arms.items()}; H['eeg_minus_noise'] = halves(np.asarray(arms['eeg']) - np.asarray(arms['noise']), sid, nP)
    o = dict(name=name, n=nP, n_trials=int(len(sid)), sb={k: dict(sb=float(sb(h)), ci95=q(sb(h, ix)), old_rule_pass_lower_gt_0p1=bool(q(sb(h, ix))[0] > 0.1)) for k, h in H.items()}, margins={})
    for nul in ('noise', 'shuffle'):
        if nul in H: d = sb(H['eeg'], ix) - sb(H[nul], ix); o['margins'][nul] = dict(d=float(sb(H['eeg']) - sb(H[nul])), ci95=q(d), point_ge_margin=bool(float(sb(H['eeg']) - sb(H[nul])) >= MARGIN), lower_gt_0=bool(q(d)[0] > 0))
    o['iii_a_lower_sb_eeg_minus_noise_gt_0'] = bool(o['sb']['eeg_minus_noise']['ci95'][0] > 0); o['iii_b_margins_ge_0p2'] = bool(all(m['point_ge_margin'] for m in o['margins'].values()))
    o['check_iii_new_pass'] = bool(o['iii_a_lower_sb_eeg_minus_noise_gt_0'] and o['iii_b_margins_ge_0p2']); o['check_iii_strict_pass'] = bool(o['iii_a_lower_sb_eeg_minus_noise_gt_0'] and all(m['lower_gt_0'] for m in o['margins'].values()))
    o['no_shuffle_arm'] = 'shuffle' not in arms; o['mean_bits'] = {k: float(np.mean(v)) for k, v in arms.items()}; return o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    saved = {o['label']: o for o in json.loads((G / 'mi_retest_summary.json').read_text())}; rows = []
    spec = [('raw voltage', 'raw', [G / 'mi_retest_logistic']), ('raw proxies-regressed', 'raw_eogreg', [G / 'mi_retest_logistic']), ('EEGNet regressed', 'eegnet_eogreg', sorted(glob.glob(str(G / 'mi_retest_eegnet_eogreg/s*'))))] + \
           [(f'{m.upper()} regressed input', f'fm_{m}', [G / f'mi_retest_fm_{m}']) for m in ('cbramod', 'reve', 'labram', 'eegpt')]
    lay = None
    for lab, row, dirs in spec:
        arms, sid, off = {}, [], 0
        for d in dirs:
            z = np.load(Path(d) / 'pertrial.npz'); sid.append(z['sid'] + off); off += len(z['participants'])
            for k in ('eeg', 'noise', 'shuffle'):
                if f'{row}__{k}' in z.files: arms.setdefault(k, []).append(z[f'{row}__{k}'])
        sid = np.concatenate(sid); arms = {k: np.concatenate(v) for k, v in arms.items()}; nP = off; assert all(len(v) == len(sid) for v in arms.values()), (lab, {k: len(v) for k, v in arms.items()}, len(sid))
        g = torch.Generator(device=DEV); g.manual_seed(0); ix = torch.randint(0, nP, (B, nP), generator=g, device=DEV); o = judge(lab, arms, sid, nP, ix); o['row'] = row
        s = saved.get(row, {}).get('check_iii_split_half')
        if s: o['gate'] = dict(saved_sb=s['sb'], this_sb_eeg_minus_noise=o['sb']['eeg_minus_noise']['sb'], this_sb_eeg=o['sb']['eeg']['sb'], abs_diff_eeg_minus_noise=abs(s['sb'] - o['sb']['eeg_minus_noise']['sb']), abs_diff_eeg=abs(s['sb'] - o['sb']['eeg']['sb']))
        sv = saved.get(row)
        if sv: o['check_ii_saved_R_gain_ci_excludes_blank'] = sv['check_ii_R_ci_excludes_blank']; o['R_saved'] = sv['R']['mean']; o['R_blank_saved'] = sv['R_blank']; o['R_gain_ci_saved'] = sv['R_gain']['ci']
        o['pass_ii_and_iii_new'] = bool(o.get('check_ii_saved_R_gain_ci_excludes_blank') and o['check_iii_new_pass']); o['pass_ii_and_iii_a'] = bool(o.get('check_ii_saved_R_gain_ci_excludes_blank') and o['iii_a_lower_sb_eeg_minus_noise_gt_0'])
        rows.append(o)
        if lay is None: lay = (sid, nP, ix)
        print(lab, 'SB eeg', round(o['sb']['eeg']['sb'], 3), o['sb']['eeg']['ci95'], '| noise', round(o['sb']['noise']['sb'], 3), '| shuffle', round(o['sb']['shuffle']['sb'], 3) if 'shuffle' in o['sb'] else None, '| eeg-noise', round(o['sb']['eeg_minus_noise']['sb'], 3), o['sb']['eeg_minus_noise']['ci95'],
              '| margins', {k: (round(m['d'], 3), [round(x, 3) for x in m['ci95']]) for k, m in o['margins'].items()}, '| NEW', o['check_iii_new_pass'], 'STRICT', o['check_iii_strict_pass'], '| gate', o.get('gate'), flush=True)
    # controls on the logistic layout
    sid, nP, ix = lay; st = torch.as_tensor(sid, device=DEV); g = torch.Generator(device=DEV); g.manual_seed(42); rn = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.float64)
    def syn(sig_eeg, off_all):
        offs = off_all * rn(nP); sig = sig_eeg * rn(nP); return {k: (offs[st] + (sig[st] if k == 'eeg' else 0) + 0.5 * rn(len(sid))).cpu().numpy() for k in ('eeg', 'noise', 'shuffle')}
    ctr = dict(planted_signal_eeg_only=judge('planted', syn(0.15, 0.0), sid, nP, ix), no_signal=judge('null', syn(0.0, 0.0), sid, nP, ix), shared_calibration_offset_only=judge('offset', syn(0.0, 0.15), sid, nP, ix),
               planted_signal_plus_shared_offset=judge('planted+offset', syn(0.15, 0.15), sid, nP, ix))
    exp = dict(planted_signal_eeg_only=True, no_signal=False, shared_calibration_offset_only=False, planted_signal_plus_shared_offset=True)
    cs = {k: dict(iii_a_pass=v['iii_a_lower_sb_eeg_minus_noise_gt_0'], iii_b_pass=v['iii_b_margins_ge_0p2'], new_pass=v['check_iii_new_pass'], strict_pass=v['check_iii_strict_pass'], expected=exp[k], old_rule_eeg_pass=v['sb']['eeg']['old_rule_pass_lower_gt_0p1'], sb_eeg=v['sb']['eeg']['sb'], sb_noise=v['sb']['noise']['sb'], sb_eeg_minus_noise=v['sb']['eeg_minus_noise']['sb']) for k, v in ctr.items()}
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, rule=__doc__, margin=MARGIN, bootstrap='participant, 10000, GPU, seed 0, paired across arms', rows=rows, controls=cs,
             controls_behave=bool(all(v['new_pass'] == v['expected'] for v in cs.values())), controls_behave_iii_a_alone=bool(all(v['iii_a_pass'] == v['expected'] for v in cs.values())),
             rows_pass_iii_a=[r['name'] for r in rows if r['iii_a_lower_sb_eeg_minus_noise_gt_0']], rows_pass_iii_b=[r['name'] for r in rows if r['iii_b_margins_ge_0p2']],
             rows_pass_ii_saved=[r['name'] for r in rows if r.get('check_ii_saved_R_gain_ci_excludes_blank')], rows_pass_ii_and_iii_new=[r['name'] for r in rows if r['pass_ii_and_iii_new']],
             rows_pass_ii_and_iii_a=[r['name'] for r in rows if r['pass_ii_and_iii_a']], R_vs_blank={r['name']: [r.get('R_saved'), r.get('R_blank_saved'), r.get('R_gain_ci_saved')] for r in rows}, rows_pass_new=[r['name'] for r in rows if r['check_iii_new_pass']], rows_pass_strict=[r['name'] for r in rows if r['check_iii_strict_pass']],
             null_rows_passing_old_rule=[f"{r['name']}|{k}" for r in rows for k in ('noise', 'shuffle') if k in r['sb'] and r['sb'][k]['old_rule_pass_lower_gt_0p1']],
             gate_max_abs_diff_eeg_minus_noise=max(r['gate']['abs_diff_eeg_minus_noise'] for r in rows if 'gate' in r), gate_max_abs_diff_eeg=max(r['gate']['abs_diff_eeg'] for r in rows if 'gate' in r))
    (out / 'summary.json').write_text(json.dumps(S, indent=1)); print(json.dumps({k: v for k, v in S.items() if k not in ('rows', 'rule')}, indent=1))

if __name__ == '__main__': main()
