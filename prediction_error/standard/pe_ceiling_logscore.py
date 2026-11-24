"""Standard test, PE ceiling and usable fraction under the common log-score (ds003458, EEGNet). METHOD CHANGE 10 (02:35, 21 Nov).
Ceiling source: results/loop/r5_eegnet_ceiling/r5_eegnet_splithalf_30ep_all.json (two EEGNets trained on interleaved halves of the pooled training
trials, 30 epochs = the training budget of the yield's EEGNet, n = 23). Per-trial split-half predictions were NOT cached, so the ceiling is in closed
form: for a Gaussian target and a linear-Gaussian read-out the held-out density gain over intercept-only is -1/2 log2(1 - rho^2) bits per trial,
evaluated at the MEAN over participants of the Spearman-Brown reliability rel_decoder_sb (= rho_max^2; target reliability ~0.99 taken as 1).
Numerator: the measured held-out Gaussian density gain of the SAME read-out, i.e. the 30-epoch LOPO EEGNet of deep_decoders.main
(results/features/eegnet/sub-XXX.npz, read by log_yield_replication.py, ds003458); floor_i = mean of its 3 circular shifts.
PRIMARY fraction = mean_i (gain_i - floor_i) / (ceiling - mean_i floor_i), two-sample bootstrap (participants for gain/floor; participants for the
ceiling reliabilities), 10000 resamples, seed 20261120. Sensitivity: the same with the 10-epoch n = 23 ceiling (r5_eegnet_splithalf_all.json);
per-participant-ceiling variant mean_i (gain_i - floor_i)/(c_i - floor_i) (paired bootstrap). Consistency check: closed-form gain at the measured r_full
vs the measured gain, per participant. Output: results/standard/pe_ceiling/{per_participant.json, summary.json}."""
import json, numpy as np
from pathlib import Path
ROOT = Path('/path/to/eegaudit'); OUT = ROOT / 'results/standard/pe_ceiling'; OUT.mkdir(parents=True, exist_ok=True)
gb = lambda r2: float(-0.5 * np.log2(1 - min(r2, 0.999999)))


def main():
    L = ROOT / 'results/loop/r5_eegnet_ceiling'
    sh = json.loads((L / 'r5_eegnet_splithalf_30ep_all.json').read_text())['rows']; sh10 = json.loads((L / 'r5_eegnet_splithalf_all.json').read_text())['rows']
    assert len(sh) == 23 and all(r['epochs'] == 30 for r in sh) and len(sh10) == 23
    rep = {r['subject']: r for r in json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/log_yield_replication.json').read_text())['datasets']['ds003458']}
    subs = sorted(rep); assert subs == [r['subject'] for r in sh], 'yield and ceiling participants differ'
    g = np.array([rep[s]['gain'] for s in subs]); f = np.array([float(np.mean(rep[s]['shifted'])) for s in subs])
    rel = np.array([max(r['rel_decoder_sb'], 0) for r in sh]); rel10 = np.array([max(r['rel_decoder_sb'], 0) for r in sh10])
    c_i = np.array([gb(x) for x in rel]); cf = np.array([gb(r['r_full'] ** 2) for r in sh])
    rows = [dict(subject=s, n=r['n'], gain=float(g[k]), shift_floor=float(f[k]), rel_decoder_sb=r['rel_decoder_sb'], ceiling_bits_own=float(c_i[k]), r_full=r['r_full'],
                 closed_form_gain_at_r_full=float(cf[k])) for k, (s, r) in enumerate(zip(subs, sh))]
    (OUT / 'per_participant.json').write_text(json.dumps(dict(note=__doc__, rows=rows), indent=1))
    rng = np.random.default_rng(20261120); B = 10000; q = lambda v: np.quantile(v, [.025, .975]).tolist()
    iy = rng.integers(0, len(g), size=(B, len(g))); ic = rng.integers(0, len(rel), size=(B, len(rel)))

    def frac(relv, icv):
        C = gb(relv.mean()); pt = float(np.mean((g - f) / (C - f.mean())))
        bs = [np.mean((g[a] - f[a]) / (gb(relv[b].mean()) - f[a].mean())) for a, b in zip(iy, icv)]
        return dict(value=pt, ci=q(np.array(bs)), ceiling_bits=float(C), rel_mean=float(relv.mean()), rho_max_mean=float(np.sqrt(relv).mean()), n_yield=len(g), n_ceiling=len(relv))
    ic10 = rng.integers(0, len(rel10), size=(B, len(rel10)))
    pp = (g - f) / (c_i - f)
    res = dict(note=__doc__, n_ceiling=len(rel), ceiling_participants=subs, ceiling_source='closed form at the mean split-half reliability, 30 epochs, n = 23 (per-trial split-half predictions not cached)',
               ceiling=dict(mean=float(gb(rel.mean())), ci=q(np.array([gb(rel[b].mean()) for b in ic])), mean_of_own_ceilings=float(c_i.mean())),
               yield_n23=dict(mean=float(g.mean()), ci=q(g[iy].mean(1)), n_positive=int((g > 0).sum()), floor=float(f.mean()), floor_ci=q(f[iy].mean(1)),
                              minus_floor=float((g - f).mean()), minus_floor_ci=q((g - f)[iy].mean(1)), n_above_all_shifts=int(sum(rep[s]['gain'] > max(rep[s]['shifted']) for s in subs))),
               fraction_primary=frac(rel, ic), fraction_sensitivity_ceiling10=frac(rel10, ic10),
               fraction_sensitivity_per_participant_ceiling=dict(value=float(pp.mean()), ci=q(pp[iy].mean(1))),
               closed_form_check=dict(mean_closed_form_at_r_full=float(cf.mean()), mean_measured=float(g.mean()), max_abs_diff=float(np.abs(cf - g).max())))
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k not in ('note', 'ceiling_participants')}))


if __name__ == '__main__':
    main()
