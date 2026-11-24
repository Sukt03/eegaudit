"""L2: latent-target swap with the decoder fixed. The cached read-out (EEGNet LOPO prediction; ensemble features) is
coupled, with the exact null and the group_stats.py statistics, to: the paper's target (PE residualised on outcome, arm,
interaction, cubic time), signed PE, |PE|, |PE| residualised, feedback valence (reward), previous reward.
Also the per-participant correlation between signed PE and valence (and |PE| vs valence, residualised PE vs valence).
Usage: python -I l2_target_swap.py eegnet [ens ...]"""
import sys, json, time, numpy as np
import sys; sys.path.insert(0, "/path/to/eegaudit/experiments/loop/learner"); from common import *
t0 = time.time(); OUT = ROOT / "results/loop/learner"; OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["pe_resid", "pe_signed", "pe_abs", "pe_abs_resid", "valence", "prev_reward"]
# target-target correlations in this task (all trials of each participant)
tt = {}
for a, b in (("pe_signed", "valence"), ("pe_abs", "valence"), ("pe_resid", "valence"), ("pe_signed", "pe_abs"), ("pe_signed", "prev_reward"), ("pe_resid", "pe_signed")):
    tri0 = np.load(ROOT / "results/features/eegnet/sub-001.npz")["tri"]
    cs = []
    for j in range(NS):
        tri = np.load(ROOT / f"results/features/eegnet/sub-{j + 1:03d}.npz")["tri"]
        cs.append(corr(target(a, j, tri), target(b, j, tri)))
    tt[f"{a}~{b}"] = {"mean": float(np.mean(cs)), "min": float(np.min(cs)), "max": float(np.max(cs)), "per_participant": cs}
    print(f"corr {a} ~ {b}: mean {np.mean(cs):.3f} [{np.min(cs):.3f}, {np.max(cs):.3f}]", flush=True)
res = {"targets": TARGETS, "target_correlations": tt, "decoders": {}}
for dec in sys.argv[1:]:
    res["decoders"][dec] = {}
    C = {}
    for t in TARGETS:
        C[t] = coupling(dec, t); c = C[t]
        res["decoders"][dec][t] = strip(c)
        print(f"{dec} -> {t}: rho_eff {c['rho_eff_pooled']:.3f} [{c['rho_eff_ci'][0]:.3f}, {c['rho_eff_ci'][1]:.3f}] mean r {c['r_mean']:.3f} null {c['null_r_mean']:.3f} "
              f"gamma0 {c['prevalence']['gamma0']:.2f} indiv {c['n_individually_above_95pct_null']}/{NS} ({time.time() - t0:.0f}s)", flush=True)
    res["decoders"][dec]["pairs"] = {f"pe_resid-{t}": paired(C["pe_resid"], C[t]) for t in TARGETS if t != "pe_resid"}
    res["decoders"][dec]["pairs"]["pe_signed-valence"] = paired(C["pe_signed"], C["valence"])
    (OUT / "l2_target_swap.json").write_text(json.dumps(res, indent=1))
res["runtime_s"] = round(time.time() - t0, 1); (OUT / "l2_target_swap.json").write_text(json.dumps(res, indent=1)); print("done", res["runtime_s"])
