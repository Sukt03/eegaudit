"""Belief mode in bits: how much information about a learner's parameters does single-trial feedback EEG add?

Reuses the cogeeg exact-posterior results (no new fitting):
  results/ceiling_RWD_v2/results.json      exact posterior contraction c = 1 - var_post/var_prior for behaviour alone
                                           (n = 60..480 trials) and for behaviour + a read-out of the prediction error at
                                           coupling rho (n = 480)
  results/worth_ceiling_RWD_v2/results.json each decoder's null-corrected coupling rho_eff and its trial-worth
For a Gaussian posterior, the information gained about a parameter is  bits = -0.5 * log2(1 - c).
Reported per parameter and decoder: bits from behaviour alone (480 trials), bits with EEG at rho_eff, the EEG increment per
session and per trial, the same at the ceiling (largest rho on the curve), and the percentage of ceiling bits.
Writes results/eegbits/belief_bits.json.
"""
import json
from pathlib import Path
import numpy as np

P = Path("/path/to/eegaudit")
C = json.load(open(P / "results/ceiling_RWD_v2/results.json"))
W = json.load(open(P / "results/worth_ceiling_RWD_v2/results.json"))
N_TRIALS = 480
PARAMS = ["logit_alpha", "logit_d", "log_beta", "kappa"]
NAMES = {"logit_alpha": "learning rate", "logit_d": "forgetting/decay", "log_beta": "inverse temperature", "kappa": "stickiness"}


def bits(c):
    c = min(max(float(c), 0.0), 1 - 1e-9)
    return -0.5 * np.log2(1 - c)


def joint_contraction(param, rho):
    rhos = sorted(float(r) for r in C["joint"])
    cs = [C["joint"][str(r) if str(r) in C["joint"] else f"{r:g}"]["exact"][param]["contraction"] for r in rhos]
    return float(np.interp(rho, rhos, cs)), max(rhos), cs[-1]


def baseline(p):
    """Behaviour-only contraction on the SAME simulation batch as the joint curve: the joint curve at rho = 0
    (the separate behaviour[480] block comes from another batch and differs by ~0.05, which is why gains in
    worth_ceiling.py are measured along the joint curve)."""
    return C["joint"]["0.0"]["exact"][p]["contraction"]


out = {"n_trials": N_TRIALS, "model": C["model"], "baseline": "joint curve at rho=0 (same batch)",
       "behaviour_block_480_for_reference": {p: C["behaviour"][str(N_TRIALS)]["exact"][p]["contraction"] for p in PARAMS},
       "params": {}, "decoders": {}}
for p in PARAMS:
    c_b = baseline(p)
    _, rho_max, c_ceiling = joint_contraction(p, 10.0)
    out["params"][p] = dict(name=NAMES[p], behaviour_contraction=c_b, behaviour_bits=bits(c_b),
                            ceiling_rho=rho_max, ceiling_contraction=c_ceiling, ceiling_bits=bits(c_ceiling),
                            ceiling_increment_bits=bits(c_ceiling) - bits(c_b),
                            ceiling_increment_bits_per_trial=(bits(c_ceiling) - bits(c_b)) / N_TRIALS,
                            behaviour_bits_per_trial=bits(c_b) / N_TRIALS)
PRIMARY = "ens"   # pre-specified primary decoder (the science ensemble, cogeeg's headline decoder); others are secondary
out["primary_decoder"] = PRIMARY
for dec, d in W["decoders"].items():
    rho = d["rho_eff"]
    out["decoders"][dec] = {"rho_eff": rho, "rho_ci": d.get("rho_ci")}
    # propagate the coupling CI to bits for the learning rate
    if d.get("rho_ci"):
        lo, hi = d["rho_ci"][0], d["rho_ci"][1]
        cb = baseline("logit_alpha")
        out["decoders"][dec]["learning_rate_bits_ci"] = [bits(joint_contraction("logit_alpha", lo)[0]) - bits(cb), bits(joint_contraction("logit_alpha", hi)[0]) - bits(cb)]
    for p in PARAMS:
        c_b = baseline(p)
        c_j, _, _ = joint_contraction(p, rho)
        inc = bits(c_j) - bits(c_b)
        out["decoders"][dec][p] = dict(joint_contraction=c_j, joint_bits=bits(c_j), eeg_increment_bits=inc,
                                       eeg_increment_bits_per_trial=inc / N_TRIALS,
                                       pct_of_ceiling_bits=100 * inc / out["params"][p]["ceiling_increment_bits"] if out["params"][p]["ceiling_increment_bits"] > 0 else None,
                                       trial_worth_from_cogeeg=d[p]["trials"], pct_ceiling_from_cogeeg=d[p]["pct_ceiling"])
Path(P / "results/eegbits").mkdir(exist_ok=True, parents=True)
(P / "results/eegbits/belief_bits.json").write_text(json.dumps(out, indent=1))
print("behaviour-only bits (480 trials):", {p: round(v["behaviour_bits"], 3) for p, v in out["params"].items()})
print("ceiling EEG increment bits / session:", {p: round(v["ceiling_increment_bits"], 3) for p, v in out["params"].items()})
for dec in out["decoders"]:
    print(f"{dec:8s} rho_eff={out['decoders'][dec]['rho_eff']:.3f}  learning-rate: +{out['decoders'][dec]['logit_alpha']['eeg_increment_bits']:.3f} bits/session = {1e3*out['decoders'][dec]['logit_alpha']['eeg_increment_bits_per_trial']:.2f} mbit/trial, {out['decoders'][dec]['logit_alpha']['pct_of_ceiling_bits']:.0f}% of ceiling bits, worth {out['decoders'][dec]['logit_alpha']['trial_worth_from_cogeeg']:.0f} trials")
