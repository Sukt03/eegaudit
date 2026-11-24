"""TRACK 1 (decision EEG): price list in bits for every decoder on the ladder, both datasets.

Analysis of EXISTING results only (no fitting, no decoding). Reuses the conversion of experiments/eegbits/belief_bits.py:
  bits(c) = -0.5 * log2(1 - c), c = exact posterior contraction 1 - var_post/var_prior
  baseline = joint curve at rho = 0 (same simulation batch as the joint curve)
  EEG increment = bits(c_joint(rho_eff)) - bits(c_joint(0)), with c_joint linearly interpolated over the exact curve
  CI = the same map applied to rho_ci (random-effects CI of the null-corrected coupling, group_stats.py)
and the trial-worth rule of worth_ceiling.py (behavioural trials beyond n0 giving the same contraction gain).

Inputs (read only)
  ds003458: results/ceiling_RWD_v2/results.json (joint curve at n0 = 240 EEG trials; the joint rho = 0 contraction equals
            behaviour[240]), n0 variants results/ceiling_RWD_v2_n0_{60,120,180,360}, couplings results/group_stats/results.json
  ds004595: results/ceiling_RWD_ds004595/results.json (n0 = 240), variant ..._n0_180, couplings results/group_stats_ds004595
Outputs: results/track_decision/price_list.json, fig_decision_ladder.{pdf,png}
"""
import json
from pathlib import Path
import numpy as np

P = Path("/path/to/eegaudit")
OUT = P / "results/track_decision"
PARAMS = ["logit_alpha", "logit_d", "log_beta", "kappa"]
NAMES = {"logit_alpha": "learning rate", "logit_d": "forgetting/decay", "log_beta": "inverse temperature", "kappa": "stickiness"}
PRIMARY = "ens"  # pre-specified primary decoder (science ensemble), as in belief_bits.py
LABEL = {"raw": "raw ERP (ridge)", "ens": "science ensemble", "interior": "interior channels", "fm": "CBraMod pooled (frozen)",
         "fmtok": "CBraMod tokens (frozen)", "labram": "LaBraM (frozen)", "eegpt": "EEGPT (frozen)", "eegnet": "EEGNet LOPO",
         "eegnet_post": "EEGNet LOPO, post-feedback", "cbramod_ft": "CBraMod fine-tuned LOPO", "reve": "REVE pooled (frozen)",
         "revetok": "REVE tokens (frozen)", "reve_lopo": "REVE tokens, LOPO ridge", "rewp_lopo": "RewP component LOPO",
         "p3_lopo": "P3 component LOPO"}
DATASETS = {
    "ds003458": dict(n_participants=23, session_trials=480, base="results/ceiling_RWD_v2/results.json", base_n0=240,
                     n0_variants={60: "results/ceiling_RWD_v2_n0_60/results.json", 120: "results/ceiling_RWD_v2_n0_120/results.json",
                                  180: "results/ceiling_RWD_v2_n0_180/results.json", 240: "results/ceiling_RWD_v2/results.json",
                                  360: "results/ceiling_RWD_v2_n0_360/results.json"},
                     group_stats="results/group_stats/results.json", cogeeg_worth="results/worth_ceiling_RWD_v2/results.json"),
    "ds004595": dict(n_participants=50, session_trials=350, base="results/ceiling_RWD_ds004595/results.json", base_n0=240,
                     n0_variants={180: "results/ceiling_RWD_ds004595_n0_180/results.json", 240: "results/ceiling_RWD_ds004595/results.json"},
                     group_stats="results/group_stats_ds004595/results.json", cogeeg_worth=None),
}


def bits(c):
    c = min(max(float(c), 0.0), 1 - 1e-9)
    return -0.5 * np.log2(1 - c)


class Curve:
    def __init__(self, path):
        C = json.load(open(P / path))
        self.rhos = np.array(sorted(float(r) for r in C["joint"]))
        self.c = {p: np.array([C["joint"][f"{r:g}" if f"{r:g}" in C["joint"] else str(r)]["exact"][p]["contraction"] for r in self.rhos]) for p in PARAMS}
        self.ns = np.array(sorted(int(n) for n in C["behaviour"]))
        self.beh = {p: np.array([C["behaviour"][str(n)]["exact"][p]["contraction"] for n in self.ns]) for p in PARAMS}
        self.rho_max = float(self.rhos[-1])

    def joint(self, p, rho):
        return float(np.interp(max(rho, 0.0), self.rhos, self.c[p]))

    def base(self, p):
        return float(self.c[p][0])

    def inc_bits(self, p, rho):
        return bits(self.joint(p, rho)) - bits(self.base(p))

    def ceiling_inc_bits(self, p):
        return bits(self.c[p][-1]) - bits(self.base(p))

    def trials(self, p, rho, n0):
        """worth_ceiling.py rule: behavioural trials beyond n0 whose contraction gain equals the EEG gain (None = off the grid)."""
        gain = self.joint(p, rho) - self.base(p)
        yb = np.maximum.accumulate(self.beh[p]); y0 = float(np.interp(n0, self.ns, yb)); yt = y0 + gain
        if gain <= 0:
            return 0.0
        return float(np.exp(np.interp(yt, yb, np.log(self.ns))) - n0) if yt < yb[-1] else None


def null_r(dec):
    if "null_z" in dec:
        z = np.array(dec["null_z"], dtype=float)  # participants x (N-1) session-permutation nulls, Fisher z
        src = "exact session-permutation nulls (group_stats null_z)"
    else:
        # group_stats_ds004595 stores only d_i = atanh(r_i) - mean_j atanh(r_ij) and v_i = var_j atanh(r_ij);
        # rebuild a Gaussian null per participant (mean, sd) with N-1 draws, fixed seed
        mu = np.arctanh(np.array(dec["r"])) - np.array(dec["d"]); sd = np.sqrt(np.array(dec["v"]))
        z = mu[:, None] + sd[:, None] * np.random.default_rng(0).standard_normal((len(mu), len(mu) - 1))
        src = "Gaussian reconstruction from per-participant null mean (atanh r - d) and variance v; p95 is approximate"
    r = np.tanh(z)
    return dict(mean_r=float(r.mean()), p95_r=float(np.percentile(r, 95)), p5_r=float(np.percentile(r, 5)), sd_r=float(r.std()),
                mean_abs_r=float(np.abs(r).mean()), n_null=int(r.size), source=src)


def price(ds, cfg):
    base = Curve(cfg["base"]); n0 = cfg["base_n0"]
    # sanity: the joint curve's rho = 0 point is behaviour at n0
    assert abs(base.base("logit_alpha") - float(np.interp(n0, base.ns, base.beh["logit_alpha"]))) < 0.005, "n0 mismatch"
    G = json.load(open(P / cfg["group_stats"]))["decoders"]
    out = dict(n_participants=cfg["n_participants"], session_trials=cfg["session_trials"], eeg_trials_in_joint_curve=n0,
               ceiling_file=cfg["base"], group_stats_file=cfg["group_stats"], baseline="joint curve at rho=0 (same batch)",
               ceiling_rho=base.rho_max, params={}, decoders={}, controls={}, trials_axis={})
    for p in PARAMS:
        cb = base.base(p)
        out["params"][p] = dict(name=NAMES[p], behaviour_contraction_n0=cb, behaviour_bits_n0=bits(cb), behaviour_bits_per_trial=bits(cb) / n0,
                                ceiling_contraction=float(base.c[p][-1]), ceiling_bits=bits(base.c[p][-1]),
                                ceiling_increment_bits=base.ceiling_inc_bits(p), ceiling_increment_bits_per_trial=base.ceiling_inc_bits(p) / n0,
                                ceiling_trial_worth=base.trials(p, base.rho_max, n0))
    W = json.load(open(P / cfg["cogeeg_worth"]))["decoders"] if cfg["cogeeg_worth"] else {}
    for dec, g in G.items():
        rho, ci = g["rho_eff_pooled"], g["rho_eff_ci"]
        d = dict(label=LABEL.get(dec, dec), rho_eff=rho, rho_ci=ci, meta_z=g["meta"]["mu"], meta_z_ci=g["meta"]["ci"], p_meta=g["meta"]["p"],
                 n_subjects=len(g["r"]), r_median=float(np.median(g["r"])), r_mean=float(np.mean(g["r"])))
        for p in PARAMS:
            inc = base.inc_bits(p, rho); ceil = base.ceiling_inc_bits(p)
            lo, hi = base.inc_bits(p, ci[0]), base.inc_bits(p, ci[1])
            gain = base.joint(p, rho) - base.base(p); cgain = float(base.c[p][-1] - base.base(p))
            e = dict(joint_contraction=base.joint(p, rho), joint_bits=bits(base.joint(p, rho)),
                     eeg_increment_bits=inc, eeg_increment_bits_ci=[lo, hi],
                     eeg_increment_bits_per_trial=inc / n0, eeg_increment_bits_per_trial_ci=[lo / n0, hi / n0],
                     trial_worth=base.trials(p, rho, n0), trial_worth_ci=[base.trials(p, ci[0], n0), base.trials(p, ci[1], n0)],
                     pct_ceiling_contraction=100 * gain / cgain, pct_ceiling_contraction_ci=[100 * (base.joint(p, c) - base.base(p)) / cgain for c in ci],
                     efficiency_bits=inc / ceil, efficiency_bits_ci=[lo / ceil, hi / ceil],
                     efficiency_total=bits(base.joint(p, rho)) / bits(base.c[p][-1]))
            if dec in W:  # cross-check against cogeeg's worth_ceiling.py numbers
                e["cogeeg_trials"] = W[dec][p]["trials"]; e["cogeeg_pct_ceiling"] = W[dec][p]["pct_ceiling"]
            d[p] = e
        d["bits_per_session_primary"] = d["logit_alpha"]["eeg_increment_bits"]
        out["decoders"][dec] = d
        nr = null_r(g)
        out["controls"][dec] = dict(label=LABEL.get(dec, dec), session_permutation_null=nr,
                                    null_bits_learning_rate=dict(at_mean_r=base.inc_bits("logit_alpha", nr["mean_r"]), at_p95_r=base.inc_bits("logit_alpha", nr["p95_r"])),
                                    null_trial_worth_learning_rate_at_p95=base.trials("logit_alpha", nr["p95_r"], n0),
                                    prevalence=g["prevalence"], n_individually_above_95pct_null=g["n_individually_above_95pct_null"],
                                    min_r=g["prevalence"]["min_r"])
    out["primary_decoder"] = PRIMARY
    # trials axis: behaviour-only bits vs n, and the EEG increment at every available n0
    out["trials_axis"]["behaviour_only_bits"] = {p: {int(n): bits(c) for n, c in zip(base.ns, base.beh[p])} for p in PARAMS}
    out["trials_axis"]["behaviour_only_contraction"] = {p: {int(n): float(c) for n, c in zip(base.ns, base.beh[p])} for p in PARAMS}
    out["trials_axis"]["eeg_increment_by_n0"] = {}
    for n0v, path in sorted(cfg["n0_variants"].items()):
        cv = Curve(path); row = dict(file=path, baseline_contraction={p: cv.base(p) for p in PARAMS},
                                     ceiling_increment_bits={p: cv.ceiling_inc_bits(p) for p in PARAMS}, decoders={})
        for dec, g in G.items():
            rho, ci = g["rho_eff_pooled"], g["rho_eff_ci"]
            row["decoders"][dec] = {p: dict(eeg_increment_bits=cv.inc_bits(p, rho), eeg_increment_bits_ci=[cv.inc_bits(p, ci[0]), cv.inc_bits(p, ci[1])],
                                            eeg_increment_bits_per_trial=cv.inc_bits(p, rho) / n0v, trial_worth=cv.trials(p, rho, n0v),
                                            efficiency_bits=cv.inc_bits(p, rho) / cv.ceiling_inc_bits(p)) for p in PARAMS}
        out["trials_axis"]["eeg_increment_by_n0"][int(n0v)] = row
    return out


def figure(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 1, "lines.linewidth": 2})
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    nmax = max(len(R["datasets"][d]["decoders"]) for d in R["datasets"])
    for ax, ds in zip(axs, ["ds003458", "ds004595"]):
        D = R["datasets"][ds]; decs = sorted(D["decoders"], key=lambda k: D["decoders"][k]["logit_alpha"]["eeg_increment_bits"])
        y = np.arange(len(decs))
        vals = np.array([D["decoders"][k]["logit_alpha"]["eeg_increment_bits"] for k in decs])
        lo = np.array([D["decoders"][k]["logit_alpha"]["eeg_increment_bits_ci"][0] for k in decs])
        hi = np.array([D["decoders"][k]["logit_alpha"]["eeg_increment_bits_ci"][1] for k in decs])
        floor = max(D["controls"][k]["null_bits_learning_rate"]["at_p95_r"] for k in decs)
        ceil = D["params"]["logit_alpha"]["ceiling_increment_bits"]
        ax.axvspan(0, floor, color="0.75", lw=0, zorder=0, label=f"noise floor: null 95th pct ({floor:.3f} bits)")
        ax.axvline(ceil, color="0.25", ls="--", lw=2, label=f"exact ceiling, perfect read-out ({ceil:.2f} bits)")
        cols = ["#c0392b" if k == PRIMARY else ("#2c6fbb" if "lopo" in k or k in ("eegnet", "eegnet_post", "cbramod_ft") else "#555555") for k in decs]
        ax.hlines(y, lo, hi, color=cols, lw=2)
        ax.scatter(vals, y, s=28, color=cols, zorder=3)
        for yi, v, h in zip(y, vals, hi):
            ax.text(h + 0.02 * ceil, yi, f"{v:.2f}", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels([D["decoders"][k]["label"] for k in decs])
        ax.set_xlim(0, ceil * 1.12); ax.set_ylim(-0.7, nmax - 0.3)
        ax.set_xlabel("EEG yield for the learning rate\n(bits per session, 240 EEG trials)")
        ax.set_title(f"{ds} (n = {D['n_participants']}, {D['session_trials']} trials)", loc="left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(width=1)
        ax.legend(loc="upper left", bbox_to_anchor=(0.08, 0.99), frameon=False, fontsize=7) if ds == "ds004595" else None
    fig.text(0.01, 0.012, "Red: pre-specified primary decoder (science ensemble). Blue: decoders trained leave-one-participant-out. Grey: fixed features or frozen foundation models.\n"
             "Bars: 95% CI of the null-corrected coupling (random-effects, group_stats.py) mapped through the exact posterior curve. Rungs sorted by yield.",
             fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_decision_ladder.pdf"); fig.savefig(OUT / "fig_decision_ladder.png", dpi=200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(model="RWD", bits_formula="-0.5*log2(1-c)", baseline="joint curve at rho=0 (same simulation batch)",
             note_n0="joint curves were computed with EEG read-outs on the first n0=240 trials (exp_rate.N0 default); bits per trial divide by 240, "
                     "not by the 480-trial session that belief_bits.py divides by", primary_decoder=PRIMARY, datasets={})
    for ds, cfg in DATASETS.items():
        R["datasets"][ds] = price(ds, cfg)
    # cross-check against cogeeg worth_ceiling_RWD_v2 (same group_stats, same curve)
    D = R["datasets"]["ds003458"]; mx = 0.0
    for dec, d in D["decoders"].items():
        for p in PARAMS:
            if "cogeeg_trials" in d[p] and d[p]["cogeeg_trials"] is not None and d[p]["trial_worth"] is not None:
                mx = max(mx, abs(d[p]["cogeeg_trials"] - d[p]["trial_worth"]))
    R["crosscheck_max_abs_trial_diff_vs_worth_ceiling_RWD_v2"] = mx
    (OUT / "price_list.json").write_text(json.dumps(R, indent=1))
    figure(R)
    print("cross-check vs worth_ceiling_RWD_v2, max |trial diff| =", round(mx, 3))
    for ds, D in R["datasets"].items():
        print(f"\n== {ds}  behaviour-only bits (learning rate, n0=240): {D['params']['logit_alpha']['behaviour_bits_n0']:.3f}; "
              f"ceiling increment {D['params']['logit_alpha']['ceiling_increment_bits']:.3f} bits")
        for dec in sorted(D["decoders"], key=lambda k: -D["decoders"][k]["logit_alpha"]["eeg_increment_bits"]):
            d = D["decoders"][dec]; a = d["logit_alpha"]; c = D["controls"][dec]
            tw = "off-grid" if a["trial_worth"] is None else f"{a['trial_worth']:.0f}"
            print(f"{dec:12s} rho={d['rho_eff']:.3f} [{d['rho_ci'][0]:.3f},{d['rho_ci'][1]:.3f}]  LR +{a['eeg_increment_bits']:.3f} [{a['eeg_increment_bits_ci'][0]:.3f},{a['eeg_increment_bits_ci'][1]:.3f}] bits/session "
                  f"= {1e3*a['eeg_increment_bits_per_trial']:.2f} mbit/trial  worth {tw} trials  {a['pct_ceiling_contraction']:.1f}% ceiling(contraction)  eff {a['efficiency_bits']:.3f}  "
                  f"null r mean {c['session_permutation_null']['mean_r']:+.3f} p95 {c['session_permutation_null']['p95_r']:.3f} -> {c['null_bits_learning_rate']['at_p95_r']:.4f} bits  gamma0 {c['prevalence']['gamma0']}")
        print("trials axis, behaviour-only LR bits:", {n: round(b, 3) for n, b in D["trials_axis"]["behaviour_only_bits"]["logit_alpha"].items()})
        for n0, row in D["trials_axis"]["eeg_increment_by_n0"].items():
            print(f"  n0={n0}: ens +{row['decoders']['ens']['logit_alpha']['eeg_increment_bits']:.3f} bits, ceiling {row['ceiling_increment_bits']['logit_alpha']:.3f}, "
                  f"ens worth {row['decoders']['ens']['logit_alpha']['trial_worth']}")


if __name__ == "__main__":
    main()
