"""Aggregate eegbits per-subject results into the paper's numbers and figures.

Inputs : results/eegbits/{embedding,raw}_sub-XX.json  (from run.py)
Outputs: results/eegbits/summary.json   (all numbers the paper quotes; register them with ledger.py)
         paper/figs/fig_bottleneck.pdf  (performance vs L, EEG vs controls, both inputs)
         paper/figs/fig_repetitions.pdf (bits per query vs repetitions k, raw input)
Light CPU; run via srun anyway (cluster rule).
"""
import glob, json
from pathlib import Path
import numpy as np

R = Path("/path/to/eegaudit/results/eegbits")
FIG = Path("/path/to/eegaudit/paper/figs")
FIG.mkdir(parents=True, exist_ok=True)


def load(input_type):
    return {Path(p).stem.split("_")[1]: json.load(open(p)) for p in sorted(glob.glob(str(R / f"{input_type}_sub-*.json")))}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    rng = np.random.default_rng(0)
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def curve(subjects, condition, metric, k=None):
    """mean (over subjects, and seeds for controls) of metric at each L; k = repetitions (raw) or None = full."""
    Ls = sorted({r["L"] for d in subjects.values() for r in d["rows"]})
    out = {}
    for L in Ls:
        vals = []
        for d in subjects.values():
            kk = max(d["ks"]) if k is None else k
            rs = [r for r in d["rows"] if r["L"] == L and r["condition"] == condition and r["k"] == kk]
            if rs:
                vals.append(np.mean([r[metric] for r in rs]))
        out[L] = (float(np.mean(vals)), ci95(vals), len(vals)) if vals else None
    return out


def bits_frac(subj, L):
    """Yield retained at width L in BITS (Wolpaw) relative to full width, k = all repetitions, eeg rows."""
    rows = [r for r in subj["rows"] if r["condition"] == "eeg"]; kmax = max(r["k"] for r in rows); Lmax = max(r["L"] for r in rows)
    bL = [r["bits"] for r in rows if r["L"] == L and r["k"] == kmax]; bF = [r["bits"] for r in rows if r["L"] == Lmax and r["k"] == kmax]
    return float(bL[0] / bF[0]) if bL and bF and bF[0] > 0 else float("nan")


def main():
    summary = {}
    for inp in ["embedding", "raw", "randenc", "trainedenc"]:     # randenc / trainedenc: encoder-vs-EEG control inputs (run.py)
        S = load(inp)
        if not S:
            continue
        subs = sorted(S)
        full_L = max(S[subs[0]]["Ls"])
        s = dict(n_subjects=len(subs), subjects=subs, full_L=full_L,
                 frac_at_L50_mean=float(np.mean([S[x]["frac_at_L50"] for x in subs])),
                 frac_at_L50_ci=ci95([S[x]["frac_at_L50"] for x in subs]),
                 frac_at_L50_per_subject={x: S[x]["frac_at_L50"] for x in subs},
                 frac_at_L25_mean=float(np.mean([S[x]["frac_by_L"]["25"] for x in subs])),
                 frac_bits_at_L25_mean=float(np.mean([bits_frac(S[x], 25) for x in subs])), frac_bits_at_L50_mean=float(np.mean([bits_frac(S[x], 50) for x in subs])),
                 L90_per_subject={x: S[x]["L90"] for x in subs},
                 L90_median=float(np.median([S[x]["L90"] for x in subs if S[x]["L90"]])),
                 Lstar_per_subject={x: S[x]["Lstar_top1_vs_controls"] for x in subs},
                 Lstar_le10_count=int(sum(1 for x in subs if S[x]["Lstar_top1_vs_controls"] and S[x]["Lstar_top1_vs_controls"] <= 10)),
                 Lstar_le25_count=int(sum(1 for x in subs if S[x]["Lstar_top1_vs_controls"] and S[x]["Lstar_top1_vs_controls"] <= 25)),
                 full_top1_mean=float(np.mean([S[x]["full_width_eeg"]["top1"] for x in subs])),
                 full_two_way_mean=float(np.mean([S[x]["full_width_eeg"]["two_way"] for x in subs])),
                 full_bits_mean=float(np.mean([S[x]["full_width_eeg"]["bits"] for x in subs])),
                 curves={c: {m: curve(S, c, m) for m in ["top1", "two_way", "bits", "cosine"]} for c in ["eeg", "noise", "trial_shuffled", "shuffled_pair"]})
        s["control_two_way_max"] = float(max(v[0] for c in ["noise", "trial_shuffled", "shuffled_pair"] for v in s["curves"][c]["two_way"].values() if v))
        s["noise_cosine_full"] = s["curves"]["noise"]["cosine"][full_L][0] if s["curves"]["noise"]["cosine"].get(full_L) else None
        if inp == "raw":
            ks = S[subs[0]]["ks"]
            s["ks"] = ks
            s["bits_by_k_fullL"] = {str(k): curve(S, "eeg", "bits", k)[full_L] for k in ks}
            s["two_way_by_k_fullL"] = {str(k): curve(S, "eeg", "two_way", k)[full_L] for k in ks}
            s["top1_by_k_fullL"] = {str(k): curve(S, "eeg", "top1", k)[full_L] for k in ks}
        summary[inp] = s
    (R / "summary.json").write_text(json.dumps(summary, indent=1))

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    import figstyle as F
    F.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(summary), figsize=(3.6 * len(summary), 3.0), squeeze=False, sharey=True)
    for ax, (inp, s) in zip(axes[0], summary.items()):
        for c in ["eeg", "noise", "trial_shuffled", "shuffled_pair"]:
            cv = s["curves"][c]["two_way"]
            Ls = [L for L in sorted(cv) if cv[L]]
            y = [100 * cv[L][0] for L in Ls]
            role = inp if c == "eeg" else c
            ci = ([100 * cv[L][1][0] for L in Ls], [100 * cv[L][1][1] for L in Ls]) if c == "eeg" else None
            F.line(ax, Ls, y, role, label=F.LABEL["eeg"] + f" ({F.LABEL[inp]})" if c == "eeg" else F.LABEL[c], ci=ci)
            if c == "eeg":
                F.end_label(ax, Ls, y, "EEG", color=F.INK)
        F.end_label(ax, Ls, [50.3], "3 controls", color=F.INK2)
        ax.axhline(50, color=F.INK2, lw=0.8, zorder=1)
        ax.text(min(Ls), 51.0, "chance", fontsize=7, color=F.INK2)
        ax.set_xscale("log"); ax.set_xlabel("bottleneck width L (linear dimensions)")
        ax.set_title(f"{F.LABEL[inp]}, {s['n_subjects']} participants")
        ax.set_ylim(45, 100); ax.set_xlim(1.6, max(Ls) * 4)
    axes[0][0].set_ylabel("2-way identification (%)")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout(); F.save(fig, "fig_bottleneck", FIG)
    if "raw" in summary:
        s = summary["raw"]; ks = s["ks"]
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        y = [s["bits_by_k_fullL"][str(k)][0] for k in ks]; lo = [s["bits_by_k_fullL"][str(k)][1][0] for k in ks]; hi = [s["bits_by_k_fullL"][str(k)][1][1] for k in ks]
        F.line(ax, ks, y, "raw", label="raw EEG, full width", ci=(lo, hi))
        ax.set_xscale("log"); ax.set_xlabel("repetitions averaged per query (k)"); ax.set_ylabel("yield per query (units)")
        ax.set_title("more repetitions, more yield")
        fig.tight_layout(); F.save(fig, "fig_repetitions", FIG)
    for inp, s in summary.items():
        print(f"{inp}: n={s['n_subjects']} frac@L50={s['frac_at_L50_mean']:.3f} {s['frac_at_L50_ci']} frac@L25={s['frac_at_L25_mean']:.3f} L90 median={s['L90_median']} L*<=10: {s['Lstar_le10_count']}/{s['n_subjects']} full top1={100*s['full_top1_mean']:.1f}% 2-way={100*s['full_two_way_mean']:.1f}% bits={s['full_bits_mean']:.2f} ctrl 2-way max={100*s['control_two_way_max']:.1f}%")
        if inp == "raw":
            print("  bits by k:", {k: round(v[0], 3) for k, v in s["bits_by_k_fullL"].items()}, " 2-way by k:", {k: round(100 * v[0], 1) for k, v in s["two_way_by_k_fullL"].items()})


if __name__ == "__main__":
    main()
