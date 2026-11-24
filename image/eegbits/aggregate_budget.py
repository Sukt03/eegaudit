"""Aggregate eegbits budget results (run_budget.py) into paper numbers and two figures.

Inputs : results/eegbits/budget_{embedding,raw}_sub-XX.json
Outputs: results/eegbits/budget_summary.json
         paper/figs/fig_bits_budget.pdf   (bits out vs bits in; quantised bottleneck; both inputs)
         paper/figs/fig_acquisition.pdf   (bits per query and per second vs electrodes and window; raw)
"""
import glob, json
from pathlib import Path
import numpy as np

R = Path("/path/to/eegaudit/results/eegbits")
FIG = Path("/path/to/eegaudit/paper/figs")


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean())] * 2
    rng = np.random.default_rng(0)
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def agg(rows_by_subject, key_fn, metric):
    out = {}
    keys = sorted({key_fn(r) for rows in rows_by_subject.values() for r in rows if key_fn(r) is not None}, key=lambda k: (str(type(k)), k))
    for k in keys:
        vals = [np.mean([r[metric] for r in rows if key_fn(r) == k]) for rows in rows_by_subject.values() if any(key_fn(r) == k for r in rows)]
        out[str(k)] = dict(mean=float(np.mean(vals)), ci=ci95(vals), n=len(vals))
    return out


def main():
    S = {}
    for inp in ["embedding", "raw"]:
        d = {Path(p).stem.split("_")[-1]: json.load(open(p)) for p in sorted(glob.glob(str(R / f"budget_{inp}_sub-*.json")))}
        if d:
            S[inp] = d
    summary = {}
    for inp, d in S.items():
        kmax = max(int(r["k"]) for s in d.values() for r in s["rows"])
        q = {s: [r for r in v["rows"] if r["part"] == "quant" and r["k"] == kmax] for s, v in d.items()}
        # bits out (Wolpaw, 200-way) and 2-way vs budget bits, for quantised and float32
        quant = agg(q, lambda r: r["budget_bits"] if r["condition"] == "eeg_quant" else None, "bits")
        quant2 = agg(q, lambda r: r["budget_bits"] if r["condition"] == "eeg_quant" else None, "two_way")
        flt = agg(q, lambda r: r["L"] if r["condition"] == "eeg_float32" else None, "two_way")
        noise = agg(q, lambda r: r["L"] if r["condition"] == "noise_float32" else None, "two_way")
        full2 = max(v["mean"] for v in flt.values())
        # smallest budget reaching 90% of the float32 full-width chance-adjusted 2-way, per subject
        per_subj = {}
        for s, rows in q.items():
            # reference = the 1024-D full-width float32 result of the same subject (results/eegbits/<inp>_sub-XX.json, k = all reps),
            # not the best point of the budget grid (code review 2026-11-20)
            fw = json.load(open(R / f"{inp}_{s}.json"))["rows"] if (R / f"{inp}_{s}.json").exists() else None
            if fw:
                e = [r for r in fw if r["condition"] == "eeg"]; Lmax = max(r["L"] for r in e); kmax_fw = max(r["k"] for r in e)
                f = [r for r in e if r["L"] == Lmax and r["k"] == kmax_fw][0]["two_way"] - 0.5
            else:
                f = max(r["two_way"] for r in rows if r["condition"] == "eeg_float32") - 0.5
            cands = sorted([r for r in rows if r["condition"] == "eeg_quant" and (r["two_way"] - 0.5) >= 0.9 * f], key=lambda r: r["budget_bits"])
            per_subj[s] = cands[0]["budget_bits"] if cands else None
        rr = {s: [r for r in v["rows"] if r["part"] == "rrr" and r["k"] == kmax] for s, v in d.items()}
        rrr = agg(rr, lambda r: r["L"] if r["condition"] == "eeg_rrr" else None, "two_way")
        rrr_noise = agg(rr, lambda r: r["L"] if r["condition"] == "noise_rrr" else None, "two_way")
        rrr_bits = agg(rr, lambda r: r["L"] if r["condition"] == "eeg_rrr" else None, "bits")
        summary[inp] = dict(n_subjects=len(d), k=kmax, bits_out_by_budget=quant, two_way_by_budget=quant2, two_way_float32_by_L=flt,
                            two_way_rrr_by_L=rrr, two_way_rrr_noise_by_L=rrr_noise, bits_rrr_by_L=rrr_bits,
                            two_way_noise_by_L=noise, budget90_per_subject=per_subj,
                            budget90_median=float(np.median([v for v in per_subj.values() if v])) if any(per_subj.values()) else None)
        if inp == "raw":
            acq = {s: [r for r in v["rows"] if r["part"] == "acq"] for s, v in d.items()}
            summary[inp]["acq_bits_per_query"] = agg(acq, lambda r: f"{r['channels']}|{r['window']}|k{r['k']}", "bits")
            summary[inp]["acq_bits_per_second"] = agg(acq, lambda r: f"{r['channels']}|{r['window']}|k{r['k']}", "bits_per_second")
            summary[inp]["acq_two_way"] = agg(acq, lambda r: f"{r['channels']}|{r['window']}|k{r['k']}", "two_way")
    (R / "budget_summary.json").write_text(json.dumps(summary, indent=1))

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    import figstyle as F
    F.setup()
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    # Figure: yield vs units of signal handed to the decoder (quantised budget), both inputs, with the informed bottleneck
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for inp, s in summary.items():
        b = sorted(int(k) for k in s["two_way_by_budget"])
        y = [100 * s["two_way_by_budget"][str(k)]["mean"] for k in b]
        ci = ([100 * s["two_way_by_budget"][str(k)]["ci"][0] for k in b], [100 * s["two_way_by_budget"][str(k)]["ci"][1] for k in b])
        F.line(ax, b, y, inp, label=f"{F.LABEL[inp]}, quantised units", ci=ci)
        F.end_label(ax, b, y, F.LABEL[inp], color=F.INK)
        if s.get("two_way_rrr_by_L"):
            Ls = sorted(int(k) for k in s["two_way_rrr_by_L"]); yr = [100 * s["two_way_rrr_by_L"][str(L)]["mean"] for L in Ls]
            ax.plot([32 * L for L in Ls], yr, ls=(0, (3, 1.5)), color=F.ROLE[inp], lw=1.6, marker="s", ms=4, label=f"{F.LABEL[inp]}, informed bottleneck (float32)")
    ax.axhline(50, color=F.INK2, lw=0.8, zorder=1); ax.text(ax.get_xlim()[0] * 1.5 if ax.get_xlim()[0] > 0 else 5, 51, "chance", fontsize=7, color=F.INK2)
    ax.set_xscale("log"); ax.set_xlabel("units of signal per trial given to the decoder (L × b)"); ax.set_ylabel("2-way identification (%)")
    ax.set_ylim(45, 100); ax.legend(loc="lower right"); ax.set_title("the price of a decoder's input")
    fig.tight_layout(); F.save(fig, "fig_bits_budget", FIG)
    # Figure: acquisition budget (raw): yield per query and per second vs electrodes, one line per window
    if "raw" in summary and summary["raw"].get("acq_bits_per_query"):
        s = summary["raw"]; fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        chans = ["occpar8", "occpar16", "occpar32", "all63"]; rand = ["random8", "random16", "random32"]
        wins = ["0-0.2s", "0-0.4s", "0-0.6s", "0-1.0s"]; wcol = [F.SERIES["blue"], F.SERIES["aqua"], F.SERIES["violet"], F.SERIES["orange"]]
        for ax, metric, lab, kk in ((axes[0], "acq_bits_per_query", "yield per query, k = 80 (units)", "k80"), (axes[1], "acq_bits_per_second", "yield per second of EEG, k = 1 (units/s)", "k1")):
            for w, col in zip(wins, wcol):
                xs, ys = [], []
                for c, n in zip(chans, [8, 16, 32, 63]):
                    key = f"{c}|{w}|{kk}"
                    if key in s[metric]:
                        xs.append(n); ys.append(s[metric][key]["mean"])
                if xs:
                    ax.plot(xs, ys, "-o", color=col, label=f"occipito-parietal, {w}")
                    F.end_label(ax, xs, ys, w, color=F.INK2)
            xs, ys = [], []
            for c, n in zip(rand, [8, 16, 32]):
                key = f"{c}|0-1.0s|{kk}"
                if key in s[metric]:
                    xs.append(n); ys.append(s[metric][key]["mean"])
            if xs:
                ax.plot(xs, ys, ls=(0, (4, 2)), color=F.GREY1, marker="s", ms=4, label="random electrodes, 0-1.0 s")
            ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 63]); ax.set_xticklabels(["8", "16", "32", "63"])
            ax.set_xlabel("electrodes"); ax.set_ylabel(lab)
        axes[0].legend(loc="upper left"); axes[0].set_title("what electrodes and time buy")
        fig.tight_layout(); F.save(fig, "fig_acquisition", FIG)
    for inp, s in summary.items():
        print(inp, "n=", s["n_subjects"], "budget90 median bits:", s["budget90_median"], "| 2-way by budget:", {k: round(100 * v["mean"], 1) for k, v in list(s["two_way_by_budget"].items())[:8]})
        if inp == "raw" and s.get("acq_bits_per_second"):
            best = sorted(s["acq_bits_per_second"].items(), key=lambda kv: -kv[1]["mean"])[:3]
            print("  top bits/s configs:", [(k, round(v["mean"], 3)) for k, v in best])


if __name__ == "__main__":
    main()
