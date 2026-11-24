"""Track 2 (EEG->image) aggregation: results/eegbits/acq_*.json -> results/track_image/summary.json + fig_repetition_ladder.{pdf,png}.
Per decoder (ATM-S, NICE, NICE-GA): mean and bootstrap 95% CI over subjects (seeds averaged within subject first) of top-1, 2-way,
bits per query (Wolpaw, 200-way) and bits per second (bits / (k x 0.2 s)) at each k; window curve at k=80; electrode subsets at k=80;
native; split-half ceiling; normalised efficiency = bits / log2(200). Light CPU; run through srun on a compute node (cluster rule)."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
import figstyle as fs

R = Path("/path/to/eegaudit/results/eegbits")
OUT = Path("/path/to/eegaudit/results/track_image"); OUT.mkdir(parents=True, exist_ok=True)
N, SOA = 200, 0.2
LOG2N = float(np.log2(N))
KS = [1, 2, 4, 8, 16, 32, 80]
WINDOWS = [0.2, 0.4, 0.6, 1.0]
METRICS = ["top1", "two_way", "bits"]
DECODERS = {"ATM-S": "acq_atm_sub-*.json", "NICE": "acq_nice_sub-*_seed*.json", "NICE-GA": "acq_nice_ga_sub-*_seed*.json"}
COLOR = {"ATM-S": fs.SERIES["blue"], "NICE": fs.SERIES["orange"], "NICE-GA": fs.SERIES["aqua"]}
MARK = {"ATM-S": "o", "NICE": "s", "NICE-GA": "D"}


FILL = sys.argv[1] if len(sys.argv) > 1 else "zero"     # "zero" (original files) or "mean_loo" (audit-D in-distribution fill files)
SUFFIX = "" if FILL == "zero" else f"_{FILL}"


def load(pattern):
    runs = []
    for p in sorted(glob.glob(str(R / pattern))):
        if p.endswith("_summary.json"):
            continue
        if FILL == "zero" and "_mean_loo" in p:
            continue
        if FILL != "zero" and not p.endswith(f"{SUFFIX}.json"):
            continue
        d = json.load(open(p))
        runs.append(dict(path=p, subject=d["subject"], seed=int(d.get("seed", 0)), rows=d["rows"]))
    return runs


def sel(rows, axis, k=None, channels=None, window=None):
    out = [r for r in rows if r["axis"] == axis]
    if k is not None: out = [r for r in out if r["k"] == k]
    if channels is not None: out = [r for r in out if r["channels"] == channels]
    if window is not None: out = [r for r in out if abs(r["window"] - window) < 1e-9]
    return out


def ci95(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def agg(runs, axis, k=None, channels=None, window=None, bps_k=None):
    """per run: mean over draws (and over listed channel subsets); per subject: mean over seeds; CI: bootstrap over subjects."""
    rng = np.random.default_rng(0)
    chans = channels if isinstance(channels, list) else [channels]
    per_sub = {}
    for r in runs:
        rows = [x for c in chans for x in sel(r["rows"], axis, k, c, window)]
        if not rows:
            continue
        per_sub.setdefault(r["subject"], []).append({m: float(np.mean([x[m] for x in rows])) for m in METRICS})
    if not per_sub:
        return None
    subs = sorted(per_sub)
    vals = {m: np.array([np.mean([v[m] for v in per_sub[s]]) for s in subs]) for m in METRICS}
    out = dict(n_subjects=len(subs), n_runs=sum(len(v) for v in per_sub.values()), subjects=subs)
    for m in METRICS:
        out[m] = dict(mean=float(vals[m].mean()), ci95=ci95(vals[m], rng), per_subject=[float(v) for v in vals[m]])
    eff = vals["bits"] / LOG2N
    out["efficiency"] = dict(mean=float(eff.mean()), ci95=ci95(eff, rng))
    if bps_k is not None:
        bps = vals["bits"] / (bps_k * SOA)
        out["bits_per_second"] = dict(mean=float(bps.mean()), ci95=ci95(bps, rng), per_subject=[float(v) for v in bps])
    return out


summary = dict(track="image", n_candidates=N, stimulus_time_s=SOA, log2N=LOG2N,
               chance=dict(top1=1 / N, two_way=0.5, bits=0.0),
               note="EEG test set of THINGS-EEG2 (200 images x 80 repetitions, 63 ch x 250 samples at 250 Hz). Mean over subjects; "
                    "seeds averaged within subject first; 95% CI = percentile bootstrap over subjects (2000 draws). bits = Wolpaw I(P,200). "
                    "bits/s = bits / (k x 0.2 s). Channel subsets and windows by zero-masking before the frozen encoder (approximation). "
                    "No noise / trial-shuffled / shuffled-pair controls were run for these three decoders; chance = 1/200.",
               decoders={})
for name, pat in DECODERS.items():
    runs = load(pat)
    if not runs:
        summary["decoders"][name] = dict(status="no result files", files=[]); continue
    D = dict(files=[r["path"] for r in runs], n_subjects=len({r["subject"] for r in runs}), n_seeds=len({r["seed"] for r in runs}),
             subjects=sorted({r["subject"] for r in runs}), seeds=sorted({r["seed"] for r in runs}))
    D["native"] = agg(runs, "native", 80, "all63", 1.0, bps_k=80)
    D["repetitions"] = {str(k): agg(runs, "repetitions", k, "all63", 1.0, bps_k=k) for k in KS}
    D["repetitions"] = {k: v for k, v in D["repetitions"].items() if v}
    D["window_k80"] = {str(w): agg(runs, "acquisition", 80, "all63", w, bps_k=80) for w in WINDOWS}
    D["window_k1"] = {str(w): agg(runs, "acquisition", 1, "all63", w, bps_k=1) for w in WINDOWS}
    subsets = sorted({r["channels"] for run in runs for r in sel(run["rows"], "acquisition")})
    groups = {"all63": ["all63"], "occpar32": ["occpar32"], "random32": [s for s in subsets if s.startswith("random32")]}
    for extra in ("occpar16", "occpar8"):
        if extra in subsets:
            groups[extra] = [extra]; groups[extra.replace("occpar", "random")] = [s for s in subsets if s.startswith(extra.replace("occpar", "random") + "_")]
    D["channels_k80_w1"] = {g: agg(runs, "acquisition", 80, c, 1.0, bps_k=80) for g, c in groups.items() if c}
    D["channels_k1_w1"] = {g: agg(runs, "acquisition", 1, c, 1.0, bps_k=1) for g, c in groups.items() if c}
    D["ceiling_split_half_40v40"] = agg(runs, "ceiling_split_half", 40, "all63", 1.0)
    ck = {str(k): agg(runs, "ceiling_k_vs_half", k, "all63", 1.0) for k in [1, 4, 16]}
    D["ceiling_k_vs_half40"] = {k: v for k, v in ck.items() if v}
    summary["decoders"][name] = D
(OUT / f"summary{SUFFIX}.json").write_text(json.dumps(summary, indent=1))

# ---------------- figure ----------------
fs.setup()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), gridspec_kw=dict(wspace=0.38))
ax_k, ax_bps, ax_w, ax_ch = axes
for name, D in summary["decoders"].items():
    if "repetitions" not in D or not D["repetitions"]:
        continue
    c, mk = COLOR[name], MARK[name]
    lab = f"{name} (n={D['n_subjects']} subj, {D['n_seeds']} seed{'s' if D['n_seeds'] > 1 else ''})"
    ks = [int(k) for k in D["repetitions"]]
    for ax, key in ((ax_k, "bits"), (ax_bps, "bits_per_second")):
        y = [D["repetitions"][str(k)][key]["mean"] for k in ks]
        lo = [D["repetitions"][str(k)][key]["ci95"][0] for k in ks]; hi = [D["repetitions"][str(k)][key]["ci95"][1] for k in ks]
        ax.plot(ks, y, color=c, marker=mk, label=lab, zorder=3)
        ax.fill_between(ks, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
    ws = [w for w in WINDOWS if D["window_k80"][str(w)]]
    y = [D["window_k80"][str(w)]["bits"]["mean"] for w in ws]
    lo = [D["window_k80"][str(w)]["bits"]["ci95"][0] for w in ws]; hi = [D["window_k80"][str(w)]["bits"]["ci95"][1] for w in ws]
    ax_w.plot(ws, y, color=c, marker=mk, zorder=3); ax_w.fill_between(ws, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
    groups = ["random32", "occpar32", "all63"]
    off = {"ATM-S": -0.12, "NICE": 0.0, "NICE-GA": 0.12}[name]
    for i, g in enumerate(groups):
        v = D["channels_k80_w1"].get(g)
        if not v: continue
        m, (l, h) = v["bits"]["mean"], v["bits"]["ci95"]
        ax_ch.errorbar(i + off, m, yerr=[[m - l], [h - m]], color=c, marker=mk, ls="none", capsize=2, elinewidth=1.2, zorder=3)
ax_k.set_xscale("log", base=2); ax_bps.set_xscale("log", base=2)
for ax in (ax_k, ax_bps):
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.set_xlabel("repetitions averaged per query, k")
ax_k.set_ylabel("bits per query (200-way)"); ax_k.set_title("(a) yield vs repetitions")
ax_bps.set_ylabel("bits per second of EEG"); ax_bps.set_title("(b) yield per recording time")
ax_w.set_xticks(WINDOWS); ax_w.set_xlabel("post-stimulus window used (s), k = 80"); ax_w.set_ylabel("bits per query"); ax_w.set_title("(c) yield vs window")
ax_ch.set_xticks(range(3)); ax_ch.set_xticklabels(["random 32", "occipito-\nparietal 32", "all 63"]); ax_ch.set_xlim(-0.5, 2.5)
ax_ch.set_ylabel("bits per query"); ax_ch.set_title("(d) yield vs electrodes, k = 80")
for ax in (ax_k, ax_w, ax_ch):
    ax.axhline(0, color=fs.GREY1, ls=(0, (4, 2)), lw=1.2, zorder=1)
    ax.set_ylim(bottom=-0.08)
ax_k.text(80, 0.03, "chance: 0 bits (top-1 = 0.5%)", color=fs.INK2, fontsize=7, va="bottom", ha="right")
ax_bps.axhline(0, color=fs.GREY1, ls=(0, (4, 2)), lw=1.2, zorder=1)
ax_k.legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=7)
fig.savefig(OUT / f"fig_repetition_ladder{SUFFIX}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"fig_repetition_ladder{SUFFIX}.png", dpi=180, bbox_inches="tight")

# console table
print(f"{'decoder':8s} {'nsub':>4s} {'nseed':>5s} {'native top1':>12s} {'bits@k80':>9s} {'bits/s@k1':>10s} {'eff@k80':>8s}")
for name, D in summary["decoders"].items():
    if "native" not in D or not D["native"]: print(name, "none"); continue
    r80, r1 = D["repetitions"]["80"], D["repetitions"]["1"]
    print(f"{name:8s} {D['n_subjects']:4d} {D['n_seeds']:5d} {100*D['native']['top1']['mean']:11.1f}% {r80['bits']['mean']:9.2f} {r1['bits_per_second']['mean']:10.2f} {r80['efficiency']['mean']:8.3f}")
