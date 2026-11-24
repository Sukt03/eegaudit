"""Paper-facing numbers from the methodology pilots (experiments/method_pilots, 3 subjects, NICE-GA seed 141):
  r1 input-side bottleneck: smallest L with >= 90% of the full-width yield when the PCA bottleneck sits on the encoder INPUT
     (basis = the subject's own single test trials), per subject, under chance-adjusted 2-way, Wolpaw bits and the soft bound;
     plus the yield share at L = 25 and L = 50.
  r3 fill check: paired difference between a fixed fill estimated on disjoint images (mean_half, queries 100-199) and the paper's
     leave-one-out fill, over all masked conditions, k = 80: max |difference| in soft bits and in Wolpaw bits.
Output: results/method_pilots/paper_summary.json (pilot numbers; the paper states the 3-subject scope)."""
import glob, json
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/method_pilots")
out = {"r1": {}, "r3": {}}
L90 = {"two_way_adj": [], "wolpaw": [], "soft": []}; share25 = {"two_way_adj": [], "wolpaw": [], "soft": []}; share50 = {k: [] for k in share25}
subs = []
for f in sorted(glob.glob(str(R / "r1_input_bottleneck_sub-*_basis-sub-*.json"))):
    d = json.load(open(f))
    if "smoke" in f or d["basis_subject"] != d["subject"]: continue
    subs.append(d["subject"]); rows = [r for r in d["rows"] if r["condition"] == "eeg" and r["k"] == 80]
    full = [r for r in rows if r["L"] == "full"][0]
    def val(r, m):
        return {"two_way_adj": r["two_way"] - 0.5, "wolpaw": r["bits"], "soft": r["soft_bits"]}[m]
    Ls = sorted({int(r["L"]) for r in rows if r["L"] != "full"})
    for m in L90:
        fr = {L: np.mean([val(r, m) for r in rows if r["L"] == L]) / val(full, m) for L in Ls}
        L90[m].append(next((L for L in Ls if fr[L] >= 0.9), None) or "full")
        share25[m].append(float(fr.get(25, np.nan))); share50[m].append(float(fr.get(50, np.nan)))
out["r1"] = dict(subjects=subs, n_subjects=len(subs), L90_per_subject=L90,
                 L90_min={m: min(v) if all(isinstance(x, int) for x in v) else None for m, v in L90.items()},
                 L90_max={m: max(v) if all(isinstance(x, int) for x in v) else None for m, v in L90.items()},
                 share_at_L25_mean={m: float(np.nanmean(v)) for m, v in share25.items()}, share_at_L50_mean={m: float(np.nanmean(v)) for m, v in share50.items()})
diffs_soft, diffs_wol = [], []
for f in sorted(glob.glob(str(R / "r3_fill_sub-*_other-sub-*.json"))):
    if "smoke" in f: continue
    rows = json.load(open(f))["rows"]
    for r in rows:
        if r["fill"] == "mean_half" and r["k"] == 80 and r["queries"] != "all200":
            m = [x for x in rows if x["fill"] == "mean_loo" and x["k"] == 80 and x["queries"] == r["queries"] and x["channels"] == r["channels"] and x["window"] == r["window"]]
            if m:
                diffs_soft.append(r["soft_bits"] - m[0]["soft_bits"]); diffs_wol.append(r["bits"] - m[0]["bits"])
out["r3"] = dict(n_pairs=len(diffs_soft), max_abs_diff_soft_bits=float(np.max(np.abs(diffs_soft))) if diffs_soft else None,
                 max_abs_diff_wolpaw_bits=float(np.max(np.abs(diffs_wol))) if diffs_wol else None,
                 mean_diff_soft_bits=float(np.mean(diffs_soft)) if diffs_soft else None)
(R / "paper_summary.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
