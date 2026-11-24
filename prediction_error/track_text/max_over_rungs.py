"""Reviewer fix: the best within-SR rung (max over decoder x participant settings) for EEG vs the same max taken over the noise,
shuffled-pair and trial-shuffled controls, seeds averaged per setting. Output: results/track_text/within_SR/max_over_rungs.json"""
import glob, json, collections
from pathlib import Path
import numpy as np
D = Path("/path/to/eegaudit/results/track_text/within_SR")
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(str(D / "ridge_*__seed*.json")):
    d = json.load(open(f)); key = (d["decoder"], d["subject"])
    for r in d["rows"]:
        if r["k"] == 1 and r.get("subjects_averaged", 1) == 1:
            acc[key][r["condition"]].append((r["two_way"], r["bits"]))
out = {"n_settings": len(acc)}
for cond in ["eeg", "noise", "shuffled_pair", "trial_shuffled"]:
    vals = {k: np.mean(v[cond], 0) for k, v in acc.items() if v.get(cond)}
    if not vals: continue
    tw = {k: v[0] for k, v in vals.items()}; bt = {k: v[1] for k, v in vals.items()}
    kb = max(tw, key=tw.get)
    out[cond] = dict(n=len(vals), max_two_way=float(tw[kb]), max_two_way_setting="|".join(kb), max_bits=float(max(bt.values())), mean_two_way=float(np.mean(list(tw.values()))))
(D / "max_over_rungs.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
