"""Measure-only cognitive target: bits per EEG trial that a decoder's read-out carries about the learning model's signed
prediction error, from the measured LOPO coupling rho via the Gaussian-channel bound I >= -1/2 log2(1 - rho^2) (exact lower bound
when the standardised target is Gaussian; minus the target's negentropy otherwise). Floor = same transform at the 95th percentile of
the decoder's session-permutation null r. Ceiling = same transform at the read-out's split-half reliability bound rho_max where measured.
Input : results/track_decision/price_list.json (+ results/method_pilots/r4_summary.json for rho_max)
Output: results/track_decision/measure_bits.json"""
import json
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results")
b = lambda r: float(-0.5 * np.log2(1 - min(abs(r), 0.999) ** 2))
pl = json.load(open(R / "track_decision/price_list.json"))
out = {"formula": "-0.5*log2(1-rho^2) bits per trial", "datasets": {}}
for ds, v in pl["datasets"].items():
    rows = {}
    for k, d in v["decoders"].items():
        if "rho_eff" not in d: continue
        c = v.get("controls", {}).get(k, {}).get("session_permutation_null", {})
        rows[k] = dict(label=d.get("label", k), rho=d["rho_eff"], rho_ci=d.get("rho_ci"), bits_per_trial=b(d["rho_eff"]),
                       bits_per_trial_ci=[b(x) for x in d["rho_ci"]] if d.get("rho_ci") and None not in d["rho_ci"] else None,
                       mbits_per_trial=1000 * b(d["rho_eff"]),
                       floor_p95_r=c.get("p95_r"), floor_bits=b(c["p95_r"]) if c.get("p95_r") is not None else None,
                       n_subjects=d.get("n_subjects"))
    out["datasets"][ds] = dict(n_participants=v.get("n_participants"), decoders=rows)
try:
    r4 = json.load(open(R / "method_pilots/r4_summary.json"))
    rm = next(r4[k] for k in r4 if "rho_max" in str(k).lower() and isinstance(r4[k], (int, float))) if any("rho_max" in str(k).lower() for k in r4) else None
except Exception as e:
    rm = None
out["ceiling_note"] = "ceiling = bits at rho_max (ensemble read-out split-half reliability) if found"
if rm: out["ceiling_rho_max_ens"] = rm; out["ceiling_bits_ens"] = b(rm)
(R / "track_decision/measure_bits.json").write_text(json.dumps(out, indent=1))
for ds, v in out["datasets"].items():
    for k, d in sorted(v["decoders"].items(), key=lambda kv: -kv[1]["bits_per_trial"]):
        print(ds, k, f"rho={d['rho']:.3f} bits/trial={d['bits_per_trial']:.4f} ci={d['bits_per_trial_ci']} floor={d['floor_bits']}")
print("ceiling", out.get("ceiling_rho_max_ens"), out.get("ceiling_bits_ens"))
