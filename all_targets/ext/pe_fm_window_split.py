"""fm_epochs broadband vs X7 split (METHOD RULING 06:55; rung R3 sensitivity). EXECUTOR-2, 21 Nov 2026.
Same read-out and score as the M3 PE broadband cell (m3_yield_map.pe_gain: in-fold z-score, ridge LAM * n, nested calibration,
Gaussian log-score, 3 circular-shift floors), ds003458, 23 participants, X7 trial set:
  fm_0_600    = fm_epochs (200 Hz, 0.3-75 Hz) broadband voltage, 64 ch, 0-600 ms, 30 bins of 20 ms   (= the M3 broadband cell; G2)
  fm_150_610  = the same features on the X7 window: 23 bins of 20 ms starting at 150 ms (bin starts = X7 sample times 150..590 ms)
  x7          = results/features/raw (X7; G2 = results/standard/pe_bottleneck full gain)
WINDOW effect = fm_0_600 - fm_150_610; SAMPLING/FILTER effect = fm_150_610 - x7. Participant bootstrap CIs on the GPU.
Output results/ext/m3/pe_fm_window_split.json. Usage: python pe_fm_window_split.py
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import m3_yield_map as M
from belief_bottleneck import LAM
from i2_complementarity import targets


def main():
    evaluate, subs, fpath, _ = targets(); rows = []
    for s in subs:
        sub, fz, tri, n, Ys, X, ch = M.pe_load(s, evaluate, fpath)
        f0 = X[:, :, 200:320].reshape(n, 64, 30, 4).mean(-1).reshape(n, -1)
        f1 = X[:, :, 230:322].reshape(n, 64, 23, 4).mean(-1).reshape(n, -1)
        g = {k: M.pe_gain(F.astype(np.float64), w, Ys, LAM) for k, F, w in (("fm_0_600", f0, np.ones(f0.shape[1])), ("fm_150_610", f1, np.ones(f1.shape[1])),
                                                                         ("x7", fz["F"], fz["colw"].astype(np.float64)))}
        r = dict(participant=sub, **{k: float(v[:, 0].mean()) for k, v in g.items()}, **{k + "_shift": float(v[:, 1:].mean()) for k, v in g.items()})
        m3 = json.loads((ROOT / f"results/ext/m3/pe_{sub}.json").read_text())["broadband_cell"]["yield_"]
        st = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text())
        r["g2"] = dict(fm_vs_m3=abs(r["fm_0_600"] - m3), x7_vs_standard=abs(r["x7"] - next(x["gain"] for x in st["rows"] if x["input"] == "eeg" and x["mode"] == "pca" and x["L"] == "full")))
        rows.append(r); print(r, flush=True)
    A = {k: torch.tensor([r[k] for r in rows], dtype=torch.float64, device="cuda") for k in ("fm_0_600", "fm_150_610", "x7")}
    gen = torch.Generator(device="cuda"); gen.manual_seed(20261121); ix = torch.randint(0, len(rows), (10000, len(rows)), generator=gen, device="cuda")
    ci = lambda v: torch.quantile(v[ix].mean(1), torch.tensor([.025, .975], dtype=v.dtype, device="cuda")).tolist()
    res = dict(n=len(rows), means={k: float(v.mean()) for k, v in A.items()}, cis={k: ci(v) for k, v in A.items()},
               window_effect=dict(mean=float((A["fm_0_600"] - A["fm_150_610"]).mean()), ci=ci(A["fm_0_600"] - A["fm_150_610"])),
               sampling_filter_effect=dict(mean=float((A["fm_150_610"] - A["x7"]).mean()), ci=ci(A["fm_150_610"] - A["x7"])),
               total=dict(mean=float((A["fm_0_600"] - A["x7"]).mean()), ci=ci(A["fm_0_600"] - A["x7"])),
               g2_max=dict(fm_vs_m3=max(r["g2"]["fm_vs_m3"] for r in rows), x7_vs_standard=max(r["g2"]["x7_vs_standard"] for r in rows)), rows=rows, note=__doc__)
    assert res["g2_max"]["fm_vs_m3"] < 1e-9 and res["g2_max"]["x7_vs_standard"] < 1e-9, res["g2_max"]
    (ROOT / "results/ext/m3/pe_fm_window_split.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "note")}, indent=1))


if __name__ == "__main__":
    main()
