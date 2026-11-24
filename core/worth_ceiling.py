"""worth_ceiling: how much is a signal worth, and how close is it to the ceiling?

Inputs
  --curve    results/ceiling_<model>/results.json  exact worth curve from exp_exact_joint.py: posterior contraction of
             each parameter (importance-sampled exact posteriors) for behaviour alone over session lengths, and for
             behaviour + a read-out of coupling rho, for rho from 0 to ~1
  --decoders name=path pairs to price_plugin.py outputs, which hold each participant's cross-validated decoding of the
             read-out quantity and the session-permutation null
Outputs (per decoder and parameter)
  rho_eff        null-corrected coupling: sqrt(median r^2 - mean null r^2)
  gain           exact contraction gain at rho_eff (linear interpolation of the exact curve)
  pct_ceiling    gain / gain at the highest rho on the curve (the belief ceiling: a perfect read-out of the quantity)
  trials         behavioural trials giving the same contraction gain (only while within the simulated session range)
plus a figure: exact curve (floor at 0, ceiling at rho -> 1) with each decoder placed at its coupling.
Usage: python worth_ceiling.py --curve results/ceiling_RWD/results.json --decoders raw=results/plugin_raw_pe ens=... --out results/worth_ceiling
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_curve(path):
    R = json.loads(Path(path).read_text())
    rhos = sorted(float(k) for k in R["joint"])
    params = list(R["joint"][list(R["joint"])[0]]["exact"].keys())
    curve = {nm: np.array([R["joint"][str(r) if str(r) in R["joint"] else repr(r)]["exact"][nm]["contraction"] for r in rhos])
             for nm in params}
    ns = sorted(int(k) for k in R["behaviour"])
    beh = {nm: np.array([R["behaviour"][str(n)]["exact"][nm]["contraction"] for n in ns]) for nm in params}
    return np.array(rhos), curve, np.array(ns), beh, params


def coupling_group(gs, name):
    """Canonical coupling: random-effects pooled, null-corrected Fisher z (group_stats.py), back-transformed, with 95% CI.
    The signed null correction removes the cross-validation bias that a squared correction leaves in."""
    G = json.loads(Path(gs).read_text())["decoders"][name]
    return G["rho_eff_pooled"], G["rho_eff_ci"]


def coupling(path):
    R = json.loads((Path(path) / "results.json").read_text())
    S = R["subjects"]
    r2 = np.median([np.mean(np.square(S[s]["r_hat"])) for s in S])
    n2 = np.mean([S[s]["r_null_rms"] ** 2 for s in S])
    return float(np.sqrt(max(r2 - n2, 0.0))), float(np.median([np.mean(S[s]["r_hat"]) for s in S]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--group_stats", default=None, help="results.json of group_stats.py; then --decoders are bare names")
    ap.add_argument("--max_rho", type=float, default=0.9, help="ceiling taken at this rho (importance sampling thins near 1)")
    ap.add_argument("--n0", type=int, default=240, help="session length the joint curve was computed at")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rhos, curve, ns, beh, params = load_curve(a.curve)
    keep = rhos <= a.max_rho + 1e-9
    rhos, curve = rhos[keep], {k: v[keep] for k, v in curve.items()}
    out = {"rhos": rhos.tolist(), "curve": {k: v.tolist() for k, v in curve.items()}, "decoders": {}}
    for item in a.decoders:
        if a.group_stats:
            name = item; reff, ci = coupling_group(a.group_stats, name)
            d = {"rho_eff": reff, "rho_ci": ci}
        else:
            name, path = item.split("=", 1)
            reff, rraw = coupling(path); ci = None
            d = {"rho_eff": reff, "r_median": rraw}
        for nm in params:
            g0 = curve[nm][0]
            gain = float(np.interp(reff, rhos, curve[nm]) - g0)
            ceil = float(curve[nm][-1] - g0)
            yb = np.maximum.accumulate(beh[nm]); y0 = float(np.interp(a.n0, ns, yb))
            yt = y0 + gain
            trials = float(np.exp(np.interp(yt, yb, np.log(ns))) - a.n0) if (gain > 0 and yt < yb[-1]) else (0.0 if gain <= 0 else None)
            d[nm] = {"gain": gain, "ceiling_gain": ceil, "pct_ceiling": 100 * gain / ceil if ceil > 0 else None, "trials": trials}
            if ci is not None:
                d[nm]["pct_ceiling_ci"] = [100 * float(np.interp(c, rhos, curve[nm]) - g0) / ceil if ceil > 0 else None for c in ci]
        out["decoders"][name] = d
        print(name, f"rho_eff={reff:.3f}", {nm: (round(d[nm]["pct_ceiling"] or 0, 1), d[nm]["trials"] and round(d[nm]["trials"], 1)) for nm in params})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "results.json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(params), figsize=(1.8 * len(params), 1.8), sharey=False)
        for ax, nm in zip(np.atleast_1d(axs), params):
            g = curve[nm] - curve[nm][0]
            ax.plot(rhos, g, "-", color="0.3", lw=1)
            ax.axhline(g[-1], color="0.6", ls=":", lw=0.8)
            for k, (dn, d) in enumerate(out["decoders"].items()):
                ax.scatter(d["rho_eff"], d[nm]["gain"], s=14, zorder=3, label=dn)
            ax.set_title(nm, fontsize=7); ax.set_xlabel("read-out coupling", fontsize=6)
        np.atleast_1d(axs)[0].set_ylabel("exact contraction gain", fontsize=6)
        np.atleast_1d(axs)[-1].legend(frameon=False, fontsize=4.5)
        fig.tight_layout(); fig.savefig(Path(a.out) / "worth_ceiling.pdf")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
