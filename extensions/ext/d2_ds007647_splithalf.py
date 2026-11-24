"""ds007647 ridge split-half anchor (, 23 Nov 2026; ideator cell 4). Same features/target/estimator as d2_ds007647_v2 (w0600, PCA-64 ridge,
D2.pe_gain, pe_resid target, all trials), computed separately on odd and even trials (alternating) per participant. Output: per-participant gain for each half and the
split-half reliability across participants (Pearson r, Spearman-Brown) with a participant bootstrap. Usage anchor = mean(half gain) per participant is NOT a repetition ceiling:
it is a trial-split reliability lower bound for the ridge row only."""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
import d2_ds007647_v2 as V
ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", default=str(ROOT / "results/ext/d2_ds007647_splithalf")); ap.add_argument("--summarize", action="store_true"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if a.summarize:
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; o = np.array([p["odd"] for p in P]); e = np.array([p["even"] for p in P]); r = float(np.corrcoef(o, e)[0, 1]); rng = np.random.default_rng(0)
    bs = []
    for _ in range(10000):
        i = rng.integers(0, len(P), len(P)); c = np.corrcoef(o[i], e[i])[0, 1]; bs.append(2 * c / (1 + c))
    S = dict(n=len(P), mean_odd=float(o.mean()), mean_even=float(e.mean()), pearson_r=r, spearman_brown=2 * r / (1 + r), sb_ci95=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
             mean_full=float(np.mean([p["full_gain"] for p in P])))
    (out / "summary.json").write_text(json.dumps(S, indent=1)); print(S); sys.exit(0)
info = D2.dataset(V.DS); common = D2.common_channels(info); B, PE, row, mname = V.raw_targets()
for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else info["subs"]):
    f = out / f"sub-{s:03d}.json"
    if f.exists(): continue
    t0 = time.time(); X, tri, times, _ = D2.epochs(info, s, common); w = (times >= 0) & (times <= 0.6 + 1e-9); Fw = X[:, :, w].reshape(len(X), -1)
    y = V.zs(PE[row[s]][tri]); n = len(y); res = dict(participant=s, n_trials=int(n)); L = (64, "full")   # pe_raw = unresidualised target of the 0.078 row
    for nm, idx in (("odd", np.arange(0, n, 2)), ("even", np.arange(1, n, 2)), ("full", np.arange(n))):
        Ys = np.column_stack([y[idx]] + [np.roll(y[idx], round(len(idx) * q / 4)) for q in (1, 2, 3)]); sc = D2.pe_gain(Fw[idx], Ys, L)
        res[nm if nm != "full" else "full_gain"] = float(sc[64][:, 0].mean())
    f.write_text(json.dumps(res)); print(s, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)}, f"{time.time()-t0:.0f}s", flush=True)
