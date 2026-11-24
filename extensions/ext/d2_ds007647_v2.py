"""ds007647 SALVAGE (IDEATOR round 17:20, item 3). FIXER, 21 Nov 2026. Reuses d2_pe_tasks (epochs, channels, pe_gain, gnoise) unchanged.
Targets, each z-scored within the subset, scored in ALL / learnable (cond 2) / unlearnable (cond 1) trials, window w0600 (D2 primary),
in-fold PCA-64 + ridge with nested calibration (D2 primary read-out) and 'full':
  pe_resid  = the D2 target (residualised PE of the best model; reference, = D2 numbers);
  pe_raw    = the same model's PE, NOT residualised;
  outcome   = reward (0/1).
Floors: circular shifts n/4, n/2, 3n/4 and N(0,1) input (seed 1000 + s), as D2.
Floor-vs-analytic recalibration cost: a read-out with no information still pays for its fitted calibration parameters. For a Gaussian
density with p extra fitted parameters over n_fit calibration rows the expected held-out cost is about p / (2 n_fit ln 2) bits per trial;
the observed noise-input floor is printed beside this approximation for p = 1 (slope) and p = 2 (slope + variance), n_fit = 9/10 of the subset.
Output results/ext/d2_ds007647_v2/{sub-XXX.json, summary.json} with code_sha256."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); DS = "ds007647"; LS = (64, "full")


def raw_targets():
    from cogmodels import doors as Dm
    B = Dm.LOADERS[DS](f"{ROOT}/data_raw/{DS}"); mname = json.loads((ROOT / f"results/fit_{DS}.json").read_text())["best_model"]
    z = torch.tensor(np.load(ROOT / f"results/map_{DS}_{mname}.npy")).float()
    with torch.no_grad():
        PE = Dm.run(Dm.MODELS[mname], z, *Dm.tensors(B), record=True)["pe"].numpy()
    return B, PE, {int(s): k for k, s in enumerate(B["sub"])}, mname


def zs(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)


def subject(s, out, info, common, B, PE, row):
    t0 = time.time(); X, tri, times, _ = D2.epochs(info, s, common); j = row[s]
    w = (times >= 0.0 - 1e-9) & (times <= 0.60 + 1e-9); Fw = X[:, :, w].reshape(len(X), -1)
    k = np.asarray(info["cond"](s))[tri]; subsets = dict(all=np.ones(len(tri), bool), learnable=k == 2, unlearnable=k == 1)
    res = dict(code_sha256=CODE_SHA256, dataset=DS, participant=f"sub-{s:03d}", window="w0600", n_trials=int(len(tri)), subsets={})
    for sn, m in subsets.items():
        tt = tri[m]; n = int(m.sum()); res["subsets"][sn] = dict(n=n)
        if n < 40: res["subsets"][sn]["excluded"] = "< 40 trials"; continue
        tg = dict(pe_resid=info["qtil"](s, tt).astype(np.float64), pe_raw=zs(PE[j][tt]), outcome=zs(B["reward"][j][tt]))
        for tn, y in tg.items():
            Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
            sc = D2.pe_gain(Fw[m], Ys, LS); nz = D2.pe_gain(D2.gnoise(Fw[m].shape, 1000 + s), Ys[:, :1], LS)
            res["subsets"][sn][tn] = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean()), noise=float(nz[L][:, 0].mean())) for L in LS}
        nfit = 0.9 * n; res["subsets"][sn]["analytic_recal_cost_bits"] = {f"p{p}": -p / (2 * nfit * np.log(2)) for p in (1, 2)}
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"sub-{s:03d}.json").write_text(json.dumps(res, indent=1))
    print(res["participant"], {sn: {tn: round(v["64"]["gain"], 4) for tn, v in d.items() if isinstance(v, dict) and "64" in v} for sn, d in res["subsets"].items()}, f"{res['runtime_s']}s", flush=True)


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; res = dict(code_sha256=CODE_SHA256, n=len(P), note=__doc__, cells={})
    for sn in ("all", "learnable", "unlearnable"):
        for tn in ("pe_resid", "pe_raw", "outcome"):
            for L in ("64", "full"):
                use = [p for p in P if tn in p["subsets"][sn]]
                if not use: continue
                g = np.array([p["subsets"][sn][tn][L]["gain"] for p in use]); fl = np.array([max(p["subsets"][sn][tn][L]["shift_mean"], p["subsets"][sn][tn][L]["noise"]) for p in use])
                nzv = np.array([p["subsets"][sn][tn][L]["noise"] for p in use]); an = np.array([p["subsets"][sn]["analytic_recal_cost_bits"]["p2"] for p in use])
                res["cells"][f"{sn}|{tn}|L{L}"] = dict(n=len(use), gain=dict(mean=float(g.mean()), ci=D2.boot_ci(g)), gain_minus_floor=dict(mean=float((g - fl).mean()), ci=D2.boot_ci(g - fl)),
                                                      g3_shift=D2.g3_test([p["subsets"][sn][tn][L]["shift_mean"] for p in use]), g3_noise=D2.g3_test(nzv),
                                                      noise_floor_mean=float(nzv.mean()), analytic_p2_mean=float(an.mean()), analytic_p1_mean=float(an.mean() / 2))
    for tn in ("pe_resid", "pe_raw", "outcome"):
        a, b = [[p["subsets"][x][tn]["64"]["gain"] for p in P if tn in p["subsets"][x] and tn in p["subsets"]["unlearnable" if x == "learnable" else "learnable"]] for x in ("learnable", "unlearnable")]
        if a: d = np.array(a) - np.array(b); res[f"learnable_minus_unlearnable|{tn}|L64"] = dict(d=float(d.mean()), ci=D2.boot_ci(d), n_pos=int((d > 0).sum()), n=len(d))
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1)[:3500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", default=str(ROOT / "results/ext/d2_ds007647_v2")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out); sys.exit(0)
    info = D2.dataset(DS); common = D2.common_channels(info); B, PE, row, mname = raw_targets()
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else info["subs"]):
        if not (out / f"sub-{s:03d}.json").exists(): subject(s, out, info, common, B, PE, row)
