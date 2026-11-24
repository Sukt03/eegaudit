"""D3.1 + D3.2 CROSS-DATASET RIDGE TRANSFER, ds003458 <-> ds004595 (B5; rung R9). EXECUTOR-3, 21 Nov 2026.
Design = EXPERIMENT_TABLES_FINAL.md D3 + "## D3 REVIEW v1" binding changes (EXTENSIONS_LOOP_20261121.md).
Data, targets and epochs = d2_pe_tasks.py (same trials, same z-scored residualised PE of each dataset's OWN locked model; the transferred
  object is the EEG -> standardised-PE map, stated because the two PE definitions differ). Channels = name intersection of the two
  datasets' common sets (printed); window 0-600 ms at 50 Hz (D2 primary window), identical time grid in both.
Source A (fit once, frozen): every A participant's features z-scored with that participant's own statistics (label-free), stacked;
  ridge (LAM * n_rows) on the stacked z-scored targets; widths L = 1,2,4,...,256 via PCA of the stacked A features, and 'full'.
Target B, per participant: dev = first contiguous 20 % of trials, test = the rest. B features standardised with the dev rows' mean/sd
  (label-free; D3 REVIEW 1). Read-out x = frozen A map. Recalibration (2 scalars: intercept + slope, + residual variance) fit on dev only,
  test score = log2 density gain over the dev intercept-only density (= d2_pe_tasks.gain_t on the GPU). Primary width PCA-64 (METHOD CHANGE 12), curve 1..256, full.
  All numerics in torch on the GPU (incl. bootstraps); CPU = loading/IO.
  Within-dataset reference on the SAME test rows (D3 REVIEW 2): X7 ridge, 10 contiguous folds over the test block, nested calibration
  (d2_pe_tasks.pe_gain). Transfer fraction = transfer gain / within gain, per participant; primary = median fraction across B
  participants with a participant-bootstrap CI; pass: >= 30 % with the CI excluding 10 %.
Floors: the transfer read-out against the target circularly shifted within the test block (n/4, n/2, 3n/4; calibration on the
  equally shifted dev rows is not possible, so the shift is applied to the whole B series before the dev/test cut), and an A map fit on
  N(0,1) features of A's shape (seed 0). P3b: L90 of the participant-mean transfer curve (X7 rule: smallest L with mean gain >= 90 % of
  the max over L) and its bootstrap CI vs B's within-dataset X7 L90 (results/standard/pe_bottleneck/summary.json for ds003458).
G2: the within-dataset reference uses d2_pe_tasks.pe_gain, whose ds003458 X7 output reproduces results/standard/pe_bottleneck exactly (D2 G2);
  when B = ds003458 the same pe_gain on ALL rows at 150-600 ms is compared with results/standard/pe_bottleneck/sub-XXX.json ('g2').
METHOD G1 PASS D3 additions: LOPO-within-A arms (lopo3458, lopo4595) through the identical path on the same 60 channels; r_test with a
  bootstrap CI; CHANGE 12 group floor gate (D2.g3_test) per arm; channel-name assert.
Usage: python d3_ridge_transfer.py --direction 3458to4595|4595to3458 --out DIR [--subjects ..] | --summarize --out DIR
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
sys.path.insert(0, str(ROOT / "experiments/ext"))
import d2_pe_tasks as D2
from belief_bottleneck import LAM

DEV = "cuda"
LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, "full"]
PRIMARY = 64                                   # METHOD CHANGE 12: in-fold PCA-64 + ridge is the primary PE read-out (D2), used here too
from d2_pe_tasks import gain_t
PAIR = {"3458to4595": ("ds003458", "ds004595"), "4595to3458": ("ds004595", "ds003458")}


_EP = {}


def feats(info, s, chans, lo, hi):
    key = (str(info["E"]), s, tuple(chans))
    if key not in _EP: _EP[key] = D2.epochs(info, s, chans)                  # cache (LOPO refits read every participant many times)
    X, tri, times, Ep = _EP[key]; w = (times >= lo - 1e-9) & (times <= hi + 1e-9)
    chl = [str(c) for c in Ep["ch"]]; assert [chl[chl.index(c)] for c in chans] == list(chans) and X.shape[1] == len(chans)   # channel-name indexing (METHOD)
    return X[:, :, w].reshape(len(X), -1), tri


def fit_source(info, chans, seed_noise=None, exclude=None):
    Fs, Ys = [], []
    for s in info["subs"]:
        if s == exclude: continue
        F, tri = feats(info, s, chans, 0.0, 0.60)
        F = D2.gnoise(F.shape, seed_noise + s) if seed_noise is not None else torch.as_tensor(F, dtype=torch.float64, device=DEV)
        Fs.append((F - F.mean(0)) / (F.std(0) + 1e-6)); Ys.append(torch.as_tensor(info["qtil"](s, tri), dtype=torch.float64, device=DEV))
    A = torch.cat(Fs); y = torch.cat(Ys)
    yc = y - y.mean(); n = len(A); W = {}
    _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False)
    for L in LS:
        if L == "full":
            if A.shape[1] < n:                                    # primal when D < n, else dual
                W[L] = torch.linalg.solve(A.T @ A + LAM * n * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ yc)
            else:
                W[L] = A.T @ torch.linalg.solve(A @ A.T + LAM * n * torch.eye(n, dtype=torch.float64, device=DEV), yc)
        else:
            V = Vt[:L].T; Z = A @ V; W[L] = V @ torch.linalg.solve(Z.T @ Z + LAM * n * torch.eye(L, dtype=torch.float64, device=DEV), Z.T @ yc)
    return W, n


def apply_target(W, Wn, IB, b, s, chans, cb, tag, extra):
    F, tri = feats(IB, s, chans, 0.0, 0.60); y = IB["qtil"](s, tri).astype(np.float64); n = len(y); nd = int(round(0.2 * n))
    te = np.arange(nd, n)
    Fg = torch.as_tensor(F, dtype=torch.float64, device=DEV); Zg = (Fg - Fg[:nd].mean(0)) / (Fg[:nd].std(0) + 1e-6)
    yg = torch.as_tensor(y, dtype=torch.float64, device=DEV); Ysh = torch.stack([yg] + [torch.roll(yg, round(n * q / 4)) for q in (1, 2, 3)])   # (4, n)
    X = torch.stack([Zg @ W[L] for L in LS]).reshape(len(LS), 1, n)                  # (nL, 1, n) frozen-source read-outs
    G = gain_t(Ysh[None, :, :nd], X[..., :nd], Ysh[None, :, nd:], X[..., nd:]).mean(-1).cpu().numpy()   # (nL, 4) GPU calibration on dev, score on test
    rt = [float(torch.corrcoef(torch.stack([X[i, 0, nd:], yg[nd:]]))[0, 1]) for i in range(len(LS))]
    rows = {str(L): dict(gain=float(G[i, 0]), shifted=[float(v) for v in G[i, 1:]], shift_mean=float(G[i, 1:].mean()), r_test=rt[i]) for i, L in enumerate(LS)}
    xn = (Zg @ Wn[PRIMARY]).reshape(1, n); noise = float(gain_t(yg[None, :nd], xn[:, :nd], yg[None, nd:], xn[:, nd:]).mean())
    ys_te = np.column_stack([y[te]] + [np.roll(y[te], round(len(te) * q / 4)) for q in (1, 2, 3)])
    within = D2.pe_gain(F[te], ys_te, (PRIMARY, "full"))
    wg = float(within[PRIMARY][:, 0].mean()); tg = rows[str(PRIMARY)]["gain"]
    res = dict(**extra, participant=f"sub-{s:03d}", n_trials=int(n), n_dev=int(nd), n_test=int(len(te)), channels=chans,
               window_ms=[0, 600], unit="bits per feedback trial (test block)", split="target: first 20 % dev (recalibration + standardisation), rest test",
               primary_L=PRIMARY, transfer=rows, noise_source=noise,
               within_test_rows=dict(gain=wg, shifted=[float(v) for v in within[PRIMARY][:, 1:].mean(0)], gain_full=float(within["full"][:, 0].mean())),
               fraction=tg / wg if wg > 0 else None, floor=max(rows[str(PRIMARY)]["shift_mean"], noise))
    if b == "ds003458" and extra.get("arm") == "cross_dataset":
        F7, _ = feats(IB, s, cb, 0.15, 0.60); g7 = float(D2.pe_gain(F7, y[:, None])[:, 0].mean())
        old = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
        o = next(r for r in old["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")
        res["g2"] = dict(old=o["gain"], new=g7, abs_diff=abs(o["gain"] - g7), passed=bool(abs(o["gain"] - g7) < 1e-9))
    return res


def run(direction, out, subjects):
    """direction 'AtoB' (cross-dataset) or 'lopo3458' / 'lopo4595' (METHOD G1 PASS D3 (A): LOPO within A through the identical path:
    the source = all OTHER participants of A, the target = the held-out participant, same dev/test protocol and within reference)."""
    t0 = time.time()
    if direction.startswith("lopo"):
        a = b = "ds00" + direction[4:]; IA = IB = D2.dataset(a)
    else:
        a, b = PAIR[direction]; IA, IB = D2.dataset(a), D2.dataset(b)
    I3, I4 = (IA, IB) if a == "ds003458" else (IB, IA)
    ca, cb = D2.common_channels(D2.dataset("ds003458")), D2.common_channels(D2.dataset("ds004595")); chans = [c for c in ca if c in cb]   # the SAME 60 channels in every arm
    cbB = D2.common_channels(IB)
    print(direction, "channels", len(chans), flush=True)
    if not direction.startswith("lopo"):
        W, nA = fit_source(IA, chans); Wn, _ = fit_source(IA, chans, seed_noise=5000); print(direction, "source fitted", nA, "rows", f"{time.time() - t0:.0f}s", flush=True)
    for s in IB["subs"]:
        if subjects and s not in subjects: continue
        dest = out / f"{direction}_sub-{s:03d}.json"
        if dest.exists(): continue
        if direction.startswith("lopo"):
            W, nA = fit_source(IA, chans, exclude=s); Wn, _ = fit_source(IA, chans, seed_noise=5000, exclude=s)
        res = apply_target(W, Wn, IB, b, s, chans, cbB, direction, dict(direction=direction, source=a, target=b, arm="lopo_within" if direction.startswith("lopo") else "cross_dataset", source_rows=int(nA)))
        dest.write_text(json.dumps(res, indent=1))
        print(direction, f"sub-{s:03d}", f"transfer {res['transfer'][str(PRIMARY)]['gain']:.4f} within {res['within_test_rows']['gain']:.4f} floor {res['floor']:.4f} r {res['transfer'][str(PRIMARY)]['r_test']:.3f}", f"{time.time() - t0:.0f}s", flush=True)


def summarize(out):
    res = {}
    def gq(x, q): return torch.quantile(x, torch.tensor(q, dtype=torch.float64, device=DEV)).cpu().tolist()
    for direction in list(PAIR) + ["lopo3458", "lopo4595"]:
        per = [json.loads(f.read_text()) for f in sorted(out.glob(f"{direction}_sub-*.json"))]
        if not per: continue
        T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
        rt = T([p["transfer"][str(PRIMARY)]["r_test"] for p in per])
        tg = T([p["transfer"][str(PRIMARY)]["gain"] for p in per]); wg = T([p["within_test_rows"]["gain"] for p in per]); fl = T([p["floor"] for p in per])
        ok = wg > 0; fr = tg[ok] / wg[ok]; g = torch.Generator(device=DEV); g.manual_seed(20261121)
        BI = torch.randint(0, len(tg), (10000, len(tg)), generator=g, device=DEV)
        med = fr[torch.randint(0, len(fr), (10000, len(fr)), generator=g, device=DEV)].median(1).values
        G = T([[p["transfer"][str(L)]["gain"] for L in LS] for p in per]); m = G.mean(0); L90 = LS[int((m >= 0.9 * m.max()).double().argmax())]
        M = G[torch.randint(0, len(G), (10000, len(G)), generator=g, device=DEV)].mean(1); Lnum = T([L if L != "full" else 10 ** 6 for L in LS])
        bl = Lnum[(M >= 0.9 * M.max(1, keepdim=True).values).double().argmax(1)]
        res[direction] = dict(n=len(per), primary_L=PRIMARY, transfer_gain=float(tg.mean()), transfer_ci=D2.boot_ci(tg.cpu().numpy()), within_gain=float(wg.mean()),
                              within_ci=D2.boot_ci(wg.cpu().numpy()), floor=float(fl.mean()), transfer_minus_floor_ci=D2.boot_ci((tg - fl).cpu().numpy()),
                              n_transfer_above_floor=int((tg > fl).sum()), n_within_positive=int(ok.sum()), median_fraction=float(fr.median()),
                              median_fraction_ci=gq(med, [.025, .975]), ratio_of_means=float(tg.mean() / wg.mean()) if wg.mean() > 0 else None,
                              ratio_of_means_ci=gq((tg[BI].mean(1) / wg[BI].mean(1)), [.025, .975]),
                              primary="ratio of participant means (METHOD BUG HUNT C2); the median over participants with within > 0 is secondary",
                              passed=bool(wg.mean() > 0 and float(tg.mean() / wg.mean()) >= 0.3 and gq((tg[BI].mean(1) / wg[BI].mean(1)), [.025])[0] > 0.1),
                              passed_median_secondary=bool(float(fr.median()) >= 0.3 and gq(med, [.025])[0] > 0.1),
                              curve={str(L): float(v) for L, v in zip(LS, m.cpu().tolist())}, L90_transfer=L90, L90_transfer_ci=[int(x) for x in gq(bl, [.025, .975])],
                              g2=[p["g2"] for p in per if "g2" in p][:3],
                              r_test_mean=float(rt.mean()), r_test_ci=gq(rt[torch.randint(0, len(rt), (10000, len(rt)), generator=g, device=DEV)].mean(1), [.025, .975]),
                              recalibration_cost=dict(label="shift floor = recalibration cost of the 2-scalar dev fit with no signal",
                                                      analytic=float(np.mean([-2 / (2 * p["n_dev"] * np.log(2)) for p in per])), formula="-p / (2 n_dev ln 2), p = 2, mean over participants",
                                                      observed_shift_floor=float(np.mean([p["transfer"][str(PRIMARY)]["shift_mean"] for p in per])), n_dev_mean=float(np.mean([p["n_dev"] for p in per]))),
                              g3_shift=D2.g3_test([p["transfer"][str(PRIMARY)]["shift_mean"] for p in per]), g3_noise_source=D2.g3_test([p["noise_source"] for p in per]),
                              floor_outliers=[p["participant"] for p in per if max(abs(p["transfer"][str(PRIMARY)]["shift_mean"]), abs(p["noise_source"])) > 0.06])
    (out / "summary_d3.json").write_text(json.dumps(dict(experiment="D3.1/D3.2 ridge transfer (B5)", note="L90 1000000 = full width; GPU bootstraps", directions=res), indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--direction", default="3458to4595,4595to3458,lopo3458,lopo4595"); ap.add_argument("--subjects", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    subj = [int(x) for x in a.subjects.split(",")] if a.subjects else []
    for d in a.direction.split(","):
        run(d, out, subj)


if __name__ == "__main__":
    main()
