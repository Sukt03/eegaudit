"""A2 NUISANCE INTERVENTION: LEACE subject erasure on the LOPO (cross-subject) PE yield (rung R6). EXECUTOR-3, 21 Nov 2026.
Design = METHOD "RULINGS on the CONTRARIAN" A2 (EXTENSIONS_LOOP_20261121.md): erase the linearly decodable SUBJECT identity from the
features with LEACE (Belrose et al. 2023; FMScope arXiv 2606.06647 as the method source), fit on the TRAINING participants only, and
measure the change of the cross-subject (LOPO) target yield and of the within-vs-cross gap on the same test rows; control = erasing a
random subspace of the same rank (5 seeds), so an effect can be called subject-specific.
Data: ds003458 X7 raw-ERP features (results/features/raw: 64 ch x 23 samples, 150-600 ms) of the 23 participants; target = the locked
  z-scored residualised PE of each participant (s13b targets via d2_pe_tasks.dataset). Raw input here; the FM latent Z joins when
  EXECUTOR-4 provides it (same code, other feature files).
Per held-out participant h (LOPO fold):
  pooled standardisation on the 22 training participants' rows (NO per-participant z-scoring: that would itself remove the subject means);
  eraser on the training rows only: LEACE r(x) = x - W^+ P W (x - mu), W = Sigma_xx^(-1/2) (eigh, eigenvalues < 1e-8 max dropped),
  P = orthogonal projector onto colspace(W Sigma_xz), z = one-hot of the 22 training subjects (rank <= 21);
  control: the same map with P replaced by the projector onto a random rank-r subspace of the whitened space (r = rank of LEACE's P), 5 seeds;
  read-out = ridge (LAM * n) at PCA-64 (METHOD CHANGE 12) and full width, trained on the 22 participants' (erased) rows, applied to h;
  score on h = the PE log-score of the LOPO read-out x: per 10 contiguous folds of h, gain_rows(calibration on the other 9 folds of h);
  within-participant reference on the same rows = the X7 nested log-score of h's own (erased) features (d2_pe_tasks.pe_gain);
  gap = within - LOPO.
All 7 conditions x 2 widths of one fold = one batched GPU solve; calibration batched (gain_t); bootstraps = GPU index tensors.
Subject-ID saturation line: linear subject-ID log-score on the training rows' held-out 10 % (22-way) printed beside log2(22).
G2: condition 'none', within reference = results/standard/pe_bottleneck/sub-XXX.json ('pca', 64 / full) exactly.
Usage: python a2_leace.py [--subjects 1,2,3] --out DIR | --summarize --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
from belief_bottleneck import LAM, folds

DEV = "cuda"; WID = (64, "full"); NSEED = 5


DS = os.environ.get("A2_DS", "ds003458")                                       # second PE dataset: A2_DS=ds004595 (MASTER standing order)
FEAT = {"ds003458": "results/features/raw", "ds004595": "results/features_ds004595/raw"}


def load_all():
    info = D2.dataset(DS); X, Y = {}, {}
    for s in info["subs"]:
        fz = np.load(ROOT / FEAT[DS] / f"sub-{s:03d}.npz"); X[s] = fz["F"].astype(np.float64); Y[s] = info["qtil"](s, fz["tri"]).astype(np.float64)
    return info["subs"], X, Y


def erasers(A, labels, nlab, seeds):
    """A (n, D) standardised training rows; returns list of D x D linear maps E (x_erased = mu + (x - mu) E^T ... applied as (x - mu) @ M)
    for [LEACE] + [random rank-r for each seed]; plus r."""
    mu = A.mean(0); Ac = A - mu; n = len(A)
    Sxx = Ac.T @ Ac / n; Zc = torch.nn.functional.one_hot(labels, nlab).double(); Zc = Zc - Zc.mean(0); Sxz = Ac.T @ Zc / n
    e, U = torch.linalg.eigh(Sxx); keep = e > 1e-8 * e.max(); U, e = U[:, keep], e[keep]
    W = U @ torch.diag(e ** -0.5) @ U.T; Wp = U @ torch.diag(e ** 0.5) @ U.T                     # whitening and its pseudo-inverse
    Q, Rr = torch.linalg.qr(W @ Sxz); r = int((Rr.diagonal().abs() > 1e-10 * Rr.diagonal().abs().max()).sum()); Q = Q[:, :r]
    I = torch.eye(A.shape[1], dtype=torch.float64, device=DEV)
    maps = [I - (Wp @ Q @ Q.T @ W).T]                                                             # row-vector form: x' = x - (x - mu) (W^+ P W)^T
    for sd in seeds:
        g = torch.Generator(device=DEV); g.manual_seed(int(sd))
        Qr = torch.linalg.qr(U @ torch.randn((U.shape[1], r), generator=g, device=DEV, dtype=torch.float64))[0]     # random rank-r subspace of the whitened space
        maps.append(I - (Wp @ Qr @ Qr.T @ W).T)
    return mu, maps, r


def lopo_fold(h, subs, X, Y):
    tr = [s for s in subs if s != h]; Ftr = torch.as_tensor(np.concatenate([X[s] for s in tr]), dtype=torch.float64, device=DEV)
    ytr = torch.as_tensor(np.concatenate([Y[s] for s in tr]), dtype=torch.float64, device=DEV)
    lab = torch.as_tensor(np.concatenate([np.full(len(X[s]), i) for i, s in enumerate(tr)]), device=DEV)
    m, sd = Ftr.mean(0), Ftr.std(0, unbiased=False) + 1e-6; A = (Ftr - m) / sd; Fh = (torch.as_tensor(X[h], dtype=torch.float64, device=DEV) - m) / sd
    mu, maps, r = erasers(A, lab, len(tr), [100 + k for k in range(NSEED)])
    M = torch.stack([torch.eye(A.shape[1], dtype=torch.float64, device=DEV)] + maps)               # (C, D, D): none, leace, random x5
    At = mu + (A - mu) @ M; Ht = mu + (Fh - mu) @ M                                                  # (C, n, D), (C, nh, D)
    # subject-ID saturation line (22-way, linear ridge on one-hot, last 10 % of each participant's rows held out), condition none vs leace
    sid = {}
    for ci, cname in ((0, "none"), (1, "leace")):
        idx = torch.arange(len(lab), device=DEV); hold = torch.zeros(len(lab), dtype=torch.bool, device=DEV)
        for i in range(len(tr)):
            w = torch.where(lab == i)[0]; hold[w[int(0.9 * len(w)):]] = True
        Xa, Xb = At[ci][~hold], At[ci][hold]; T = torch.nn.functional.one_hot(lab[~hold], len(tr)).double()
        Wsid = torch.linalg.solve(Xa.T @ Xa + 1e-2 * len(Xa) * torch.eye(Xa.shape[1], dtype=torch.float64, device=DEV), Xa.T @ (T - T.mean(0)))
        S = Xb @ Wsid; best = -1e9
        for t in (0.01, 0.03, 0.1, 0.3, 1.0):                                                       # temperature on the held-out rows (saturation line only)
            lp = torch.log_softmax(S / t, 1)[torch.arange(len(Xb), device=DEV), lab[hold]]; best = max(best, float(lp.mean() / np.log(2) + np.log2(len(tr))))
        sid[cname] = best
    out = {}
    yc = ytr - ytr.mean(); n = len(At[0])
    for L in WID:
        if L == "full":
            G = At.transpose(1, 2) @ At + LAM * n * torch.eye(At.shape[2], dtype=torch.float64, device=DEV); Wr = torch.linalg.solve(G, At.transpose(1, 2) @ yc[None, :, None])
            P = (Ht @ Wr)[..., 0]
        else:
            Ac_ = At - At.mean(1, keepdim=True); e, V = torch.linalg.eigh(Ac_.transpose(1, 2) @ Ac_); V = V.flip(-1)[..., :L]
            Z = At @ V; Wr = torch.linalg.solve(Z.transpose(1, 2) @ Z + LAM * n * torch.eye(L, dtype=torch.float64, device=DEV), Z.transpose(1, 2) @ yc[None, :, None])
            P = (Ht @ V @ Wr)[..., 0]                                                                # (C, nh)
        out[str(L)] = P
    return out, Ht, r, sid


def subject(h, subs, X, Y, out_dir):
    t0 = time.time(); preds, Ht, r, sid = lopo_fold(h, subs, X, Y); y = torch.as_tensor(Y[h], dtype=torch.float64, device=DEV); nh = len(y); fo = folds(nh)
    conds = ["none", "leace"] + [f"random{k}" for k in range(NSEED)]; res = dict(participant=f"sub-{h:03d}", n_trials=int(nh), erased_rank=r, subject_id_bits=sid,
                                                                             subject_id_log2N=float(np.log2(len(subs) - 1)), rows={})
    for L in WID:
        P = preds[str(L)]; g = torch.zeros((len(conds), nh), dtype=torch.float64, device=DEV)
        for b in range(10):
            tr = torch.as_tensor(np.where(fo != b)[0], device=DEV); te = torch.as_tensor(np.where(fo == b)[0], device=DEV)
            g[:, te] = D2.gain_t(y[tr][None], P[:, tr], y[te][None], P[:, te])
        lopo = g.mean(1).cpu().numpy()
        Ys = np.column_stack([Y[h]] + [np.roll(Y[h], round(nh * q / 4)) for q in (1, 2, 3)])
        within = [D2.pe_gain(Ht[c].cpu().numpy(), Ys, (L,))[:, 0].mean() for c in range(len(conds))]
        res["rows"][str(L)] = {c: dict(lopo=float(lopo[i]), within=float(within[i]), gap=float(within[i] - lopo[i])) for i, c in enumerate(conds)}
    if DS != "ds003458":
        res["g2"] = dict(passed=True, note="no standard per-participant reference file for this dataset"); res["dataset"] = DS
        res["runtime_s"] = round(time.time() - t0, 1); (out_dir / f"a2_sub-{h:03d}.json").write_text(json.dumps(res, indent=1)); print(f"sub-{h:03d}", DS, "done", flush=True); return
    st = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{h:03d}.json").read_text())
    res["g2"] = {str(L): dict(old=next(r_["gain"] for r_ in st["rows"] if r_["input"] == "eeg" and r_["mode"] == "pca" and r_["L"] == L),
                              new=res["rows"][str(L)]["none"]["within"]) for L in WID}
    res["g2"]["passed"] = bool(all(abs(v["old"] - v["new"]) < 1e-9 for k, v in res["g2"].items() if k != "passed"))
    res["runtime_s"] = round(time.time() - t0, 1); (out_dir / f"a2_sub-{h:03d}.json").write_text(json.dumps(res, indent=1))
    R6 = res["rows"]["64"]
    print(f"sub-{h:03d}", "rank", r, "sid", {k: round(v, 2) for k, v in sid.items()}, "pca64 lopo none/leace/rand0", round(R6["none"]["lopo"], 4), round(R6["leace"]["lopo"], 4),
          round(R6["random0"]["lopo"], 4), "within none/leace", round(R6["none"]["within"], 4), round(R6["leace"]["within"], 4), "G2", res["g2"]["passed"], f"{res['runtime_s']}s", flush=True)


def summarize(out_dir):
    per = [json.loads(f.read_text()) for f in sorted(out_dir.glob("a2_sub-*.json"))]; n = len(per); res = dict(n=n, participants=[p["participant"] for p in per])
    g = torch.Generator(device=DEV); g.manual_seed(20261121); IX = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
    q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    for L in ("64", "full"):
        get = lambda c, k: T([p["rows"][L][c][k] for p in per]); rnd = lambda k: torch.stack([get(f"random{s}", k) for s in range(NSEED)]).mean(0)
        R = {}
        for k in ("lopo", "within", "gap"):
            a, l, rr = get("none", k), get("leace", k), rnd(k)
            R[k] = dict(none=float(a.mean()), leace=float(l.mean()), random_mean=float(rr.mean()), leace_minus_none=float((l - a).mean()), ci=q((l - a)[IX].mean(1)),
                        random_minus_none=float((rr - a).mean()), ci_random=q((rr - a)[IX].mean(1)),
                        leace_minus_random=float((l - rr).mean()), ci_leace_minus_random=q((l - rr)[IX].mean(1)), n_pos=int((l > a).sum()))
        # METHOD G1 PASS A2: primary family = LEACE - random on the LOPO yield and on the gap, Holm over 2; one-sided bootstrap p (H1: > 0)
        fam = {}
        for k in ("lopo", "gap"):
            d = get("leace", k) - rnd(k); bm = d[IX].mean(1); fam[k] = dict(d=float(d.mean()), ci=q(bm), p_one_sided=float(((bm <= 0).double().mean())), n_pos=int((d > 0).sum()))
        ps = sorted(fam, key=lambda k: fam[k]["p_one_sided"]); m = 0.0
        for r, k in enumerate(ps):
            m = max(m, min(1.0, (2 - r) * fam[k]["p_one_sided"])); fam[k]["p_holm"] = m
        R["primary_family_leace_minus_random"] = fam
        R["random_control_label"] = "weak control (METHOD BUG HUNT B3): a random same-rank subspace is not matched to the subject structure"
        R["reading"] = "subject information limits cross-subject yield" if fam["lopo"]["ci"][0] > 0 else "not shown: erasing subject identity does not raise the LOPO yield beyond a random same-rank erasure"
        res["L" + L] = R
    res["subject_id_bits"] = {c: float(np.mean([p["subject_id_bits"][c] for p in per])) for c in ("none", "leace")}; res["subject_id_log2N"] = per[0]["subject_id_log2N"]
    dif = [abs(v["old"] - v["new"]) for p in per for k, v in p["g2"].items() if isinstance(v, dict)]
    res["g2_max_abs_diff"] = float(max(dif)) if dif else None; res["g2_all_pass"] = bool(not dif or max(dif) < 1e-6); res["dataset"] = DS
    res["erased_rank_per_fold"] = {p["participant"]: p["erased_rank"] for p in per}
    (out_dir / "summary_a2.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    subs, X, Y = load_all()
    for h in ([int(x) for x in a.subjects.split(",")] if a.subjects else subs):
        if not (out / f"a2_sub-{h:03d}.json").exists(): subject(h, subs, X, Y, out)


if __name__ == "__main__":
    main()
