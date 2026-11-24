"""X1 CROSS-SUBJECT ALIGNMENT, ds003458 LOPO (METHOD "## G1 SPEC X1", 18:15, USER-APPROVED; rung R9). FIXER, 21 Nov 2026.
Path = d3_ridge_transfer.py lopo3458 (imported, not copied): the 60 shared channels, 0-600 ms at 50 Hz; source = the OTHER 22
participants, each z-scored with its own statistics (label-free), stacked; ridge LAM * n_rows. Held-out participant h: the D3 outer split
= first 20 % of h's trials = h's TRAINING rows (standardisation, recalibration, and here U_h), the rest = test rows; score = D2.gain_t
(calibration on the training rows, log2 density gain on the test rows).
Per h and r in {8, 16, 32 (PRIMARY), 64}:
  U_c = top-r right singular vectors of the centred stacked source matrix (= D3's PCA basis); beta_r = D3's ridge on Z = A U_c.
  U_h = top-r right singular vectors of h's centred training rows (label-free; the test rows never enter).
  R   = V W^T from the SVD U_h^T U_c = V S W^T (orthogonal Procrustes); principal angles = arccos(S).
  arms: LOPO_full = D3.apply_target, all widths (G2 vs results/ext/d3/lopo3458_sub-XXX.json, |d| < 1e-9);
        LOPO_r    = z_h U_c beta_r (= D3's width-r read-out);
        ALIGNED_r = z_h U_h R beta_r;
        RANDOM_r  = z_h U_h Q beta_r, Q random orthogonal (QR of N(0,1), 5 seeds, gains averaged);
        WITHIN    = D3 within_test_rows (X7 PCA-64 ridge, 10 contiguous folds on the test rows);
        EA (secondary) = Euclidean alignment in channel space (each trial X_i -> C^{-1/2} X_i, C = mean_i X_i X_i^T; h: training rows
                     only, others: all rows), then the D3 full-width LOPO ridge.
Floors per arm: h's target circularly shifted (n/4, n/2, 3n/4, as D3) and N(0,1) target-side input (seed 7000 + h).
Primary: recovered_r = (ALIGNED_r - LOPO_r) / (WITHIN - LOPO_r) at r = 32, ratio of participant means, participant bootstrap (10,000, GPU).
  Falsifier: CI upper < 0.25. Main-text bar: CI lower > 0.5. Also ALIGNED_r - RANDOM_r (paired) and LOPO_r vs LOPO_full.
  Holm over {recovered > 0, ALIGNED - RANDOM > 0} (bootstrap p).
Usage: python -I -u x1_alignment.py [--subjects 1,2,3] --out results/ext/x1_alignment ; --summarize --out DIR"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext"))
import d2_pe_tasks as D2
import d3_ridge_transfer as D3
from belief_bottleneck import LAM
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
DEV = "cuda"; RS = [8, 16, 32, 64]; PRIMARY_R = 32; NRAND = 5


def source(IA, chans, exclude, ea=False):
    """Stacked z-scored source rows (as D3.fit_source); ea=True applies channel-space Euclidean alignment per participant first."""
    Fs, Ys = [], []
    for s in IA["subs"]:
        if s == exclude: continue
        F, tri = D3.feats(IA, s, chans, 0.0, 0.60); F = torch.as_tensor(F, dtype=torch.float64, device=DEV)
        if ea: F = ea_apply(F, ea_matrix(F, len(chans)), len(chans))
        Fs.append((F - F.mean(0)) / (F.std(0) + 1e-6)); Ys.append(torch.as_tensor(IA["qtil"](s, tri), dtype=torch.float64, device=DEV))
    return torch.cat(Fs), torch.cat(Ys)


def ea_matrix(F, nch):
    X = F.reshape(len(F), nch, -1); C = torch.einsum("nct,ndt->cd", X, X) / len(X)
    e, U = torch.linalg.eigh(C); return U @ torch.diag(e.clamp_min(1e-10 * e.max()) ** -0.5) @ U.T


def ea_apply(F, M, nch):
    return torch.einsum("cd,ndt->nct", M, F.reshape(len(F), nch, -1)).reshape(len(F), -1)


def score(x, Ysh, nd):
    """x (B, n) read-outs -> (B, 4) gains [true, 3 shifts] with D3's calibration on the first nd rows."""
    return D2.gain_t(Ysh[None, :, :nd], x[:, None, :nd], Ysh[None, :, nd:], x[:, None, nd:]).mean(-1)


def subject(s, out, IA, chans):
    t0 = time.time()
    # (a) LOPO_full: the registered D3 path, unchanged
    W, nA = D3.fit_source(IA, chans, exclude=s); Wn, _ = D3.fit_source(IA, chans, seed_noise=5000, exclude=s)
    d3 = D3.apply_target(W, Wn, IA, "ds003458", s, chans, D2.common_channels(IA), "lopo3458", dict(direction="lopo3458", source="ds003458", target="ds003458", arm="lopo_within", source_rows=int(nA)))
    reg = json.loads((ROOT / f"results/ext/d3/lopo3458_sub-{s:03d}.json").read_text())
    g2 = max(abs(reg["transfer"][L]["gain"] - d3["transfer"][L]["gain"]) for L in reg["transfer"])
    # source basis and ridge per r (D3's construction)
    A, y = source(IA, chans, s); yc = y - y.mean(); n = len(A)
    _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False)
    F, tri = D3.feats(IA, s, chans, 0.0, 0.60); yh = IA["qtil"](s, tri).astype(np.float64); nh = len(yh); nd = int(round(0.2 * nh))
    Fg = torch.as_tensor(F, dtype=torch.float64, device=DEV); Zg = (Fg - Fg[:nd].mean(0)) / (Fg[:nd].std(0) + 1e-6)
    yg = torch.as_tensor(yh, dtype=torch.float64, device=DEV); Ysh = torch.stack([yg] + [torch.roll(yg, round(nh * q / 4)) for q in (1, 2, 3)])
    gN = torch.Generator(device=DEV); gN.manual_seed(7000 + s); Zn = torch.randn(Zg.shape, generator=gN, dtype=torch.float64, device=DEV)
    _, _, Vth = torch.linalg.svd(Zg[:nd] - Zg[:nd].mean(0), full_matrices=False)                        # h's TRAINING rows only
    # v3 (IDEATOR item 2): U_h refit label-free on ALL of h's trials, and two U_h estimates from disjoint halves (odd / even trials)
    _, _, Vta = torch.linalg.svd(Zg - Zg.mean(0), full_matrices=False)
    _, _, Vto = torch.linalg.svd(Zg[0::2] - Zg[0::2].mean(0), full_matrices=False); _, _, Vte = torch.linalg.svd(Zg[1::2] - Zg[1::2].mean(0), full_matrices=False)
    res = dict(code_sha256=CODE_SHA256, participant=f"sub-{s:03d}", n_trials=nh, n_train=nd, n_test=nh - nd, source_rows=n, channels=len(chans), primary_r=PRIMARY_R,
               lopo_full=dict(transfer=d3["transfer"], g2_vs_registered_max_abs=g2, g2_pass=bool(g2 < 1e-9)), within=d3["within_test_rows"], r={})
    for r in RS:
        Uc = Vt[:r].T; Z = A @ Uc; beta = torch.linalg.solve(Z.T @ Z + LAM * n * torch.eye(r, dtype=torch.float64, device=DEV), Z.T @ yc)
        Uh = Vth[:r].T; V_, S_, Wt_ = torch.linalg.svd(Uh.T @ Uc); R = V_ @ Wt_
        gq = torch.Generator(device=DEV); gq.manual_seed(100 + r)
        Qs = [torch.linalg.qr(torch.randn((r, r), generator=gq, dtype=torch.float64, device=DEV))[0] for _ in range(NRAND)]
        Ua = Vta[:r].T; Va_, Sa_, Wa_ = torch.linalg.svd(Ua.T @ Uc); Ra = Va_ @ Wa_
        Sh_ = torch.linalg.svdvals(Vto[:r] @ Vte[:r].T); half_ang = torch.rad2deg(torch.arccos(Sh_.clamp(-1, 1)))
        maps = [Uc @ beta, Uh @ R @ beta, Uh @ (Uh.T @ (Uc @ beta)), Ua @ Ra @ beta, Ua @ (Ua.T @ (Uc @ beta))] + [Uh @ Q @ beta for Q in Qs]   # + PROJECT_r, + v3 all-trial arms                                     # (D,) read-out vectors
        Mv = torch.stack(maps, 1); G = score((Zg @ Mv).T, Ysh, nd); Gn = score((Zn @ Mv).T, Ysh, nd)       # (2 + NRAND, 4)
        ang = torch.rad2deg(torch.arccos(S_.clamp(-1, 1)))
        arm = lambda i: dict(gain=float(G[i, 0]), shifted=G[i, 1:].tolist(), shift_mean=float(G[i, 1:].mean()), noise=float(Gn[i, 0]))
        res["r"][str(r)] = dict(lopo_r=arm(0), aligned_r=arm(1), project_r=arm(2), aligned_all_r=arm(3), project_all_r=arm(4), random_r=dict(gain=float(G[5:, 0].mean()), gain_per_seed=G[5:, 0].tolist(), shift_mean=float(G[5:, 1:].mean()), noise=float(Gn[5:, 0].mean())),
                                split_half_angles_deg=dict(mean=float(half_ang.mean()), max=float(half_ang.max())), all_vs_consensus_angles_deg=float(torch.rad2deg(torch.arccos(Sa_.clamp(-1, 1))).mean()),
                                principal_angles_deg=dict(mean=float(ang.mean()), max=float(ang.max())),
                                lopo_r_equals_d3_width=None if str(r) not in d3["transfer"] else abs(float(G[0, 0]) - d3["transfer"][str(r)]["gain"]))
    # EA (secondary): channel-space Euclidean alignment, then the full-width LOPO ridge
    Ae, ye = source(IA, chans, s, ea=True); yce = ye - ye.mean(); ne = len(Ae)
    We = Ae.T @ torch.linalg.solve(Ae @ Ae.T + LAM * ne * torch.eye(ne, dtype=torch.float64, device=DEV), yce) if Ae.shape[1] >= ne else torch.linalg.solve(Ae.T @ Ae + LAM * ne * torch.eye(Ae.shape[1], dtype=torch.float64, device=DEV), Ae.T @ yce)
    Fe = ea_apply(Fg, ea_matrix(Fg[:nd], len(chans)), len(chans)); Ze = (Fe - Fe[:nd].mean(0)) / (Fe[:nd].std(0) + 1e-6)
    Ge = score((Ze @ We)[None], Ysh, nd)[0]; res["ea_full"] = dict(gain=float(Ge[0]), shifted=Ge[1:].tolist(), label="secondary: Euclidean alignment (channel space), full-width ridge")
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"sub-{s:03d}.json").write_text(json.dumps(res, indent=1))
    p = res["r"][str(PRIMARY_R)]
    print(res["participant"], "G2", res["lopo_full"]["g2_pass"], f"{g2:.1e}", "r32 LOPO_r", round(p["lopo_r"]["gain"], 4), "ALIGNED", round(p["aligned_r"]["gain"], 4), "PROJECT", round(p["project_r"]["gain"], 4), "RANDOM", round(p["random_r"]["gain"], 4),
          "WITHIN", round(res["within"]["gain"], 4), "EA", round(res["ea_full"]["gain"], 4), "angles", {k: round(v, 1) for k, v in p["principal_angles_deg"].items()}, f"{res['runtime_s']}s", flush=True)
    del A, Ae; torch.cuda.empty_cache()


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P)
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); q = lambda b: torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
    res = dict(note=__doc__, code_sha256=CODE_SHA256, n=n, participants=[p["participant"] for p in P], g2_all_pass=all(p["lopo_full"]["g2_pass"] for p in P), r={})
    W = T([p["within"]["gain"] for p in P]); res["within_mean"] = float(W.mean())
    for r in RS:
        L, Al, Rn, Pj = [T([p["r"][str(r)][a]["gain"] for p in P]) for a in ("lopo_r", "aligned_r", "random_r", "project_r")]; dp = Al - Pj; bp = dp[ix].mean(1)
        v3 = {}
        if "aligned_all_r" in P[0]["r"][str(r)]:
            Aa = T([p["r"][str(r)]["aligned_all_r"]["gain"] for p in P]); na = Aa - L; ba = na[ix].mean(1) / (W - L)[ix].mean(1)
            v3 = dict(aligned_all_r=float(Aa.mean()), project_all_r=float(np.mean([p["r"][str(r)]["project_all_r"]["gain"] for p in P])),
                      recovered_all=dict(ratio_of_means=float(na.mean() / (W - L).mean()), ci=q(ba)),
                      split_half_angles_mean_deg=float(np.mean([p["r"][str(r)]["split_half_angles_deg"]["mean"] for p in P])),
                      all_vs_consensus_angles_mean_deg=float(np.mean([p["r"][str(r)]["all_vs_consensus_angles_deg"] for p in P])),
                      change12_gate={a: dict(shift=D2.g3_test([p["r"][str(r)][a]["shift_mean"] for p in P]), noise=D2.g3_test([p["r"][str(r)][a]["noise"] for p in P]))
                                     for a in ("lopo_r", "aligned_r", "project_r", "aligned_all_r", "project_all_r", "random_r")})
        num, den = Al - L, W - L; b = num[ix].mean(1) / den[ix].mean(1); dr = (Al - Rn); bd = dr[ix].mean(1)
        res["r"][str(r)] = dict(lopo_r=float(L.mean()), aligned_r=float(Al.mean()), random_r=float(Rn.mean()), recovered=dict(ratio_of_means=float(num.mean() / den.mean()), ci=q(b), p_le0=float((b <= 0).double().mean())),
                                aligned_minus_random=dict(d=float(dr.mean()), ci=q(bd), p_le0=float((bd <= 0).double().mean()), n_pos=int((dr > 0).sum())),
                                **v3, project_r=float(Pj.mean()), aligned_minus_project=dict(d=float(dp.mean()), ci=q(bp), n_pos=int((dp > 0).sum())),
                                lopo_r_minus_lopo_full64=float((L - T([p["lopo_full"]["transfer"]["64"]["gain"] for p in P])).mean()),
                                principal_angles_mean_deg=float(np.mean([p["r"][str(r)]["principal_angles_deg"]["mean"] for p in P])),
                                floors=dict(shift_max=float(max(abs(p["r"][str(r)][a]["shift_mean"]) for p in P for a in ("lopo_r", "aligned_r", "random_r", "project_r"))),
                                            noise_max=float(max(abs(p["r"][str(r)][a]["noise"]) for p in P for a in ("lopo_r", "aligned_r", "random_r", "project_r")))))
    pr = res["r"][str(PRIMARY_R)]; ps = [pr["recovered"]["p_le0"], pr["aligned_minus_random"]["p_le0"]]; o = np.argsort(ps); adj = [0.0, 0.0]; run_ = 0.0
    for k, i in enumerate(o): run_ = max(run_, min(1.0, ps[i] * (2 - k))); adj[i] = run_
    res["holm_primary"] = dict(recovered=adj[0], aligned_minus_random=adj[1])
    ci = pr["recovered"]["ci"]
    res["reading"] = ("falsified: the cross-subject loss is not a rotation of a shared subspace (missing / idiosyncratic signal)" if ci[1] < 0.25 else
                      "main-text bar met (CI lower > 0.5)" if ci[0] > 0.5 else "partial / inconclusive (appendix)")
    res["ea_full_mean"] = float(np.mean([p["ea_full"]["gain"] for p in P])); res["lopo_full64_mean"] = float(np.mean([p["lopo_full"]["transfer"]["64"]["gain"] for p in P]))
    (out / "summary_x1.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", default=str(ROOT / "results/ext/x1_alignment")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    IA = D2.dataset("ds003458"); ca, cb = D2.common_channels(IA), D2.common_channels(D2.dataset("ds004595")); chans = [c for c in ca if c in cb]   # D3's 60 channels
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else IA["subs"]):
        if not (out / f"sub-{s:03d}.json").exists(): subject(s, out, IA, chans)


if __name__ == "__main__":
    main()
