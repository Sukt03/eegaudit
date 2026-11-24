"""W1 WITHIN-CATEGORY TRIAL-SWAP COUNTERFACTUAL (METHOD "WORKSHOP IDEAS RULING", W1 trial-swap point). FIXER, 21 Nov 2026.
Frozen decoders: NICE-GA (seed 141, CLIP class centres) and BrainFLORA (released, ViT-H test features), via s1_averaging.models on the
THINGS-EEG2 test cache; UBP (released intra-subject checkpoint, whitened 17 posterior channels, RN50 'medium'), as ubp_card.py.
For each query image i with an exact-category-profile partner (pools >= 2 images, as KM3/S1), the decoder's latent is computed from
  matched : i's own EEG;
  within  : the EEG of a random OTHER image j of i's exact-profile pool (20 draws);
  across  : the EEG of a random image j outside i's pool (20 draws; the category-free reference);
and scored as 200-way identification of i (InfoNCE, LOO temperature over the queries on the LOCKED soft12 grid run._TAUS, 10^-2.5..1).
k80 (all repetitions averaged) PRIMARY; k1 (single repetitions 0..19, averaged scores) secondary (NICE-GA / BrainFLORA only).
Share of the matched yield delivered by category structure alone = (within - across) / (matched - across), ratio of participant means,
participant bootstrap CI (10,000, GPU).
v3 (FIXER-3; REVIEWER overnight D, fixes 1-5):
 1. ident() uses the locked soft12 grid (run._TAUS) directly (v1 ran on 1e-4.5, v2 on 1e-6, unrecorded). Each JSON records tau_grid,
    s1_code_sha256 and per construction (matched / within / across) n_q_lower_edge and max_query_CE_bits.
 2. Oracle: sigma is found on the MEAN matched yield over the 5 evaluation noise draws (the same draws that are then scored): a coarse
    log-sigma scan (-4..3, step 0.25) brackets the LAST crossing of the target (the high-noise side, where the statistic is monotone), then
    30 bisection steps inside the bracket. If |oracle matched - decoder matched| > 0.10 bits the cell is written with oracle_match_failed =
    true and the participant is dropped from that oracle contrast (printed in the summary). sigma and the achieved gap are stored.
 3. G2: each decoder's 200-query k80 yield (all 200 images as queries) vs its registered value (NICE-GA r1 and BrainFLORA registered
    k80 from the s1_v6 participant file g2 block; UBP released soft12 card k80_bits), printed in the summary.
 4. Estimand amendment (logged, accepted in METHOD's 17:30 wording): a 200-way identification share replaces the spec's "drop in exemplar yield".
 5. decoder_minus_oracle_share is a MEAN OF PER-PARTICIPANT RATIOS (labelled); the shares themselves are ratios of participant means.
Output results/ext/w1_v3/{sub-XX.json, summary_w1.json} with code_sha256.
Usage: python -I -u w1_trialswap.py --subjects 1,2,3 --out DIR ; --summarize --out DIR"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT / "experiments/baselines"))
import s1_averaging as S1
import run as R                                                                            # locked soft12 grid (LOCK §3)
from safe_load import safe_load
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
NICE_IMG = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature/clip_feature_maps_test.npy")
DEV = "cuda"; ND = 20; TAUS = torch.as_tensor(R._TAUS, dtype=torch.float64, device="cuda"); MATCH_TOL = 0.10; UD = ROOT / "data_raw/third_party/ubp"
CH = open(ROOT / "experiments/baselines/ubp_card.py").read()
CH = eval(CH[CH.index("CH = ") + 5:CH.index("\nSEL = ")])                                  # the card's channel list, verbatim
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]


class Res(torch.nn.Module):
    def __init__(self, f):
        super().__init__(); self.f = f
    def forward(self, x):
        return x + self.f(x)


class Enc(torch.nn.Module):  # = ubp_card.Enc
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024), Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))), torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))


def ident(S, true, diag=False):
    """S (B, nq, 200) cosines; true (nq,) column of the true image -> (B,) InfoNCE bits, LOO temperature over the nq queries (soft12 grid).
    diag=True also returns dict(n_q_lower_edge, n_q, max_query_CE_bits)."""
    T = TAUS; nq = S.shape[1]; lp = torch.log_softmax(S[:, None] / T[None, :, None, None], -1)
    CE = -lp[..., torch.arange(nq, device=DEV), true] / math.log(2); loo = (CE.sum(-1, keepdim=True) - CE) / (nq - 1)
    ti = loo.argmin(1); per = torch.gather(CE, 1, ti[:, None])[:, 0]; v = math.log2(S.shape[-1]) - per.mean(-1)
    if not diag: return v
    return v, dict(n_q_lower_edge=int((ti == 0).sum()), n_q=int(ti.numel()), max_query_CE_bits=float(per.max()))


def merge_diag(ds):
    return dict(n_q_lower_edge=sum(d["n_q_lower_edge"] for d in ds), n_q=sum(d["n_q"] for d in ds), max_query_CE_bits=max(d["max_query_CE_bits"] for d in ds))


def swap_scores(E, Y, q, pool, rng):
    """E (B, 200, d) normalised latents, Y (200, d) normalised candidates; q query indices -> dict of bits (B-averaged)."""
    qt = torch.as_tensor(q, device=DEV); S = E @ Y.T
    within = np.stack([[rng.choice([j for j in pool[i] if j != i]) for i in q] for _ in range(ND)])
    across = np.stack([[rng.choice([j for j in range(200) if j not in pool[i]]) for i in q] for _ in range(ND)])
    m, dm = ident(S[:, qt], qt, True); m = m.mean()
    W_ = [ident(S[:, torch.as_tensor(jj, device=DEV)], qt, True) for jj in within]; w = torch.stack([x[0].mean() for x in W_])
    A_ = [ident(S[:, torch.as_tensor(jj, device=DEV)], qt, True) for jj in across]; a = torch.stack([x[0].mean() for x in A_])
    return dict(matched=float(m), within=float(w.mean()), within_sd=float(w.std()), across=float(a.mean()), across_sd=float(a.std()),
                share_category=float((w.mean() - a.mean()) / (m - a.mean())) if float(m - a.mean()) != 0 else None,
                diag=dict(matched=dm, within=merge_diag([x[1] for x in W_]), across=merge_diag([x[1] for x in A_])))


def oracle(Y, target, q, pool, rng, seed, nseed=5):
    """noisy-oracle latents nrm(Y + sigma * N(0,1)) with sigma matched so the oracle's MATCHED k80 yield, averaged over the nseed evaluation
    noise draws (seeds seed+1..seed+nseed), equals the decoder's (v3: scan-bracketed bisection on that mean; see the module note, fix 2).
    The oracle carries image-specific information only (no extra category emphasis), so its category share is the geometry baseline."""
    qt = torch.as_tensor(q, device=DEV)
    def lat(sig, sd):
        g = torch.Generator(device=DEV); g.manual_seed(sd); return S1.nrm(Y + sig * torch.randn(Y.shape, generator=g, dtype=Y.dtype, device=DEV))
    def mm(x):
        return float(np.mean([float(ident((lat(10 ** x, seed + 1 + j) @ Y.T)[None][:, qt], qt).mean()) for j in range(nseed)]))
    grid = np.arange(-4.0, 3.0 + 1e-9, 0.25); vals = [mm(x) for x in grid]
    cross = [i for i in range(len(grid) - 1) if vals[i] > target >= vals[i + 1]]
    if cross:
        lo, hi = grid[cross[-1]], grid[cross[-1] + 1]
        for _ in range(30):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if mm(mid) > target else (lo, mid)
        x = (lo + hi) / 2
    else:
        x = float(grid[int(np.argmin([abs(v - target) for v in vals]))])
    sig = 10 ** x; R_ = [swap_scores(lat(sig, seed + 1 + j)[None], Y, q, pool, rng) for j in range(nseed)]
    out_ = {k: float(np.mean([r[k] for r in R_])) for k in ("matched", "within", "across")}; out_["sigma"] = sig
    out_["share_category"] = (out_["within"] - out_["across"]) / (out_["matched"] - out_["across"])
    out_["target_matched"] = float(target); out_["achieved_gap"] = out_["matched"] - float(target); out_["bracket_found"] = bool(cross)
    out_["oracle_match_failed"] = bool(abs(out_["achieved_gap"]) > MATCH_TOL); out_["diag_matched"] = merge_diag([r["diag"]["matched"] for r in R_])
    out_["scan"] = dict(log10_sigma=grid.tolist(), mean_matched=vals); return out_


def subject(s, out):
    t0 = time.time(); sub = f"sub-{s:02d}"; te = S1.categories(); keyed = {}
    for i, r in enumerate(te):
        if r.sum() > 0: keyed.setdefault(tuple(r), []).append(i)
    pool = {i: v for v in keyed.values() for i in v if len(v) >= 2}; q = sorted(pool); rng = np.random.default_rng(1000 + s)
    res = dict(participant=sub, n_queries=len(q), n_draws=ND, code_sha256=CODE_SHA256, s1_code_sha256=S1.CODE_SHA256, tau_grid=TAUS.tolist(),
               match_tol_bits=MATCH_TOL, g2_yield200_k80={}, models={})
    X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); MD = S1.models(sub, DEV)
    for m, (ft, _, Yc) in MD.items():
        Y = S1.nrm(torch.as_tensor(Yc, device=DEV))
        E80 = S1.nrm(S1.embed(ft, np.asarray(X.mean(1))))[None]
        E1 = torch.stack([S1.nrm(S1.embed(ft, np.asarray(X[:, r]))) for r in range(20)])
        res["models"][m] = dict(k80=swap_scores(E80, Y, q, pool, rng), k1=swap_scores(E1, Y, q, pool, rng))
        res["g2_yield200_k80"][m] = float(ident((E80 @ Y.T)[:, torch.arange(200, device=DEV)], torch.arange(200, device=DEV))[0])
        res["models"][m]["k80_oracle"] = oracle(Y, res["models"][m]["k80"]["matched"], q, pool, rng, 5000 + s)
        if m == "nice_ga":   # v2 geometry check: the same frozen NICE-GA latents against SINGLE-IMAGE CLIP test features
            Yi = S1.nrm(torch.as_tensor(np.squeeze(np.load(NICE_IMG)).astype(np.float64), device=DEV))
            res["models"]["nice_ga_clipimg"] = dict(k80=swap_scores(E80, Yi, q, pool, rng), k1=swap_scores(E1, Yi, q, pool, rng))
            res["models"]["nice_ga_clipimg"]["k80_oracle"] = oracle(Yi, res["models"]["nice_ga_clipimg"]["k80"]["matched"], q, pool, rng, 6000 + s)
    # UBP (released checkpoint, its own data and space)
    t = safe_load(str(UD / f"things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt")); imgs = list(np.asarray(t["img"])[:, 0]); assert imgs == sorted(imgs)
    meta = np.load(ROOT / "data_raw/things_eeg2_images/images_set/image_metadata.npy", allow_pickle=True).item()
    assert [p.replace("\\", "/").split("/")[-2] for p in imgs] == list(meta["test_img_concepts"])
    FT = safe_load(str(UD / "things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt"))["img_features"]
    Xu = torch.from_numpy(np.ascontiguousarray(np.asarray(t["eeg"])[:, :, SEL])).mean(1).float()[..., 0:250].to(DEV)
    ck = safe_load(str(UD / f"ckpt/sub-{s:02d}/last.ckpt")); enc = Enc().to(DEV); enc.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("brain.")}); enc.eval()
    with torch.no_grad():
        Eu = F.normalize(enc(Xu).float(), dim=1).double()[None]
    Yu = F.normalize(torch.stack([FT["medium"][p] for p in imgs]).float().to(DEV), dim=1).double()
    res["models"]["ubp"] = dict(k80=swap_scores(Eu, Yu, q, pool, rng)); res["g2_yield200_k80"]["ubp"] = float(ident(Eu @ Yu.T, torch.arange(200, device=DEV))[0]); res["models"]["ubp"]["k80_oracle"] = oracle(Yu, res["models"]["ubp"]["k80"]["matched"], q, pool, rng, 7000 + s)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, {m: {k: {a: round(b, 3) for a, b in v.items() if isinstance(b, float) and a in ("matched", "within", "across", "share_category", "achieved_gap")}
                    for k, v in d.items()} for m, d in res["models"].items()}, "g2", {m: round(v, 3) for m, v in res["g2_yield200_k80"].items()},
          "oracle_failed", [m for m, d in res["models"].items() if d.get("k80_oracle", {}).get("oracle_match_failed")], f"{res['runtime_s']}s", flush=True)


def g2_refs(sub, s1ref):
    """registered k80 yields: NICE-GA r1 and BrainFLORA registered (from the S1 participant file's g2 block), UBP released soft12 card."""
    r = {}
    f = s1ref / f"{sub}.json"
    if f.exists():
        g = json.loads(f.read_text()).get("g2", {}); r["nice_ga"] = g.get("r1"); r["brainflora"] = g.get("bf_registered")
    s = int(sub[4:]); u = ROOT / f"results/baselines/ubp/ubp_card_soft12_sub{s}.json"
    if u.exists(): r["ubp"] = json.loads(u.read_text())["subjects"][str(s)]["released"]["k80_bits"]
    return r


def summarize(out, s1ref):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
    q95 = lambda b: torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
    def boot(nn):
        g = torch.Generator(device=DEV); g.manual_seed(20261121); return torch.randint(0, nn, (10000, nn), generator=g, device=DEV)
    res = dict(note=__doc__, n=n, code_sha256=CODE_SHA256, s1_code_sha256=sorted({p.get("s1_code_sha256") for p in P}), tau_grid=P[0]["tau_grid"],
               participants=[p["participant"] for p in P], models={}, oracle_dropped={}, g2={}, max_query_CE_bits={}, n_q_lower_edge={},
               labels=dict(share_category="ratio of participant means, participant-bootstrap CI",
                           decoder_minus_oracle_share="MEAN OF PER-PARTICIPANT RATIOS (decoder share - oracle share), participants with oracle_match_failed dropped",
                           estimand="200-way identification share (amendment of the spec's 'drop in exemplar yield', METHOD 17:30)"))
    for m in P[0]["models"]:
        for k in P[0]["models"][m]:
            keep = [p for p in P if not (k == "k80_oracle" and p["models"][m][k].get("oracle_match_failed"))]
            if k == "k80_oracle":
                res["oracle_dropped"][m] = [(p["participant"], round(p["models"][m][k]["achieved_gap"], 4)) for p in P if p["models"][m][k].get("oracle_match_failed")]
                res["models"][f"{m}_k80_oracle_gap"] = dict(max_abs=max(abs(p["models"][m][k]["achieved_gap"]) for p in P), sigma=[p["models"][m][k]["sigma"] for p in P])
            if not keep: continue
            ix = boot(len(keep)); g_ = lambda a: T([p["models"][m][k][a] for p in keep]); M, W, A = g_("matched"), g_("within"), g_("across")
            b = (W - A)[ix].mean(1) / (M - A)[ix].mean(1)
            res["models"][f"{m}_{k}"] = dict(n=len(keep), matched=float(M.mean()), within=float(W.mean()), across=float(A.mean()),
                                            share_category=dict(ratio_of_means=float((W - A).mean() / (M - A).mean()), ci=q95(b)))
            if k != "k80_oracle":
                dg = [p["models"][m][k]["diag"] for p in P]
                res["max_query_CE_bits"][f"{m}_{k}"] = max(d[c]["max_query_CE_bits"] for d in dg for c in ("matched", "within", "across"))
                res["n_q_lower_edge"][f"{m}_{k}"] = sum(d[c]["n_q_lower_edge"] for d in dg for c in ("matched", "within", "across"))
            if k == "k80_oracle":   # decoder share minus the yield-matched oracle share (paired), mean of per-participant ratios
                d0 = lambda a: T([p["models"][m]["k80"][a] for p in keep]); Md, Wd, Ad = d0("matched"), d0("within"), d0("across")
                sd_, so_ = (Wd - Ad) / (Md - Ad), (W - A) / (M - A); dd = sd_ - so_
                res["models"][f"{m}_decoder_minus_oracle_share"] = dict(n=len(keep), mean_of_per_participant_ratios=float(dd.mean()), ci=q95(dd[ix].mean(1)), n_pos=int((dd > 0).sum()))
    for p in P:
        ref = g2_refs(p["participant"], s1ref)
        res["g2"][p["participant"]] = {m: dict(w1=v, registered=ref.get(m), abs_diff=None if ref.get(m) is None else abs(v - ref[m])) for m, v in p["g2_yield200_k80"].items()}
    dif = [d["abs_diff"] for x in res["g2"].values() for d in x.values() if d["abs_diff"] is not None]
    res["g2_max_abs_diff"] = max(dif) if dif else None; res["g2_n_compared"] = len(dif); res["g2_s1_ref"] = str(s1ref)
    (out / "summary_w1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(models=res["models"], oracle_dropped=res["oracle_dropped"], g2_max_abs_diff=res["g2_max_abs_diff"], g2_n=res["g2_n_compared"],
                          max_query_CE_bits=res["max_query_CE_bits"]), indent=1)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/w1_v3")); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--s1ref", default=str(ROOT / "results/ext/s1_v6"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out, Path(a.s1ref))
    else:
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists(): subject(s, out)
