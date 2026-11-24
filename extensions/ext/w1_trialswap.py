"""W1 WITHIN-CATEGORY TRIAL-SWAP COUNTERFACTUAL (METHOD "WORKSHOP IDEAS RULING", W1 trial-swap point). FIXER, 21 Nov 2026.
Frozen decoders: NICE-GA (seed 141, CLIP class centres) and BrainFLORA (released, ViT-H test features), via s1_averaging.models on the
THINGS-EEG2 test cache; UBP (released intra-subject checkpoint, whitened 17 posterior channels, RN50 'medium'), as ubp_card.py.
For each query image i with an exact-category-profile partner (pools >= 2 images, as KM3/S1), the decoder's latent is computed from
  matched : i's own EEG;
  within  : the EEG of a random OTHER image j of i's exact-profile pool (20 draws);
  across  : the EEG of a random image j outside i's pool (20 draws; the category-free reference);
and scored as 200-way identification of i (InfoNCE, LOO temperature over the queries on the S1 grid = soft12 extended to 1e-4.5).
k80 (all repetitions averaged) PRIMARY; k1 (single repetitions 0..19, averaged scores) secondary (NICE-GA / BrainFLORA only).
Share of the matched yield delivered by category structure alone = (within - across) / (matched - across), ratio of participant means,
participant bootstrap CI (10,000, GPU). Output results/ext/w1_trialswap/{sub-XX.json, summary_w1.json}.
Usage: python -I -u w1_trialswap.py --subjects 1,2,3 --out DIR ; --summarize --out DIR"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT / "experiments/baselines"))
import s1_averaging as S1
from safe_load import safe_load
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
NICE_IMG = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature/clip_feature_maps_test.npy")
DEV = "cuda"; ND = 20; UD = ROOT / "data_raw/third_party/ubp"
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


def ident(S, true):
    """S (B, nq, 200) cosines; true (nq,) column of the true image -> (B,) InfoNCE bits, LOO temperature over the nq queries."""
    T = S1.taus(); nq = S.shape[1]; lp = torch.log_softmax(S[:, None] / T[None, :, None, None], -1)
    CE = -lp[..., torch.arange(nq, device=DEV), true] / math.log(2); loo = (CE.sum(-1, keepdim=True) - CE) / (nq - 1)
    return math.log2(S.shape[-1]) - torch.gather(CE, 1, loo.argmin(1)[:, None])[:, 0].mean(-1)


def swap_scores(E, Y, q, pool, rng):
    """E (B, 200, d) normalised latents, Y (200, d) normalised candidates; q query indices -> dict of bits (B-averaged)."""
    qt = torch.as_tensor(q, device=DEV); S = E @ Y.T
    within = np.stack([[rng.choice([j for j in pool[i] if j != i]) for i in q] for _ in range(ND)])
    across = np.stack([[rng.choice([j for j in range(200) if j not in pool[i]]) for i in q] for _ in range(ND)])
    m = ident(S[:, qt], qt).mean()
    w = torch.stack([ident(S[:, torch.as_tensor(jj, device=DEV)], qt).mean() for jj in within])
    a = torch.stack([ident(S[:, torch.as_tensor(jj, device=DEV)], qt).mean() for jj in across])
    return dict(matched=float(m), within=float(w.mean()), within_sd=float(w.std()), across=float(a.mean()), across_sd=float(a.std()),
                share_category=float((w.mean() - a.mean()) / (m - a.mean())) if float(m - a.mean()) != 0 else None)


def oracle(Y, target, q, pool, rng, seed, nseed=5):
    """v2 (IDEATOR item 1): noisy-oracle latents nrm(Y + sigma * N(0,1)) with sigma set by bisection (log scale, 40 steps, seed 'seed') so
    the oracle's MATCHED k80 yield equals the decoder's; returns the oracle's swap scores averaged over nseed noise draws at that sigma.
    The oracle carries image-specific information only (no extra category emphasis), so its category share is the geometry baseline."""
    qt = torch.as_tensor(q, device=DEV)
    def lat(sig, sd):
        g = torch.Generator(device=DEV); g.manual_seed(sd); return S1.nrm(Y + sig * torch.randn(Y.shape, generator=g, dtype=Y.dtype, device=DEV))
    lo, hi = -4.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2; m = float(ident((lat(10 ** mid, seed) @ Y.T)[None][:, qt], qt).mean())
        lo, hi = (mid, hi) if m > target else (lo, mid)
    sig = 10 ** ((lo + hi) / 2); R_ = [swap_scores(lat(sig, seed + 1 + j)[None], Y, q, pool, rng) for j in range(nseed)]
    out_ = {k: float(np.mean([r[k] for r in R_])) for k in ("matched", "within", "across")}; out_["sigma"] = sig
    out_["share_category"] = (out_["within"] - out_["across"]) / (out_["matched"] - out_["across"]); return out_


def subject(s, out):
    t0 = time.time(); sub = f"sub-{s:02d}"; te = S1.categories(); keyed = {}
    for i, r in enumerate(te):
        if r.sum() > 0: keyed.setdefault(tuple(r), []).append(i)
    pool = {i: v for v in keyed.values() for i in v if len(v) >= 2}; q = sorted(pool); rng = np.random.default_rng(1000 + s)
    res = dict(participant=sub, n_queries=len(q), n_draws=ND, code_sha256=CODE_SHA256, models={})
    X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); MD = S1.models(sub, DEV)
    for m, (ft, _, Yc) in MD.items():
        Y = S1.nrm(torch.as_tensor(Yc, device=DEV))
        E80 = S1.nrm(S1.embed(ft, np.asarray(X.mean(1))))[None]
        E1 = torch.stack([S1.nrm(S1.embed(ft, np.asarray(X[:, r]))) for r in range(20)])
        res["models"][m] = dict(k80=swap_scores(E80, Y, q, pool, rng), k1=swap_scores(E1, Y, q, pool, rng))
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
    res["models"]["ubp"] = dict(k80=swap_scores(Eu, Yu, q, pool, rng)); res["models"]["ubp"]["k80_oracle"] = oracle(Yu, res["models"]["ubp"]["k80"]["matched"], q, pool, rng, 7000 + s)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, {m: {k: {a: round(b, 3) for a, b in v.items() if isinstance(b, float)} for k, v in d.items()} for m, d in res["models"].items()}, f"{res['runtime_s']}s", flush=True)


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P)
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
    res = dict(note=__doc__, n=n, code_sha256=CODE_SHA256, participants=[p["participant"] for p in P], models={})
    for m in P[0]["models"]:
        for k in P[0]["models"][m]:
            g_ = lambda a: T([p["models"][m][k][a] for p in P]); M, W, A = g_("matched"), g_("within"), g_("across")
            b = (W - A)[ix].mean(1) / (M - A)[ix].mean(1)
            res["models"][f"{m}_{k}"] = dict(matched=float(M.mean()), within=float(W.mean()), across=float(A.mean()),
                                            share_category=dict(ratio_of_means=float((W - A).mean() / (M - A).mean()), ci=torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()))
            if k == "k80_oracle":   # decoder share minus the yield-matched oracle share (paired)
                d0 = lambda a: T([p["models"][m]["k80"][a] for p in P]); Md, Wd, Ad = d0("matched"), d0("within"), d0("across")
                sd_, so_ = (Wd - Ad) / (Md - Ad), (W - A) / (M - A); dd = sd_ - so_
                res["models"][f"{m}_decoder_minus_oracle_share"] = dict(mean=float(dd.mean()), ci=torch.quantile(dd[ix].mean(1), torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist(), n_pos=int((dd > 0).sum()))
    (out / "summary_w1.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["models"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/w1_v2")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out)
    else:
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists(): subject(s, out)
