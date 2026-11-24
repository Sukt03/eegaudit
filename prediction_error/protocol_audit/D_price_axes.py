"""Protocol audit D: are the price axes (repetitions k, window, electrodes) measured correctly and in the right units?

Part A (JSON only): yield-per-second conventions, draw variance, log-linear fit in k, price-list table for every decoder with an
acq_*.json (ATM-S sub-08, NICE 10 subjects, NICE-GA 6 subjects x 5 seeds).
Part B (GPU, frozen ATM-S sub-08 checkpoint, functions/constants imported from run_acq_atm.py and run.py, nothing edited):
  - first-k vs random-k repetition draws;
  - masking artefact: zero fill vs in-distribution fills (other-image EEG, grand-mean EEG) for windows and channel subsets;
  - order test: embed(mean of masked trials) vs mean of embed(masked trial) at k=80.
Writes results/eegbits/audit_D_price_axes.json (new file) and the report in outputs/.
"""
import json, sys, time, types, importlib, os
from pathlib import Path
import numpy as np

ROOT = Path("/path/to/eegaudit")
RES = ROOT / "results/eegbits"
sys.path.insert(0, str(ROOT / "experiments/eegbits"))
from run import metrics, wolpaw_bits                                  # noqa: E402
import run_acq_atm as A                                                # constants + the module docstring; main() is not called

IMG_T, SOA, WIN = 0.1, 0.2, 1.0
KS = [1, 2, 4, 8, 16, 32, 80]


def load(p):
    return json.loads(Path(p).read_text())


def rep_curve(d):
    """mean/sd of bits and top1 over draws per k from the 'repetitions' rows."""
    out = {}
    for k in KS:
        rows = [r for r in d["rows"] if r["axis"] == "repetitions" and r["k"] == k]
        b = np.array([r["bits"] for r in rows]); t = np.array([r["top1"] for r in rows])
        out[k] = dict(n=len(rows), bits=b.mean(), bits_sd=b.std(ddof=1) if len(b) > 1 else 0.0, top1=t.mean(), top1_sd=t.std(ddof=1) if len(t) > 1 else 0.0, draws=b.tolist())
    return out


def acq(d, k, channels, w):
    rows = [r for r in d["rows"] if r["axis"] == "acquisition" and r["k"] == k and r["channels"] == channels and abs(r["window"] - w) < 1e-6]
    return float(np.mean([r["bits"] for r in rows])) if rows else np.nan, float(np.mean([r["top1"] for r in rows])) if rows else np.nan


def conventions(bits_by_k):
    """bits/s under four denominators. Returns {name: {k: bps}} and argmax k per convention."""
    den = {"image_0.1s": lambda k: k * IMG_T, "soa_0.2s": lambda k: k * SOA, "soa_plus_window": lambda k: k * SOA + (WIN - SOA), "window_1.0s": lambda k: k * WIN}
    out = {}
    for name, f in den.items():
        bps = {k: bits_by_k[k] / f(k) for k in bits_by_k}
        out[name] = dict(bps=bps, kopt=int(max(bps, key=bps.get)), peak=max(bps.values()))
    return out


def loglin(curve):
    ks = np.array(KS, float); y = np.array([curve[k]["bits"] for k in KS])
    Xd = np.c_[np.ones_like(ks), np.log2(ks)]
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef; resid = y - pred; r2 = 1 - resid.var() / y.var()
    # saturating alternative: bits = c - a * k^-b  is 3-parameter; use the simpler  bits vs log2 k on k<=32 only to test curvature
    coef5, *_ = np.linalg.lstsq(Xd[:-1], y[:-1], rcond=None); r2_5 = 1 - (y[:-1] - Xd[:-1] @ coef5).var() / y[:-1].var()
    return dict(a=coef[0], b=coef[1], r2=r2, resid={k: float(r) for k, r in zip(KS, resid)}, r2_excl80=r2_5, b_excl80=coef5[1], extrap80_from_le32=float(np.r_[1, np.log2(80)] @ coef5), obs80=float(y[-1]))


def ci95(v):
    v = np.asarray(v, float); n = len(v)
    return v.mean(), (1.96 * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)


# ---------------------------------------------------------------- Part A: JSONs
def part_a():
    dec = {}
    dec["ATM-S (sub-08, frozen public ckpt)"] = [load(RES / "acq_atm_sub-08.json")]
    dec["NICE (10 subj, seed 141)"] = [load(p) for p in sorted(RES.glob("acq_nice_sub-*_seed141.json"))]
    dec["NICE-GA (6 subj x 5 seeds)"] = [load(p) for p in sorted(RES.glob("acq_nice_ga_sub-*_seed*.json"))]
    out = {}
    for name, ds in dec.items():
        curves = [rep_curve(d) for d in ds]
        # per-file quantities, then mean +- 95% CI across files (subjects/seeds)
        per = []
        for d, c in zip(ds, curves):
            bk = {k: c[k]["bits"] for k in KS}; conv = conventions(bk); ll = loglin(c)
            b04, t04 = acq(d, 80, "all63", 0.4); b32, t32 = acq(d, 80, "occpar32", 1.0); b32r = np.nanmean([acq(d, 80, f"random32_s{s}", 1.0)[0] for s in range(3)])
            b1_32, _ = acq(d, 1, "occpar32", 1.0); b1_04, _ = acq(d, 1, "all63", 0.4)
            b8, _ = acq(d, 80, "occpar8", 1.0); b16, _ = acq(d, 80, "occpar16", 1.0)
            ceil = [r["bits"] for r in d["rows"] if r["axis"] == "ceiling_split_half"]
            per.append(dict(subject=d["subject"], seed=d.get("seed"), bits1=bk[1], bits80=bk[80], top1_1=c[1]["top1"], top1_80=c[80]["top1"], cv1=c[1]["bits_sd"] / bk[1], sd_by_k={k: c[k]["bits_sd"] for k in KS},
                            conv={n: dict(kopt=v["kopt"], peak=v["peak"], bps=v["bps"]) for n, v in conv.items()}, loglin=ll, bits_w04_k80=b04, top1_w04_k80=t04, bits_occ32_k80=b32, top1_occ32_k80=t32, bits_rand32_k80=b32r,
                            bits_occ32_k1=b1_32, bits_w04_k1=b1_04, bits_occ8_k80=b8, bits_occ16_k80=b16, ceiling_bits=ceil[0] if ceil else np.nan))
        agg = {}
        for key in ["bits1", "bits80", "top1_1", "top1_80", "cv1", "bits_w04_k80", "top1_w04_k80", "bits_occ32_k80", "top1_occ32_k80", "bits_rand32_k80", "bits_occ32_k1", "bits_w04_k1", "bits_occ8_k80", "bits_occ16_k80", "ceiling_bits"]:
            agg[key] = ci95([p[key] for p in per])
        agg["bits_by_k"] = {k: ci95([p["conv"]["soa_0.2s"]["bps"][k] * k * SOA for p in per]) for k in KS}
        agg["sd_by_k"] = {k: float(np.mean([p["sd_by_k"][k] for p in per])) for k in KS}
        agg["conv"] = {}
        for n in ["image_0.1s", "soa_0.2s", "soa_plus_window", "window_1.0s"]:
            bps_k = {k: ci95([p["conv"][n]["bps"][k] for p in per]) for k in KS}
            mean_curve = {k: bps_k[k][0] for k in KS}; kopt_mean = int(max(mean_curve, key=mean_curve.get))
            agg["conv"][n] = dict(bps_k=bps_k, kopt_of_mean_curve=kopt_mean, kopt_per_file=[p["conv"][n]["kopt"] for p in per], peak=bps_k[kopt_mean])
        agg["loglin"] = dict(r2=ci95([p["loglin"]["r2"] for p in per]), b=ci95([p["loglin"]["b"] for p in per]), a=ci95([p["loglin"]["a"] for p in per]), r2_excl80=ci95([p["loglin"]["r2_excl80"] for p in per]),
                             resid={k: float(np.mean([p["loglin"]["resid"][k] for p in per])) for k in KS}, extrap80=ci95([p["loglin"]["extrap80_from_le32"] for p in per]), obs80=ci95([p["loglin"]["obs80"] for p in per]))
        out[name] = dict(n_files=len(ds), per_file=per, agg=agg)
    # Cui et al. 2026 Table 1 Base (HCF retriever, THINGS-EEG2, 200-way) as a check of the "ITR peaks at k=2" claim under each convention
    cui = {1: .140, 4: .432, 8: .604, 16: .725, 32: .788, 64: .821, 80: .834}
    cui_n2n = {1: .184, 2: .313, 4: .475, 8: .621, 16: .717, 32: .772, 64: .799, 80: .805}   # Table 2, N2N query (+Q), has k=2
    cui_out = {}
    den = {"image_0.1s": lambda k: k * IMG_T, "soa_0.2s": lambda k: k * SOA, "soa_plus_window": lambda k: k * SOA + (WIN - SOA), "window_1.0s": lambda k: k * WIN}
    for lab, acc in [("cui_base_tab1", cui), ("cui_plusQ_tab2", cui_n2n)]:
        bk = {k: wolpaw_bits(p, 200) for k, p in acc.items()}
        cui_out[lab] = {}
        for n, f in den.items():
            bps = {k: bk[k] / f(k) for k in bk}
            cui_out[lab][n] = dict(kopt=int(max(bps, key=bps.get)), bits_per_min={k: round(60 * v, 1) for k, v in bps.items()})
    out["cui2026_check"] = cui_out
    return out


# ---------------------------------------------------------------- Part B: encoder runs (GPU)
def load_atm():
    import torch
    os.chdir(A.RETRIEVAL_DIR); sys.path.insert(0, str(A.RETRIEVAL_DIR)); sys.path.insert(0, str(A.RETRIEVAL_DIR.parent))
    def stub(name, **attrs):
        m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]; sys.modules[name] = m; return m
    stub("clip"); stub("eegdatasets_leaveone", EEGDataset=object); stub("wandb")
    try:
        import reformer_pytorch  # noqa
    except Exception:
        stub("reformer_pytorch", LSHSelfAttention=object)
    bd = stub("braindecode"); stub("braindecode.models", EEGNetv4=object, ATCNet=object, EEGConformer=object, EEGITNet=object, ShallowFBCSPNet=object); bd.models = sys.modules["braindecode.models"]
    try:
        importlib.import_module("models.util")
    except Exception:
        stub("models.util", wandb_logger=object)
    from ATMS_retrieval import ATMS
    dev = torch.device("cuda")
    model = ATMS(63, 250); model.load_state_dict(torch.load(A.CKPT, map_location=dev, weights_only=False), strict=True); model.to(dev).eval()
    return model, dev


def part_b(subject="sub-08", smoke=False):
    import torch
    from run import CLIP, load_torch
    model, dev = load_atm()
    sid = torch.full((200,), int(subject[-2:]), dtype=torch.long, device=dev)
    d = np.load(A.RAW / subject / "preprocessed_eeg_test.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
    X = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32); ch = list(d["ch_names"]); Yte = load_torch(CLIP / "ViT-H-14_features_test.pt")
    occ = [i for i, c in enumerate(ch) if c in A.OCC_PAR]
    times = d.get("times"); tinfo = (float(np.min(times)), float(np.max(times)), len(times)) if times is not None else None

    @torch.no_grad()
    def embed(x):
        return model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev), sid).float().cpu().numpy()

    def mask_fill(x, chans=None, w=None, fill="zero", rng=None, pool=None):
        """x (200,63,250). fill: zero | other (same positions from a different image's 80-rep average) | mean (grand mean over images)."""
        x = x.copy(); m = np.zeros((63, 250), bool)
        if chans is not None:
            cm = np.ones(63, bool); cm[chans] = False; m[cm, :] = True
        if w is not None:
            m[:, int(round(w * 250)):] = True
        if fill == "zero":
            f = np.zeros_like(x)
        elif fill == "mean":
            f = np.broadcast_to(pool.mean(0, keepdims=True), x.shape)
        elif fill == "mean_loo":                                      # grand mean over the OTHER 199 images (no self-leak)
            f = (pool.sum(0, keepdims=True) - pool) / (pool.shape[0] - 1)
        elif fill == "other":
            perm = rng.permutation(200); perm = np.where(perm == np.arange(200), (perm + 1) % 200, perm)   # derangement-ish: no image gets itself
            f = pool[perm]
        x[:, m] = f[:, m]
        return x

    R = dict(subject=subject, times=tinfo, n_draws=2 if smoke else 5)
    ND = R["n_draws"]; rng = np.random.default_rng(1)
    full = X.mean(1)
    ks = [1, 4] if smoke else [1, 2, 4, 8, 16, 32]
    # ---- B1 first-k vs random-k vs contiguous-k (random start)
    rows = []
    for k in ks:
        rows.append(dict(scheme="first", k=k, draw=0, **metrics(embed(X[:, :k].mean(1)), Yte)))
        rows.append(dict(scheme="last", k=k, draw=0, **metrics(embed(X[:, 80 - k:].mean(1)), Yte)))
        for dr in range(ND):
            idx = rng.choice(80, k, replace=False); rows.append(dict(scheme="random", k=k, draw=dr, **metrics(embed(X[:, idx].mean(1)), Yte)))
            s = rng.integers(0, 81 - k); rows.append(dict(scheme="contiguous", k=k, draw=dr, **metrics(embed(X[:, s:s + k].mean(1)), Yte)))
            if k <= 20:   # within a 20-rep block (one session if blocks are sessions)
                b = rng.integers(0, 4); idx = b * 20 + rng.choice(20, k, replace=False); rows.append(dict(scheme="within_block20", k=k, draw=dr, **metrics(embed(X[:, idx].mean(1)), Yte)))
    R["B1_schemes"] = rows
    # ---- B2 masking: zero vs other-image vs mean fill, k=80 and k=1, windows and channel subsets
    rows = []
    pool80 = full
    wins = [0.4, 1.0] if smoke else [0.2, 0.4, 0.6, 1.0]
    subsets = {"all63": list(range(63)), "occpar32": occ[:32], "occpar16": occ[:16], "occpar8": occ[:8]}
    rs = np.random.default_rng(0)
    for s in range(3):
        subsets[f"random32_s{s}"] = sorted(rs.choice(63, 32, replace=False).tolist())
    if smoke:
        subsets = {k: v for k, v in subsets.items() if k in ("all63", "occpar32", "occpar8")}
    singles = [X[:, rng.integers(80)] for _ in range(ND)]
    for name, chans in subsets.items():
        for w in wins:
            if name == "all63" and w == 1.0:
                fills = ["zero"]
            else:
                fills = ["zero", "other", "mean", "mean_loo"]
            for fill in fills:
                reps = 1 if fill != "other" else ND
                for dr in range(reps):
                    m80 = metrics(embed(mask_fill(full, chans, w, fill, rng, pool80)), Yte)
                    rows.append(dict(k=80, channels=name, n_channels=len(chans), window=w, fill=fill, draw=dr, **m80))
                for dr, xs in enumerate(singles):
                    # k=1 query; the fill pool is the 80-rep average of OTHER images (in-distribution content, zero information about this image)
                    m1 = metrics(embed(mask_fill(xs, chans, w, fill, rng, pool80)), Yte)
                    rows.append(dict(k=1, channels=name, n_channels=len(chans), window=w, fill=fill, draw=dr, **m1))
    R["B2_fills"] = rows
    # ---- B3 order test: embed(mean of masked trials) vs mean(embed(masked trial)) at k=80 (and k=8), zero fill; also unmasked
    rows = []
    E_single = None
    for name, chans in [("all63", None), ("occpar32", occ[:32]), ("occpar8", occ[:8])]:
        for w in ([None, 0.4] if smoke else [None, 0.2, 0.4, 0.6]):
            if name == "all63" and w is None:
                pass
            xm = mask_fill(X.reshape(-1, 63, 250), chans, w, "zero").reshape(200, 80, 63, 250) if (chans is not None or w is not None) else X
            e_in = embed(xm.mean(1))                                     # average in input space, then encode (what the JSONs did)
            E = np.stack([embed(xm[:, r]) for r in range(80)])           # encode each trial, then average embeddings
            e_out = E.mean(0)
            cos = float(np.mean(np.sum(e_in * e_out, 1) / np.linalg.norm(e_in, axis=1) / np.linalg.norm(e_out, axis=1)))
            mi, mo = metrics(e_in, Yte), metrics(e_out, Yte)
            # k=8 version from the same single-trial embeddings (random 8)
            idx = rng.choice(80, 8, replace=False)
            mi8, mo8 = metrics(embed(xm[:, idx].mean(1)), Yte), metrics(E[idx].mean(0), Yte)
            rows.append(dict(channels=name, window=w if w else 1.0, cos_in_vs_out=cos, k80_input_avg=mi, k80_embed_avg=mo, k8_input_avg=mi8, k8_embed_avg=mo8))
            print("B3", name, w, round(cos, 3), round(mi["bits"], 3), round(mo["bits"], 3), flush=True)
    R["B3_order"] = rows
    # ---- B2b per-sample image discriminability (between-image variance of the 80-rep mean / within-image variance / 80), averaged over channels
    mu = X.mean(1); Fr = (mu.var(0) / (X.var(1).mean(0) / 80)).mean(0)
    R["B2b_discriminability_F_per_100ms"] = Fr.reshape(-1, 25).mean(1).tolist(); R["B2b_F_argmax_ms"] = int(Fr.argmax()) * 4
    # ---- B4 what does the encoder emit for an all-zero input and for pure noise of matched variance (masking sanity)
    z = metrics(embed(np.zeros_like(full)), Yte); nz = metrics(embed(rng.standard_normal(full.shape).astype(np.float32) * full.std()), Yte)
    R["B4_sanity"] = dict(all_zero_input=z, noise_input=nz, data_std=float(full.std()), data_std_single=float(X[:, 0].std()))
    return R


def main():
    smoke = "--smoke" in sys.argv; t0 = time.time()
    out = dict(part_a=part_a())
    print("part A done", round(time.time() - t0, 1), "s", flush=True)
    if "--no-gpu" not in sys.argv:
        out["part_b"] = part_b(smoke=smoke)
    out["runtime_s"] = round(time.time() - t0, 1)
    tag = "_smoke" if smoke else ""
    p = RES / f"audit_D_price_axes{tag}.json"
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    p.write_text(json.dumps(conv(out), indent=1)); print("wrote", p, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
