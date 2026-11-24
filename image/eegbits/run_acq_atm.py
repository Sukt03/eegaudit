"""Acquisition budget through the trained encoder (ATM-S, sub-08 checkpoint, frozen): what repetitions, electrodes and
post-stimulus time buy, measured as the encoder's own 200-way retrieval on the held-out test images.

Inputs: THINGS-EEG2 preprocessed test EEG for sub-08 (200 images x 80 repetitions x 63 ch x 250 samples = 0..1.0 s),
        ATM-S retrieval checkpoint (local, 40.pth), CLIP ViT-H/14 test image embeddings.
Axes:   repetitions k in {1,2,4,8,16,32,80} (mean of a random k-subset, 5 draws);
        electrodes: occipito-parietal 8/16/32, all 63, random 8/16/32 (3 draws) -- channels outside the set are zeroed
        (the encoder was trained on all 63 channels, so this is a masking approximation, stated as such);
        window: samples after w seconds zeroed, w in {0.2, 0.4, 0.6, 1.0}.
Ceiling: split-half EEG-to-EEG identification (embedding of reps 0-39 vs 40-79), in the same retrieval units.
Yield per second divides bits per query by k x 0.2 s (RSVP stimulus-onset asynchrony of THINGS-EEG2).
Output: results/eegbits/acq_atm_<subject>.json. GPU required.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import CLIP, load_torch, metrics, wolpaw_bits

RETRIEVAL_DIR = Path("/path/to/image_benchmark/models/ATM/EEG_Image_decode/Retrieval")
CKPT = RETRIEVAL_DIR / "models/contrast/ATMS/sub-08/05-06_15-49/40.pth"
RAW = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/Preprocessed_data_250Hz")
SOA = 0.2
OCC_PAR = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
           "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP9", "TP10", "C5", "C3", "C1", "Cz"]


def main(subject="sub-08"):
    t0 = time.time()
    os.chdir(RETRIEVAL_DIR); sys.path.insert(0, str(RETRIEVAL_DIR)); sys.path.insert(0, str(RETRIEVAL_DIR.parent))   # models.subject_layers lives one level up
    import types
    # ATMS_retrieval.py imports training-time machinery at module level (CLIP, the dataset that builds OpenCLIP, braindecode,
    # wandb). Only the model classes are needed here, so those modules are stubbed before import.
    def stub(name, **attrs):
        m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]; sys.modules[name] = m; return m
    stub("clip"); stub("eegdatasets_leaveone", EEGDataset=object); stub("wandb")
    try:
        import reformer_pytorch  # noqa
    except Exception:
        stub("reformer_pytorch", LSHSelfAttention=object)   # ATM-S uses full attention; the LSH import is unused at inference
    bd = stub("braindecode"); stub("braindecode.models", EEGNetv4=object, ATCNet=object, EEGConformer=object, EEGITNet=object, ShallowFBCSPNet=object); bd.models = sys.modules["braindecode.models"]
    import importlib
    try:
        importlib.import_module("models.util")
    except Exception:
        stub("models.util", wandb_logger=object)
    from ATMS_retrieval import ATMS
    dev = torch.device("cuda")
    model = ATMS(63, 250); model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False), strict=True); model.to(dev).eval()
    sid = torch.full((200,), int(subject[-2:]), dtype=torch.long, device=dev)
    d = np.load(RAW / subject / "preprocessed_eeg_test.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
    X = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32)           # (200, 80, 63, 250)
    ch = list(d["ch_names"]); Yte = load_torch(CLIP / "ViT-H-14_features_test.pt")
    occ = [i for i, c in enumerate(ch) if c in OCC_PAR]

    @torch.no_grad()
    def embed(x):                                                           # x: (200, 63, 250) numpy
        return model(torch.from_numpy(np.ascontiguousarray(x)).to(dev), sid).float().cpu().numpy()

    # FILL=zero (default): masked channels/samples set to 0. FILL=mean_loo: masked entries replaced by the grand-mean 80-rep ERP of
    # the OTHER 199 images (in-distribution, carries no information about the queried image; audit D showed zero fill is out of
    # distribution for the frozen encoders and under-reports the window axis).
    FILL = os.environ.get("FILL", "zero")
    pool = X.mean(1); pool_loo = (pool.sum(0, keepdims=True) - pool) / (pool.shape[0] - 1)

    def q(x, chans=None, w=None):
        x = x.copy(); m = np.zeros((63, 250), bool)
        if chans is not None:
            cm = np.ones(63, bool); cm[chans] = False; m[cm, :] = True
        if w is not None:
            m[:, int(round(w * 250)):] = True
        fill = pool_loo if FILL == "mean_loo" else np.zeros_like(x)
        x[:, m] = fill[:, m]
        return x

    rows = []
    rng = np.random.default_rng(0)
    full = X.mean(1)
    native = metrics(embed(full), Yte); rows.append(dict(axis="native", k=80, channels="all63", window=1.0, draw=0, **native))
    print("native k=80 all63:", {k: round(v, 3) for k, v in native.items()}, flush=True)
    # repetitions
    DRAWS = int(os.environ.get("DRAWS", "5"))      # audit D: 5 draws give an SE of the order of the k=2->4 step; 20 is enough
    for k in [1, 2, 4, 8, 16, 32, 80]:
        for draw in range(1 if k == 80 else DRAWS):
            idx = rng.choice(80, k, replace=False)
            m = metrics(embed(X[:, idx].mean(1)), Yte)
            rows.append(dict(axis="repetitions", k=int(k), channels="all63", window=1.0, draw=draw, bits_per_second=m["bits"] / (k * SOA), **m))
    # electrodes x window at k = 80 and k = 1 (5 draws for k = 1)
    subsets = {"occpar8": occ[:8], "occpar16": occ[:16], "occpar32": occ[:32], "all63": list(range(63))}
    for n in (8, 16, 32):
        for s in range(3):
            subsets[f"random{n}_s{s}"] = sorted(rng.choice(63, n, replace=False).tolist())
    for name, chans in subsets.items():
        for w in (0.2, 0.4, 0.6, 1.0):
            m80 = metrics(embed(q(full, chans, w)), Yte)
            rows.append(dict(axis="acquisition", k=80, channels=name, n_channels=len(chans), window=w, draw=0, bits_per_second=m80["bits"] / (80 * SOA), **m80))
            for draw in range(5):
                idx = rng.choice(80, 1, replace=False)
                m1 = metrics(embed(q(X[:, idx].mean(1), chans, w)), Yte)
                rows.append(dict(axis="acquisition", k=1, channels=name, n_channels=len(chans), window=w, draw=draw, bits_per_second=m1["bits"] / (1 * SOA), **m1))
    # split-half ceiling in retrieval units (EEG-to-EEG), and the k-rep half against the 40-rep half
    A, B = embed(X[:, :40].mean(1)), embed(X[:, 40:].mean(1))
    rows.append(dict(axis="ceiling_split_half", k=40, channels="all63", window=1.0, draw=0, **metrics(A, B)))
    for k in [1, 4, 16]:
        for draw in range(5):
            idx = rng.choice(40, k, replace=False)
            rows.append(dict(axis="ceiling_k_vs_half", k=int(k), channels="all63", window=1.0, draw=draw, **metrics(embed(X[:, idx].mean(1)), B)))
    out = dict(subject=subject, checkpoint=str(CKPT), soa_s=SOA, n_test=200, note=f"channels outside a subset and samples after the window are masked (fill={FILL}) before the frozen encoder (masking approximation)", rows=rows, runtime_s=round(time.time() - t0, 1))
    Path(f"/path/to/eegaudit/results/eegbits/acq_atm_{subject}" + ("" if FILL == "zero" else f"_{FILL}") + ("" if DRAWS == 5 else f"_draws{DRAWS}") + ".json").write_text(json.dumps(out, indent=1))
    rep = {k: round(100 * np.mean([r["top1"] for r in rows if r["axis"] == "repetitions" and r["k"] == k]), 1) for k in [1, 2, 4, 8, 16, 32, 80]}
    bps = {k: round(np.mean([r["bits_per_second"] for r in rows if r["axis"] == "repetitions" and r["k"] == k]), 3) for k in [1, 2, 4, 8, 16, 32, 80]}
    print("top-1 by repetitions:", rep); print("bits/s by repetitions:", bps)
    print("k=80 top-1 by electrodes (1.0 s):", {n: round(100 * np.mean([r["top1"] for r in rows if r["axis"] == "acquisition" and r["k"] == 80 and r["window"] == 1.0 and r["channels"].startswith(n)]), 1) for n in ["occpar8", "occpar16", "occpar32", "all63", "random8", "random16", "random32"]})
    print("k=80 top-1 by window (all63):", {w: round(100 * np.mean([r["top1"] for r in rows if r["axis"] == "acquisition" and r["k"] == 80 and r["channels"] == "all63" and r["window"] == w]), 1) for w in (0.2, 0.4, 0.6, 1.0)})
    print("split-half ceiling 2-way/top-1:", [(round(100 * r["two_way"], 1), round(100 * r["top1"], 1)) for r in rows if r["axis"] == "ceiling_split_half"], f"({out['runtime_s']}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sub-08")
