"""Retraining check for the electrode and time prices (results-auditor follow-up, 2026-11-20): the price axes through frozen
encoders are read by input masking, an approximation. Here NICE-GA is retrained from scratch with the SAME training script,
seed, epochs and checkpoint rule as the BMVC models (nice_stand_ga_dgx_v2.py) but on a reduced input:
  occpar32 : the 32 occipito-parietal electrodes only (the same set as run_acq_nice.OCC_PAR), full 1.0 s
  w06      : all 63 electrodes, first 0.6 s (150 samples) after onset
  pca100   : full input but every training and test epoch replaced by its rank-100 PCA reconstruction (basis fitted on the training
             epochs only): the retrained BrainBits-style input bottleneck, the exact measurement of the EEG dimensions the decoder needs
  shuffle  : full input, but the training EEG epochs are permuted against their images (training-time trial-shuffled control for the
             encoder ladder: must give chance on the test set)
  dinov2   : full input (63 ch, 1.0 s) but the TARGET space swapped from CLIP ViT-L/14 to DINOv2 ViT-B/14 (target-swap test: if the
             yield shortfall moves with the target space, the price is set by the cross-modal read-out, not the electrodes)
and its native 200-way yield (k = 80 and k = 1, 5 draws) is compared with the masked numbers.
The training source is loaded as text and only its constants, data path, output dirs and the two data-slicing lines are
changed, so the model, loss, optimiser and checkpoint selection are the published ones.
Output: results/eegbits/retrain_<variant>_<subject>_seed<seed>.json ; checkpoints in results/eegbits/nice_ga_variants/.
Usage: python retrain_nice_ga_variant.py occpar32 sub-01 141 [--epoch 200]
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics
from run_acq_nice import OCC_PAR, NICE
from run_acq_controls import load_test
ROOT = Path("/path/to/eegaudit")
DATA = ROOT / "data_raw/things_eeg2_250hz/Preprocessed_data_250Hz"
OUTDIR = ROOT / "results/eegbits/nice_ga_variants"; OUTDIR.mkdir(parents=True, exist_ok=True)
SRC = NICE / "nice_stand_ga_dgx_v2.py"


PCA_SRC = '''
def PCA_RECON(train_data, test_data, rank):
    """Rank-`rank` PCA reconstruction of every epoch (N, 1, C, T) -> same shape; basis = top right-singular vectors of the standardised
    TRAINING epochs (GPU SVD); test epochs projected with the training basis. Nothing from the test set enters the basis."""
    import torch as _t
    ntr = train_data.shape[0]; shp = train_data.shape[1:]
    Xtr = _t.from_numpy(train_data.reshape(ntr, -1).astype(np.float32)).cuda(); Xte = _t.from_numpy(test_data.reshape(test_data.shape[0], -1).astype(np.float32)).cuda()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    _, _, Vh = _t.linalg.svd(Ztr, full_matrices=False); V = Vh[:rank].T
    Rtr = ((Ztr @ V) @ V.T) * sd + mu; Rte = ((Zte @ V) @ V.T) * sd + mu
    out_tr = Rtr.cpu().numpy().reshape((ntr,) + shp).astype(np.float32); out_te = Rte.cpu().numpy().reshape((test_data.shape[0],) + shp).astype(np.float32)
    del Xtr, Xte, Ztr, Zte, Vh, V, Rtr, Rte; _t.cuda.empty_cache()
    print("PCA_RECON: rank", rank, "train", out_tr.shape, "test", out_te.shape, flush=True)
    return out_tr, out_te
'''


def build_source(variant, ch_idx):
    s = SRC.read_text()
    n_ch = len(ch_idx) if variant in ("occpar32", "occpar16", "occ32w06") else 63
    assert variant in ("occpar32", "occpar16", "w06", "w04", "occ32w06", "dinov2", "pixpca", "randtarget", "pca100", "shuffle"), variant
    TARGET = variant if variant in ("dinov2", "pixpca", "randtarget") else None
    n_t = {"w06": 150, "occ32w06": 150, "w04": 100}.get(variant, 250)
    enc = 40 * ((n_t - 24 - 51) // 5 + 1)                     # Conv(1,25) -> AvgPool((1,51),(1,5)) -> 40 maps x time positions
    DINO = f"/path/to/eegaudit/results/eegbits/{variant}_feature/"      # target-swap feature dir (dinov2 / pixpca)
    reps = [("EEG_CHANNELS = 63", f"EEG_CHANNELS = {n_ch}"), ("EEG_TIME_POINTS = 250", f"EEG_TIME_POINTS = {n_t}"),
            ("ENC_EEG_FEATURES = 1440", f"ENC_EEG_FEATURES = {enc}"),
            ("nn.Conv2d(40, 40, (63, 1), (1, 1))", "nn.Conv2d(40, 40, (EEG_CHANNELS, 1), (1, 1))"),
            ("def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):", "def __init__(self, embedding_dim=ENC_EEG_FEATURES, proj_dim=768, drop_proj=0.5):"),
            ("model_dir = PROJECT_ROOT + '/models/nice/external/NICE-EEG/model/'", f"model_dir = '{OUTDIR}/'"),
            ("result_path = PROJECT_ROOT + '/results/nice/'", f"result_path = '{OUTDIR}/'"),
            ("model_idx = 'nice_ga_dgx_v2'", f"model_idx = 'nice_ga_{variant}'"),
            ("self.eeg_data_path = '/path/to/image_benchmark/datasets/things-eeg2/hf_data/Preprocessed_data_250Hz/'", f"self.eeg_data_path = '{DATA}/'"),
            ("self.img_data_path = '/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature/'", f"self.img_data_path = '{DINO if variant in ('dinov2', 'pixpca', 'randtarget') else str(NICE / 'dnn_feature') + '/'}'"),
            ("self.test_center_path = '/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature/'", f"self.test_center_path = '{DINO if variant in ('dinov2', 'pixpca', 'randtarget') else str(NICE / 'dnn_feature') + '/'}'"),
            ("        train_data = np.expand_dims(train_data, axis=1)\n", f"        train_data = np.expand_dims(train_data, axis=1)\n        train_data = np.ascontiguousarray(train_data[:, :, CH_IDX, :EEG_TIME_POINTS])\n" + ("        train_data = train_data[np.random.default_rng(12345).permutation(len(train_data))]\n" if variant == "shuffle" else "")),
            ("        test_data = np.expand_dims(test_data, axis=1)\n", f"        test_data = np.expand_dims(test_data, axis=1)\n        test_data = np.ascontiguousarray(test_data[:, :, CH_IDX, :EEG_TIME_POINTS])\n" + ("        train_data, test_data = PCA_RECON(train_data, test_data, PCA_RANK)\n" if variant == "pca100" else ""))]
    for a, b in reps:
        assert s.count(a) == 1, (a, s.count(a)); s = s.replace(a, b)
    s = s.replace("EEG_CHANNELS = ", f"CH_IDX = {list(map(int, ch_idx))}\nEEG_CHANNELS = ", 1)
    if variant == "pca100":
        s = s.replace("CH_IDX = ", PCA_SRC + "\nPCA_RANK = 100\nCH_IDX = ", 1)
    return s, n_ch, n_t, enc


def main():
    variant, subject, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); epochs = int(sys.argv[sys.argv.index("--epoch") + 1]) if "--epoch" in sys.argv else 200
    t0 = time.time(); sub = int(subject[-2:])
    d = np.load(DATA / subject / "preprocessed_eeg_test.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
    ch = list(d["ch_names"]); occ = [i for i, c in enumerate(ch) if c in OCC_PAR]; assert len(occ) == 32, len(occ)
    ch_idx = occ if variant in ("occpar32", "occ32w06") else ([ch.index(c) for c in OCC_PAR[:16]] if variant == "occpar16" else list(range(63)))   # occpar16 = the first 16 names of OCC_PAR (PO7..O2, P7..P8), in OCC_PAR order (code review 2026-11-20: occ[:16] was channel-file order = central electrodes)
    src, n_ch, n_t, enc = build_source(variant, ch_idx)
    print(f"variant {variant}: channels {n_ch}, time points {n_t}, encoder features {enc}", flush=True)
    g = {"__name__": "__main__", "__file__": str(SRC)}
    old_argv, old_cwd = sys.argv, os.getcwd(); os.chdir(NICE)
    sys.argv = [str(SRC), "--subject", str(sub), "--fixed_seed", str(seed), "--epoch", str(epochs)] + (["--dnn", variant] if variant in ("dinov2", "pixpca", "randtarget") else [])
    try:
        exec(compile(src, str(SRC), "exec"), g)
    finally:
        sys.argv, _ = old_argv, os.chdir(old_cwd)
    # ---- evaluate the saved best checkpoint in our units (same scoring as run_acq_nice: 200 CLIP centres, cosine)
    dev = torch.device("cuda"); Enc_eeg, Proj_eeg = g["Enc_eeg"], g["Proj_eeg"]
    tag = f"nice_ga_{variant}_sub{sub:02d}_seed{seed}"
    enc_m, proj = Enc_eeg().to(dev).eval(), Proj_eeg().to(dev).eval()
    enc_m.load_state_dict(torch.load(OUTDIR / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
    proj.load_state_dict(torch.load(OUTDIR / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
    centers = np.squeeze(np.load((Path(f"/path/to/eegaudit/results/eegbits/{variant}_feature/center_{variant}.npy")) if variant in ("dinov2", "pixpca", "randtarget") else NICE / "dnn_feature/center_clip.npy", allow_pickle=True)).astype(np.float32)
    X = np.asarray(load_test(subject))[:, :, ch_idx, :n_t]
    if variant == "pca100":                                   # evaluate on rank-100 reconstructions with the training basis (same as training)
        tr = np.load(DATA / subject / "preprocessed_eeg_training.npy", allow_pickle=True); tr = tr.item() if isinstance(tr, np.ndarray) else tr
        Xtr_full = np.asarray(tr["preprocessed_eeg_data"], dtype=np.float32).mean(1)[:, None]
        Xq = X.reshape(-1, 1, X.shape[2], X.shape[3])
        _, Xq = g["PCA_RECON"](Xtr_full, Xq, 100); X = Xq.reshape(X.shape); del Xtr_full

    @torch.no_grad()
    def embed(x):
        return proj(enc_m(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).float().cpu().numpy()
    rng = np.random.default_rng(seed % 1000); rows = []
    rows.append(dict(axis="native", k=80, draw=0, **metrics(embed(X.mean(1)), centers)))
    for draw in range(5):
        rows.append(dict(axis="repetitions", k=1, draw=draw, **metrics(embed(X[:, rng.choice(80, 1)].mean(1)), centers)))
    rows.append(dict(axis="control", condition="noise", k=80, draw=0, **metrics(embed(rng.standard_normal(X.shape[0:1] + X.shape[2:]).astype(np.float32)), centers)))
    out = dict(variant=variant, subject=subject, seed=seed, epochs=epochs, n_channels=n_ch, n_timepoints=n_t, channel_indices=ch_idx,
               note="NICE-GA retrained from scratch on the reduced input with the published script, seed, epochs and checkpoint rule; scored like run_acq_nice.py",
               rows=rows, runtime_s=round(time.time() - t0, 1))
    (ROOT / "results/eegbits" / f"retrain_{variant}_{subject}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    r = rows[0]; print(f"RETRAIN {variant} {subject} seed{seed} epochs={epochs}: native top-1 {100*r['top1']:.1f}% 2-way {100*r['two_way']:.1f}% bits {r['bits']:.2f} | k=1 bits {np.mean([x['bits'] for x in rows if x['axis']=='repetitions']):.3f} | noise top-1 {100*rows[-1]['top1']:.1f}%  ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
