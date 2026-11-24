"""Shared helpers for the IMAGE smokes (2026-11-20 loop). Read-only on existing project files.
Loads a RETRAINED NICE-GA variant checkpoint from results/eegbits/nice_ga_variants/ (reduced channels / window / swapped target)
by re-using retrain_nice_ga_variant.build_source for the constants and run_acq_nice-style class extraction."""
import json, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import metrics  # noqa  (returns soft_bits)
from run_acq_controls import load_test, load_encoder  # noqa
from run_acq_nice import NICE, OCC_PAR  # noqa
import retrain_nice_ga_variant as RV

ROOT = Path("/path/to/eegaudit")
EB = ROOT / "results/eegbits"
VARDIR = EB / "nice_ga_variants"
OUT = ROOT / "results/loop/image"; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["PatchEmbedding", "ResidualAdd", "FlattenHead", "EEG_GAT", "Enc_eeg", "Proj_eeg", "Proj_img"]


def variant_classes(variant, ch_idx):
    from torch_geometric.nn import GATConv
    src, n_ch, n_t, enc = RV.build_source(variant, ch_idx)
    ns = dict(nn=nn, torch=torch, F=F, Tensor=Tensor, Rearrange=Rearrange, np=np, GATConv=GATConv,
              EEG_CHANNELS=n_ch, EEG_TIME_POINTS=n_t, ENC_EEG_FEATURES=enc, CLIP_FEATURES=768)
    for cls in CLASSES:
        m = re.search(rf"^class {cls}\b.*?(?=^class |^def |\Z)", src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns["Enc_eeg"], ns["Proj_eeg"], n_ch, n_t


def centers_for(variant):
    if variant in ("dinov2", "pixpca", "randtarget"):
        p = EB / f"{variant}_feature" / f"center_{variant}.npy"
    else:
        p = NICE / "dnn_feature/center_clip.npy"
    return np.squeeze(np.load(p, allow_pickle=True)).astype(np.float32)


def load_variant(variant, subject, seed, dev):
    """Returns (embed, Y, X_sliced, meta) for a retrained variant checkpoint; X already sliced to its channels / window."""
    meta = json.loads((EB / f"retrain_{variant}_{subject}_seed{seed}.json").read_text())
    ch_idx = list(meta["channel_indices"]); n_t = int(meta["n_timepoints"])
    Enc_eeg, Proj_eeg, n_ch, n_t2 = variant_classes(variant, ch_idx); assert n_t == n_t2, (n_t, n_t2)
    tag = f"nice_ga_{variant}_sub{int(subject[-2:]):02d}_seed{seed}"
    enc, proj = Enc_eeg().to(dev).eval(), Proj_eeg().to(dev).eval()
    enc.load_state_dict(torch.load(VARDIR / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
    proj.load_state_dict(torch.load(VARDIR / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
    X = np.asarray(load_test(subject))[:, :, ch_idx, :n_t]

    @torch.no_grad()
    def embed(x):
        return proj(enc(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).float().cpu().numpy()
    return embed, centers_for(variant), X, meta


def save(name, obj):
    p = OUT / name; p.write_text(json.dumps(obj, indent=1)); print("->", p, flush=True); return p
