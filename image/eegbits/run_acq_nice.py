"""Acquisition budget through the trained NICE / NICE-GA encoders (your BMVC checkpoints, 10 subjects, NICE-GA 5 seeds):
repetitions, windows and (masked) electrode subsets, scored exactly as the training scripts score the test set:
EEG embedding = Proj_eeg(Enc_eeg(x)) with x of shape (B, 1, 63, 250); candidates = the 200 CLIP centres (dnn_feature/center_clip.npy),
both L2-normalised, cosine, 200-way. Also EEG-to-EEG split-half identification as the ceiling.
Model classes are taken from the training scripts' source (they carry module-level argparse/training code, so the class blocks are
exec'd in a clean namespace). Output: results/eegbits/acq_nice_<variant>_<subject>_seed<seed>.json
Usage: python run_acq_nice.py nice sub-01 141      |  python run_acq_nice.py nice_ga sub-01 1024
"""
import json, os, re, sys, time, types
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics

NICE = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG")
RAW = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/Preprocessed_data_250Hz")
SOA = 0.2
OCC_PAR = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
           "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP9", "TP10", "C5", "C3", "C1", "Cz"]


def load_classes(variant):
    src = (NICE / ("nice_stand_ga_dgx_v2.py" if variant == "nice_ga" else "nice_stand_dgx_v2.py")).read_text()
    ns = dict(nn=nn, torch=torch, F=F, Tensor=Tensor, Rearrange=Rearrange, np=np, EEG_CHANNELS=63, EEG_TIME_POINTS=250, ENC_EEG_FEATURES=1440, CLIP_FEATURES=768)
    if variant == "nice_ga":
        from torch_geometric.nn import GATConv
        ns["GATConv"] = GATConv
    for cls in ["PatchEmbedding", "ResidualAdd", "FlattenHead", "EEG_GAT", "Enc_eeg", "Proj_eeg", "Proj_img"]:
        m = re.search(rf"^class {cls}\b.*?(?=^class |^def |\Z)", src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns["Enc_eeg"], ns["Proj_eeg"]


def main(variant, subject, seed):
    t0 = time.time(); dev = torch.device("cuda")
    Enc_eeg, Proj_eeg = load_classes(variant)
    tag = f"{'nice_ga' if variant == 'nice_ga' else 'nice'}_dgx_v2_{subject.replace('-', '')}_seed{seed}"
    enc, proj = Enc_eeg().to(dev).eval(), Proj_eeg().to(dev).eval()
    enc.load_state_dict(torch.load(NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
    proj.load_state_dict(torch.load(NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
    centers = np.squeeze(np.load(NICE / "dnn_feature/center_clip.npy", allow_pickle=True)).astype(np.float32)   # (200, 768)
    d = np.load(RAW / subject / "preprocessed_eeg_test.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
    X = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32); ch = list(d["ch_names"])
    occ = [i for i, c in enumerate(ch) if c in OCC_PAR]

    @torch.no_grad()
    def embed(x):
        xt = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev)
        return proj(enc(xt)).float().cpu().numpy()

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

    rows, rng = [], np.random.default_rng(int(seed) % 1000)
    full = X.mean(1)
    native = metrics(embed(full), centers); rows.append(dict(axis="native", k=80, channels="all63", window=1.0, draw=0, **native))
    for k in [1, 2, 4, 8, 16, 32, 80]:
        for draw in range(1 if k == 80 else 5):
            idx = rng.choice(80, k, replace=False); m = metrics(embed(X[:, idx].mean(1)), centers)
            rows.append(dict(axis="repetitions", k=int(k), channels="all63", window=1.0, draw=draw, bits_per_second=m["bits"] / (k * SOA), **m))
    subsets = {"occpar32": occ[:32], "all63": list(range(63))}
    for s in range(3):
        subsets[f"random32_s{s}"] = sorted(rng.choice(63, 32, replace=False).tolist())
    for name, chans in subsets.items():
        for w in (0.2, 0.4, 0.6, 1.0):
            m80 = metrics(embed(q(full, chans, w)), centers)
            rows.append(dict(axis="acquisition", k=80, channels=name, n_channels=len(chans), window=w, draw=0, bits_per_second=m80["bits"] / (80 * SOA), **m80))
            for draw in range(3):
                idx = rng.choice(80, 1, replace=False); m1 = metrics(embed(q(X[:, idx].mean(1), chans, w)), centers)
                rows.append(dict(axis="acquisition", k=1, channels=name, n_channels=len(chans), window=w, draw=draw, bits_per_second=m1["bits"] / SOA, **m1))
    A, B = embed(X[:, :40].mean(1)), embed(X[:, 40:].mean(1))
    rows.append(dict(axis="ceiling_split_half", k=40, channels="all63", window=1.0, draw=0, **metrics(A, B)))
    out = dict(variant=variant, subject=subject, seed=int(seed), soa_s=SOA, note=f"masking fill={FILL} (approximation; encoder trained on 63 channels)", rows=rows, runtime_s=round(time.time() - t0, 1))
    Path(f"/path/to/eegaudit/results/eegbits/acq_{variant}_{subject}_seed{seed}" + ("" if FILL == "zero" else f"_{FILL}") + ".json").write_text(json.dumps(out, indent=1))
    rep = {k: round(100 * np.mean([r["top1"] for r in rows if r["axis"] == "repetitions" and r["k"] == k]), 1) for k in [1, 4, 16, 80]}
    print(f"{variant} {subject} seed{seed}: native top1 {100*native['top1']:.1f}% bits {native['bits']:.2f} | top1 by k {rep} | window(all63,k80) " + str({w: round(100*np.mean([r['top1'] for r in rows if r['axis']=='acquisition' and r['k']==80 and r['channels']=='all63' and r['window']==w]),1) for w in (0.2,0.4,0.6,1.0)}) + f" ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
