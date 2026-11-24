"""Controls through the frozen encoders (audit C: the price axes had only the analytic chance level).
For every checkpoint (ATM-S sub-08; NICE seed 141 and NICE-GA 5 seeds, 10 subjects): at k = 80 and k = 1 (5 draws),
  noise        : N(0,1) input of the same shape -> encoder -> scored against the true candidates
  shuffled_pair: real EEG -> encoder -> scored against a permuted candidate set (chance floor of the metric itself)
  eeg          : the real row for reference
Output: results/eegbits/acq_controls_<variant>_<subject>_seed<seed>.json  (rows with axis="control").
Usage: python run_acq_controls.py nice sub-01 141 | nice_ga sub-01 1024 | atm sub-08
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics
OUT = Path("/path/to/eegaudit/results/eegbits")
RAW = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/Preprocessed_data_250Hz")
CACHE = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/test_cache")


def load_test(subject):
    """Test EEG (200, 80, 63, 250) float32. The released file is a 2 GB pickled float64 dict that takes minutes to load from
    the full filesystem; the first call converts it to a memory-mappable float32 .npy, later calls open it instantly."""
    CACHE.mkdir(exist_ok=True); c = CACHE / f"{subject}_test.npy"
    if not c.exists():
        d = np.load(RAW / subject / "preprocessed_eeg_test.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
        np.save(c, np.asarray(d["preprocessed_eeg_data"], dtype=np.float32))
    return np.load(c, mmap_mode="r")


def load_encoder(variant, subject, seed, dev):
    """Returns (embed, Y, tag, X): the frozen encoder's embedding function, its candidate set and the test EEG."""
    X = load_test(subject)
    if variant == "atm":
        import run_acq_atm as A
        os.chdir(A.RETRIEVAL_DIR); sys.path.insert(0, str(A.RETRIEVAL_DIR)); sys.path.insert(0, str(A.RETRIEVAL_DIR.parent))
        import types, importlib
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
        model = ATMS(63, 250); model.load_state_dict(torch.load(A.CKPT, map_location=dev, weights_only=False), strict=True); model.to(dev).eval()
        sid = torch.full((200,), int(subject[-2:]), dtype=torch.long, device=dev)
        Y = A.load_torch(A.CLIP / "ViT-H-14_features_test.pt")
        @torch.no_grad()
        def embed(x): return model(torch.from_numpy(np.ascontiguousarray(x)).to(dev), sid).float().cpu().numpy()
        tag = "atm"
    else:
        import run_acq_nice as N
        Enc_eeg, Proj_eeg = N.load_classes(variant)
        tag = f"{'nice_ga' if variant == 'nice_ga' else 'nice'}_dgx_v2_{subject.replace('-', '')}_seed{seed}"
        enc, proj = Enc_eeg().to(dev).eval(), Proj_eeg().to(dev).eval()
        enc.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
        proj.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
        Y = np.squeeze(np.load(N.NICE / "dnn_feature/center_clip.npy", allow_pickle=True)).astype(np.float32)
        @torch.no_grad()
        def embed(x): return proj(enc(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).float().cpu().numpy()
        tag = variant
    return embed, Y, tag, X


def main(variant, subject, seed):
    t0 = time.time(); dev = torch.device("cuda")
    embed, Y, tag, X = load_encoder(variant, subject, seed, dev)
    rng = np.random.default_rng(int(seed) % 1000 + 7)
    rows = []
    for k in (80, 1):
        for draw in range(1 if k == 80 else 5):
            idx = np.arange(80) if k == 80 else rng.choice(80, k, replace=False)
            x = X[:, idx].mean(1)
            e = embed(x)
            rows.append(dict(axis="control", condition="eeg", k=k, draw=draw, **metrics(e, Y)))
            rows.append(dict(axis="control", condition="noise", k=k, draw=draw, **metrics(embed(rng.standard_normal(x.shape).astype(np.float32)), Y)))
            rows.append(dict(axis="control", condition="shuffled_pair", k=k, draw=draw, **metrics(e, Y[rng.permutation(200)])))
    out = dict(variant=tag, subject=subject, seed=int(seed), rows=rows, runtime_s=round(time.time() - t0, 1))
    (OUT / f"acq_controls_{tag}_{subject}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    s = {c: round(100 * np.mean([r["top1"] for r in rows if r["condition"] == c and r["k"] == 80]), 1) for c in ("eeg", "noise", "shuffled_pair")}
    print(f"{tag} {subject} seed{seed} top-1 at k=80 (eeg/noise/shuffled_pair): {s}  ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "0")
