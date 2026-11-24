"""Frozen embeddings from more EEG foundation models (braindecode zoo): LaBraM (ICLR 2024) and EEGPT (NeurIPS 2024).

Input: the 200 Hz feedback epochs (-1..1 s, uV/100) used for CBraMod; EEGPT receives them resampled to 256 Hz.
Embedding: forward_features tokens, mean and s.d. over tokens of the post-feedback second (all tokens if the model
does not expose time), saved in the decoder-ladder format (results/features/<model>/sub-XXX.npz).
Usage: python fm_zoo.py --model labram
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample

from belief_bottleneck import E

CFG = {"labram": ("Labram", "external/labram", 200), "eegpt": ("EEGPT", "external/eegpt", 256)}


def load(name, chans, n_times, dev):
    """Build at our window length from the checkpoint's config, then load weights; time-position tables are cut to the
    number of patches in our 2-s epoch, channels are mapped by name."""
    import json
    import braindecode.models as bm
    from safetensors.torch import load_file
    cls, path, sf = CFG[name]
    cfg = json.load(open(f"{path}/config.json"))
    kw = {k: v for k, v in cfg.items() if k in ("chs_info", "n_chans", "n_outputs") and v is not None}
    kw.update(n_times=n_times, sfreq=sf)
    if name == "eegpt":                                      # pretrained encoder only, our channels by name, no new adapter
        kw.update(chan_proj_type="none")                     # checkpoint's 62-channel layout, pretrained channel embedding
    model = getattr(bm, cls)(**kw)
    sd = load_file(f"{path}/model.safetensors")
    own = model.state_dict()
    fixed, skipped = [], []
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape:
            if v.dim() == own[k].dim() and all(o <= w for o, w in zip(own[k].shape, v.shape)) and k != "chans_id":
                sd[k] = v[tuple(slice(0, o) for o in own[k].shape)]; fixed.append(k)
            else:
                sd.pop(k); skipped.append(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("cut:", fixed, "skipped:", skipped, "missing:", missing[:5], flush=True)
    model = model.to(dev).eval()
    return model, sf


def tokens(model, x, name, chans):
    if name == "labram":
        return model(x, ch_names=[c.upper() for c in chans])
    out = model(x, return_features=True)
    if isinstance(out, dict):
        if not getattr(tokens, "_shown", False):
            print("EEGPT outputs:", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")}, flush=True); tokens._shown = True
        out = out.get("features", next(v for v in out.values() if hasattr(v, "dim") and v.dim() >= 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--subs", default=",".join(str(i) for i in range(1, 24)))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    out_dir = Path(f"results/features/{a.model}"); out_dir.mkdir(parents=True, exist_ok=True)
    global COMMON
    COMMON = None
    for k in range(1, 24):
        c = [str(x) for x in np.load(E / f"sub-{k:03d}_epochs.npz")["ch"]]
        COMMON = c if COMMON is None else [x for x in COMMON if x in c]
    for s in [int(x) for x in a.subs.split(",")]:
        ref = np.load(Path("results/features/fm") / f"sub-{s:03d}.npz")      # same trials as the other ladder sets
        tri = ref["tri"]
        Fm = np.load(Path("results/fm_epochs") / f"sub-{s:03d}.npz")
        pos = {t: k for k, t in enumerate(Fm["trial"])}
        fch = [str(c) for c in Fm["ch"]]
        chans = COMMON
        X = Fm["X"][[pos[t] for t in tri]][:, [fch.index(c) for c in chans]].astype(np.float32).reshape(len(tri), len(chans), 400)
        sf = CFG[a.model][2]
        if sf != 200:
            X = resample(X, int(round(400 * sf / 200)), axis=2).astype(np.float32)
        if model is None:
            model, _ = load(a.model, chans, X.shape[2], dev)
        if a.model == "eegpt":                               # EEGPT takes its own 62-channel layout: map by name
            import json
            names = [c["ch_name"].upper() for c in json.load(open("external/eegpt/config.json"))["chs_info"]]
            up = [c.upper() for c in chans]
            X62 = np.zeros((len(X), len(names), X.shape[2]), np.float32)
            hit = 0
            for k, nmx in enumerate(names):
                if nmx in up:
                    X62[:, k] = X[:, up.index(nmx)]; hit += 1
            if s == int(a.subs.split(",")[0]):
                print("EEGPT channels matched:", hit, "of", len(names), flush=True)
            X = X62
        feats = []
        with torch.no_grad():
            for b in range(0, len(X), 32):
                t = tokens(model, torch.tensor(X[b:b + 32], device=dev), a.model, chans)
                if isinstance(t, (tuple, list)):
                    t = t[0]
                t = t.reshape(t.shape[0], -1, t.shape[-1]) if t.dim() > 2 else t[:, None]
                feats.append(torch.cat([t.mean(1), t.std(1) if t.shape[1] > 1 else t[:, 0] * 0], 1).cpu().numpy())
        F = np.concatenate(feats)
        np.savez_compressed(out_dir / f"sub-{s:03d}.npz", F=F.astype(np.float32), colw=np.ones(F.shape[1], np.float32), tri=tri)
        print(a.model, s, F.shape, flush=True)


if __name__ == "__main__":
    main()
