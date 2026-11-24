"""EEGNet trained leave-one-participant-out on ds004595 (non-circular read-out), following deep_decoders.py.
For held-out participant h: train on the other participants' feedback epochs (50 Hz, -0.2..0.8 s, uV/100, common
channels, the non-rejected trials of the ensemble feature set) to predict their own residualized, z-scored signed PE
(RW with decay + reset at insertion, at their MAP; residualized on outcome, chosen slot, slot x outcome, cubic time);
one further participant for early stopping (best validation r over 30 epochs); predict h. Output: one-column feature
set results/features_ds004595/eegnet/sub-XXX.npz (same trials as raw/ens) with r_heldout.
EEGNet (Lawhern et al. 2018, braindecode) with kernel_length 16 (0.32 s at 50 Hz; the 64-sample default exceeds the
50-sample epoch).
--post: post-feedback samples only (0-0.8 s; the epochs are already low-passed at 20 Hz) on interior channels (the
frontopolar/temporal ring of rung0_ocular.RING removed), as deep_decoders_post.py -> features_ds004595/eegnet_post.
Usage: python deep_decoders_ds004595.py [--post] --subs 1,2,...   (subject numbers as in the BIDS dataset)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from cogmodels import models as M
from ds004595_task import run_reset
from gpu_features import EP, SUBS, common_channels

B = np.load("results/ds004595/behaviour.npz")


def resid(v, C):
    return v - C @ np.linalg.lstsq(C, v, rcond=None)[0]


def load_all(post=False):
    common = common_channels()
    if post:
        from rung0_ocular import RING
        common = [c for c in common if c not in RING]
    model = M.EXTRA_MODELS["RWD"]
    Z = torch.tensor(np.load("results/ds004595/map_RWD.npy")).float()
    T = int(B["T"])
    with torch.no_grad():
        pe = run_reset(model, Z, torch.full((3, T), 0.5), torch.tensor(B["resets"]).float(), torch.tensor(B["choice"]),
                       torch.tensor(B["reward"]).float(), torch.tensor(B["missing"]), record=True)["q"]["pe"].numpy()
    row = {int(s): i for i, s in enumerate(B["sub"])}
    data = {}
    for s in SUBS:
        i = row[s]
        Ep = np.load(EP / f"sub-{s:03d}_epochs.npz")
        chl = [str(x) for x in Ep["ch"]]
        pos = {t: k for k, t in enumerate(Ep["trial"])}
        tri = np.load(f"results/features_ds004595/ens/sub-{s:03d}.npz")["tri"]
        X = Ep["X"][[pos[t] for t in tri]][:, [chl.index(c) for c in common]].astype(np.float32) / 100.0
        if post:
            X = np.ascontiguousarray(X[:, :, Ep["times"] >= 0])
        c, r = B["choice"][i][tri], B["reward"][i][tri]
        oh = np.eye(3)[c]
        C = np.column_stack([np.ones(len(tri)), r, oh, oh * r[:, None], tri / T, (tri / T) ** 2, (tri / T) ** 3])
        y = resid(pe[i][tri].astype(float), C); y = (y - y.mean()) / (y.std() + 1e-9)
        data[s] = (X, y.astype(np.float32), tri)
    return data, len(common)


def train_fold(data, h, n_ch, dev, seed, epochs=30, lr=1e-3, bs=128):
    from braindecode.models import EEGNet
    torch.manual_seed(seed); np.random.seed(seed)
    rest = [s for s in data if s != h]
    val = rest[(SUBS.index(h) * 7) % len(rest)]
    tr = [s for s in rest if s != val]
    Xtr = np.concatenate([data[s][0] for s in tr]); ytr = np.concatenate([data[s][1] for s in tr])
    n_t = Xtr.shape[2]
    net = EEGNet(n_chans=n_ch, n_outputs=1, n_times=n_t, kernel_length=16).to(dev)
    fwd = lambda x: net(x).reshape(-1)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    Xv = torch.tensor(data[val][0], device=dev); yv = torch.tensor(data[val][1], device=dev)
    best, best_state = -1e9, None
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(Xtr))
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            xb = torch.tensor(Xtr[idx], device=dev); yb = torch.tensor(ytr[idx], device=dev)
            loss = ((fwd(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(Xv[b:b + 512]) for b in range(0, len(Xv), 512)])
            r = float(torch.corrcoef(torch.stack([pv, yv]))[0, 1])
        if r > best:
            best = r; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    Xh = torch.tensor(data[h][0], device=dev)
    with torch.no_grad():
        ph = torch.cat([fwd(Xh[b:b + 512]) for b in range(0, len(Xh), 512)]).cpu().numpy()
    return ph, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(str(s) for s in SUBS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--post", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, n_ch = load_all(a.post)
    name = "eegnet_post" if a.post else "eegnet"
    out = Path(f"results/features_ds004595/{name}"); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.subs.split(",")]:
        ph, rval = train_fold(data, s, n_ch, dev, a.seed + s)
        r_held = float(np.corrcoef(ph, data[s][1])[0, 1])
        np.savez_compressed(out / f"sub-{s:03d}.npz", F=ph[:, None].astype(np.float32), colw=np.ones(1, np.float32),
                            tri=data[s][2], r_heldout=r_held, r_val=rval)
        print(f"{name} n_ch={n_ch} sub {s}: held-out r={r_held:.3f} (early-stop val r={rval:.3f})", flush=True)


if __name__ == "__main__":
    main()
