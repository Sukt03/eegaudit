"""Trained deep decoders, leave-one-participant-out (no circularity): does a trained decoder raise EEG's worth?

For each held-out participant h: train on the other 22 participants' feedback epochs to predict their own read-out
quantity (prediction error of Rescorla-Wagner with decay at their MAP estimate, residualized on outcome, chosen arm,
their interaction and cubic time, z-scored), with one further participant held out for early stopping; then predict
h's trials. h's labels are never used in training. The prediction is saved as a one-column feature set
(results/features/<name>/sub-XXX.npz) on exactly the trials of the other ladder sets, and priced with
price_plugin.py like every other decoder.
Models: eegnet (compact CNN, Lawhern et al. 2018; braindecode EEGNet) and cbramod_ft (CBraMod, all layers fine-tuned,
channel-mean of the post-feedback token -> linear head).
Input: 200 Hz feedback epochs, -1..1 s (two 1-s patches), uV/100 (results/fm_epochs).
Usage: python deep_decoders.py --model eegnet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "external/CBraMod")
from cogmodels import models as M  # noqa: E402

QUANT = "results/full_36318/behaviour_quantities_flow.npz"
E = Path("results/eeg_measures")


def resid(v, cols):
    C = np.c_[np.ones(len(v)), np.column_stack(cols)]
    b, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ b


def load_all():
    Q = np.load(QUANT)
    model = M.EXTRA_MODELS["RWD"]
    Z = torch.tensor(np.load("results/map_RWD.npy")).float()
    probs = M.reward_schedule(480)
    common = None
    for s in range(1, 24):
        c = [str(x) for x in np.load(E / f"sub-{s:03d}_epochs.npz")["ch"]]
        common = c if common is None else [x for x in common if x in c]
    data = []
    for i in range(23):
        s = i + 1
        tri = np.load(f"results/features/fm/sub-{s:03d}.npz")["tri"]          # same trials as the ladder
        Fm = np.load(f"results/fm_epochs/sub-{s:03d}.npz")
        pos = {t: k for k, t in enumerate(Fm["trial"])}
        fch = [str(x) for x in Fm["ch"]]
        X = Fm["X"][[pos[t] for t in tri]][:, [fch.index(c) for c in common]].astype(np.float32)   # (n, ch, 2, 200)
        with torch.no_grad():
            q = M.run(model, Z[i][None], probs, torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None],
                      torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy()
        c_, r_ = Q["choice"][i][tri], Q["reward"][i][tri]
        cols = [r_] + [(c_ == k).astype(float) for k in range(3)] + [(c_ == k).astype(float) * r_ for k in range(3)] + \
               [(tri / 480.0) ** p for p in (1, 2, 3)]
        y = resid(q[tri].astype(float), cols); y = (y - y.mean()) / (y.std() + 1e-9)
        assert len(y) == len(X), (s, len(y), len(X))
        data.append((X, y.astype(np.float32), tri))
    return data, len(common)


class CBraHead(nn.Module):
    def __init__(self):
        super().__init__()
        from models.cbramod import CBraMod
        self.enc = CBraMod()
        self.enc.load_state_dict(torch.load("external/CBraMod/pretrained_weights/pretrained_weights.pth", map_location="cpu"))
        self.enc.proj_out = nn.Identity()
        self.head = nn.Linear(200, 1)

    def forward(self, x):                          # x (B, C, 2, 200)
        return self.head(self.enc(x)[:, :, 1, :].mean(1)).squeeze(-1)


def make(name, n_ch):
    if name == "eegnet":
        from braindecode.models import EEGNet
        net = EEGNet(n_chans=n_ch, n_outputs=1, n_times=400)
        return lambda x: net(x.reshape(x.shape[0], x.shape[1], -1)).reshape(-1), net, 1e-3, 30
    m = CBraHead()
    return lambda x: m(x), m, 1e-4, 6


def train_fold(name, data, h, n_ch, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rest = [k for k in range(len(data)) if k != h]
    val = rest[(h * 7) % len(rest)]
    tr = [k for k in rest if k != val]
    Xtr = np.concatenate([data[k][0] for k in tr]); ytr = np.concatenate([data[k][1] for k in tr])
    fwd, net, lr, epochs = make(name, n_ch)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    Xv = torch.tensor(data[val][0], device=dev); yv = torch.tensor(data[val][1], device=dev)
    best, best_state, bs = -1e9, None, 128
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
            pv = torch.cat([fwd(Xv[b:b + 256]) for b in range(0, len(Xv), 256)])
            r = float(torch.corrcoef(torch.stack([pv, yv]))[0, 1])
        if r > best:
            best = r; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    Xh = torch.tensor(data[h][0], device=dev)
    with torch.no_grad():
        ph = torch.cat([fwd(Xh[b:b + 256]) for b in range(0, len(Xh), 256)]).cpu().numpy()
    return ph, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["eegnet", "cbramod_ft"])
    ap.add_argument("--subs", default=",".join(str(i) for i in range(1, 24)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, n_ch = load_all()
    out = Path(f"results/features/{a.model}"); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.subs.split(",")]:
        h = s - 1
        ph, rval = train_fold(a.model, data, h, n_ch, dev, a.seed + s)
        r_held = float(np.corrcoef(ph, data[h][1])[0, 1])
        np.savez_compressed(out / f"sub-{s:03d}.npz", F=ph[:, None].astype(np.float32), colw=np.ones(1, np.float32),
                            tri=data[h][2], r_heldout=r_held, r_val=rval)
        print(f"{a.model} sub {s}: held-out r={r_held:.3f} (early-stop val r={rval:.3f})", flush=True)


if __name__ == "__main__":
    main()
