"""Two-choice learning tasks with a built-in "nothing to learn" control, for the multi-dataset pipeline.

ds007647 "Different Doors" (Oloriz & Hassall 2026): 40 people, 20 blocks x 20 trials, choose one of two doors; door
  values reset each block. Learnable blocks (cond 2): better door wins p = 0.6, other p = 0.1. Unlearnable blocks
  (cond 1): outcomes replayed from learnable blocks in random order, independent of the choice. Not told to participants.
ds004147 "Average Task Value" (Hassall, Hunt & Holroyd): 12 people, 3 tasks x 144 trials, 6 cues per task, left/right.
  High-value cues: feedback validity 0.8 (learnable, cond 2); low-value cues: validity 0.5, a coin toss (cond 1).
  Values reset per task; one Q pair per cue (state).

Models (batched torch), Q reset to 0.5 at block start, one (Q_left, Q_right) pair per state:
  RW   learning rate alpha, softmax(beta) on Q plus stickiness kappa on the previous response (same state)
  RWD  RW plus decay of the unchosen value toward 0.5 at rate d
  RW2  separate learning rates for learnable (cond 2) and unlearnable (cond 1) trials
pe_t = r_t - Q[state, choice] before the update; missed trials skipped (pe = 0, valid = False).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _pack(rows, keep):
    out = {k: np.stack(v) for k, v in rows.items()}
    out["sub"] = np.array(keep)
    return out


def load_doors(root="data_raw/ds007647"):
    root = Path(root)
    rows = {k: [] for k in ("block", "cond", "state", "choice", "reward", "valid")}
    keep = []
    for p in sorted(root.glob("sub-*")):
        s = int(p.name[4:])
        f = p / f"beh/sub-{s:02d}_task-differentdoors_beh.tsv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        d = pd.read_csv(f, sep="\t", na_values="n/a")
        if len(d) != 400:
            continue
        v = d["responded"].fillna(0).astype(int).values == 1
        rows["block"].append(d["blockNum"].values.astype(int) - 1)
        rows["cond"].append(d["condition"].values.astype(int))
        rows["state"].append(np.zeros(400, int))
        rows["choice"].append(np.where(v, d["response"].fillna(1).values - 1, 0).astype(int))
        rows["reward"].append(np.where(v, (d["outcome"].fillna(2).values == 1).astype(float), 0.0))
        rows["valid"].append(v)
        keep.append(s)
    return _pack(rows, keep)


def load_cues(root="data_raw/ds004147"):
    root = Path(root)
    rows = {k: [] for k in ("block", "cond", "state", "choice", "reward", "valid")}
    keep = []
    for p in sorted(root.glob("sub-*")):
        s = int(p.name[4:])
        f = p / f"beh/sub-{s:02d}_task-casinos_beh.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t", na_values="n/a")
        if len(d) != 432:
            continue
        v = (d["early"].fillna(1) == 0).values & (d["invalid"].fillna(1) == 0).values & d["response"].isin([1, 2]).values
        task = d["task"].values.astype(int) - 1
        cue = d["cue"].values.astype(int)
        # cue ids are unique within a task; map (task, cue) to 0..5 within the block
        st = np.zeros(len(d), int)
        for t in np.unique(task):
            u = sorted(np.unique(cue[task == t]))
            st[task == t] = [u.index(c) for c in cue[task == t]]
        rows["block"].append(task)
        rows["cond"].append(np.where(d["prob"].values >= 80, 2, 1))
        rows["state"].append(st)
        rows["choice"].append(np.where(v, d["response"].fillna(1).values - 1, 0).astype(int))
        rows["reward"].append(np.where(v, d["outcome"].fillna(0).values.astype(float), 0.0))
        rows["valid"].append(v)
        keep.append(s)
    return _pack(rows, keep)


LOADERS = {"ds007647": load_doors, "ds004147": load_cues}


class Model:
    def __init__(self, name, param_names, prior_mean, prior_sd):
        self.name, self.param_names = name, param_names
        self.prior_mean, self.prior_sd = prior_mean, prior_sd


MODELS = {
    "RW": Model("RW", ["logit_alpha", "log_beta", "kappa"], [0.0, 1.0, 0.0], [1.5, 1.0, 1.0]),
    "RWD": Model("RWD", ["logit_alpha", "logit_d", "log_beta", "kappa"], [0.0, -2.0, 1.0, 0.0], [1.5, 1.5, 1.0, 1.0]),
    "RW2": Model("RW2", ["logit_alpha_L", "logit_alpha_U", "log_beta", "kappa"], [0.0, 0.0, 1.0, 0.0], [1.5, 1.5, 1.0, 1.0]),
}


def run(model, z, block, cond, state, choice, reward, valid, record=False, n_states=6):
    """z (B, D); task arrays (B, T) torch tensors. Returns loglik (B,) and, with record, pe (B, T)."""
    B, T = choice.shape
    dt, dev = z.dtype, z.device
    nm = model.name
    beta = torch.exp(z[:, model.param_names.index("log_beta")])
    kappa = z[:, model.param_names.index("kappa")]
    if nm == "RW2":
        aL, aU = torch.sigmoid(z[:, 0]), torch.sigmoid(z[:, 1])
    else:
        a = torch.sigmoid(z[:, 0])
    d = torch.sigmoid(z[:, 1]) if nm == "RWD" else None
    Q = torch.full((B, n_states, 2), 0.5, dtype=dt, device=dev)
    prev = torch.zeros(B, n_states, 2, dtype=dt, device=dev)
    ll = torch.zeros(B, dtype=dt, device=dev)
    ar = torch.arange(B, device=dev)
    pes = []
    for t in range(T):
        if t > 0:
            nb = (block[:, t] != block[:, t - 1]).to(dt)[:, None, None]
            Q = Q * (1 - nb) + 0.5 * nb
            prev = prev * (1 - nb)
        s = state[:, t].long()
        qs, ps = Q[ar, s], prev[ar, s]
        lp = torch.log_softmax(beta[:, None] * qs + kappa[:, None] * ps, -1)
        c = choice[:, t].long(); v = valid[:, t].to(dt)
        ll = ll + v * lp.gather(1, c[:, None])[:, 0]
        oh = torch.nn.functional.one_hot(c, 2).to(dt)
        pe = reward[:, t].to(dt) - (qs * oh).sum(1)
        lr = torch.where(cond[:, t] == 2, aL, aU) if nm == "RW2" else a
        qn = qs + (lr * pe * v)[:, None] * oh
        if d is not None:
            qn = qn + (d[:, None] * (0.5 - qn)) * (1 - oh) * v[:, None]
        pn = torch.where(v[:, None] > 0, oh, ps)
        sel = torch.nn.functional.one_hot(s, n_states).to(dt)[:, :, None]
        Q = Q * (1 - sel) + sel * qn[:, None, :]
        prev = prev * (1 - sel) + sel * pn[:, None, :]
        pes.append(pe * v)
    out = {"loglik": ll}
    if record:
        out["pe"] = torch.stack(pes, 1)
    return out


def tensors(B, device="cpu", rep=1):
    return [torch.tensor(B[k], device=device).repeat_interleave(rep, 0) for k in ("block", "cond", "state", "choice", "reward", "valid")]


def fit_map(model, B, starts=8, steps=400, device="cpu", seed=0, n_states=6):
    """Multi-start MAP, batched over participants x starts. Returns z (N, D) and loglik (N,)."""
    g = torch.Generator().manual_seed(seed)
    N = B["choice"].shape[0]; D = len(model.param_names)
    mu = torch.tensor(model.prior_mean, device=device); sd = torch.tensor(model.prior_sd, device=device)
    z = (mu + sd * torch.randn(N * starts, D, generator=g).to(device)).requires_grad_(True)
    arr = tensors(B, device, starts)
    opt = torch.optim.Adam([z], lr=0.08)
    for _ in range(steps):
        lp = run(model, z, *arr, n_states=n_states)["loglik"]
        loss = -(lp - 0.5 * (((z - mu) / sd) ** 2).sum(1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lp = run(model, z, *arr, n_states=n_states)["loglik"]
        obj = (lp - 0.5 * (((z - mu) / sd) ** 2).sum(1)).view(N, starts)
        best = obj.argmax(1)
        zb = z.view(N, starts, D)[torch.arange(N), best]
        llb = lp.view(N, starts)[torch.arange(N), best]
    return zb.detach().cpu(), llb.cpu()
