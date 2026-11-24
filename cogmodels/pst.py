"""Q-learning models and data loading for the Cavanagh-lab probabilistic learning tasks with affective feedback
(OpenNeuro ds003822 = Exp 1, ds003753 = Exp 2; Brown, Jackson & Cavanagh, CABN).

Task (decoded from code/Beh*/<id>_RL_Learn_*.mat, column data_trn, and the EEG triggers).
Exp 2 (ds003753): 160 trials, 40 per condition, valid-feedback rate .9 (conditions 1, 3) or .7 (2, 4); feedback
trigger S10x with x = col 6 (odd = positive); same columns 1, 2, 4, 5. Exp 1 (ds003822):
  200 learning trials; 4 interleaved conditions (col 1: 1..4), 50 trials each, each a two-option choice whose better
  option gives positive feedback with p = .7 (col 2: predetermined valid-feedback flag, 35/50 per condition).
  col 4 response side (EEG S10/S20), col 5 positive feedback (EEG S51/S53 = positive, S52/S54 = negative),
  col 6 money (+1 gain / -1 loss in conditions 1/2, 0 otherwise), col 8 response-to-feedback delay.
  Better option chosen <=> col5 == col2. Feedback EEG event: S5x (col 8 after the response); S10x follows ~1.8 s later.

Models (batched torch, unconstrained z, Gaussian priors in z; same conventions as cogmodels.models):
  QL   one learning rate            z = [logit alpha, log beta, kappa]
  QL2  gain / loss learning rates   z = [logit alpha_gain, logit alpha_loss, log beta, kappa]   (Frank et al. 2007)
Values Q[condition, option] start at 0.5; option 1 = better option. Stickiness kappa: bonus for repeating the last
choice made in the same condition. Records the signed prediction error pe = r - Q[cond, choice].
"""
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_COND = 4


class QL:
    name = "QL"
    param_names = ["logit_alpha", "log_beta", "kappa"]
    prior_mean = [0.0, 1.5, 0.0]
    prior_sd = [1.5, 0.8, 0.8]

    def rates(self, z):
        a = torch.sigmoid(z[:, 0])
        return a, a, torch.exp(z[:, 1])


class QL2:
    name = "QL2"
    param_names = ["logit_alpha_gain", "logit_alpha_loss", "log_beta", "kappa"]
    prior_mean = [0.0, 0.0, 1.5, 0.0]
    prior_sd = [1.5, 1.5, 0.8, 0.8]

    def rates(self, z):
        return torch.sigmoid(z[:, 0]), torch.sigmoid(z[:, 1]), torch.exp(z[:, 2])


MODELS = {"QL": QL(), "QL2": QL2()}


def run(model, z, cond, choice=None, reward=None, valid=None, p_good=0.7, record=False):
    """Simulate (choice is None) or score observed data. cond (B,T) long 0..3; choice (B,T) long {0 worse, 1 better};
    reward (B,T) float {0,1}; valid (B,T) bool. Returns loglik (B,) and, if record, q['pe'] (B,T).
    Conditions are independent learning problems with equal trial counts, so the recursion runs over the n = T/4
    trials of each condition for all four conditions at once (identical to the trial-by-trial recursion)."""
    B, T, dev = z.shape[0], cond.shape[1], z.device
    aG, aL, beta = model.rates(z)
    kappa = z[:, -1]
    n = T // N_COND
    tix = torch.arange(T, device=dev)[None, :].expand(B, T)
    order = torch.sort(cond * T + tix, dim=1).indices.view(B, N_COND, n)        # trial indices per condition, in time
    assert bool((torch.gather(cond, 1, order.view(B, T)).view(B, N_COND, n) == torch.arange(N_COND, device=dev)[None, :, None]).all())
    sim = choice is None
    if sim:
        choice = torch.zeros(B, T, dtype=torch.long, device=dev)
        reward = torch.zeros(B, T, device=dev)
        valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        pg = (p_good if torch.is_tensor(p_good) else torch.full((N_COND,), float(p_good))).to(dev)[None, :].expand(B, N_COND)
    g = lambda x: torch.gather(x, 1, order.view(B, T)).view(B, N_COND, n)
    C, R, V = (None, None, None) if sim else (g(choice).clamp(0, 1), g(reward).float(), g(valid).float())
    if sim:
        V = torch.ones(B, N_COND, n, device=dev)
    Q = torch.full((B, N_COND, 2), 0.5, device=dev)
    last = torch.zeros(B, N_COND, 2, device=dev)
    ll = torch.zeros(B, device=dev)
    pes, cs, rs = [], [], []
    for j in range(n):
        lg = beta[:, None, None] * Q + kappa[:, None, None] * last
        v = V[:, :, j]
        if sim:
            c = torch.multinomial(F.softmax(lg, -1).view(-1, 2), 1).view(B, N_COND)
            r = (torch.rand(B, N_COND, device=dev) < torch.where(c == 1, pg, 1 - pg)).float()
            cs.append(c); rs.append(r)
        else:
            c, r = C[:, :, j], R[:, :, j]
        oh = F.one_hot(c, 2).float()
        ll = ll + (v * (F.log_softmax(lg, -1) * oh).sum(-1)).sum(1)
        pe = (r - (Q * oh).sum(-1)) * v
        a = torch.where(pe >= 0, aG[:, None], aL[:, None])
        Q = Q + oh * (a * pe)[..., None] * v[..., None]
        last = torch.where(v[..., None] > 0, oh, last)
        pes.append(pe)
    def scatter(seq):
        x = torch.stack(seq, -1).reshape(B, T)
        out = torch.zeros(B, T, dtype=x.dtype, device=dev)
        return out.scatter(1, order.view(B, T), x)
    if sim:
        choice, reward = scatter(cs), scatter(rs)
    out = {"choice": choice, "reward": reward, "valid": valid, "loglik": ll}
    if record:
        out["q"] = {"pe": scatter(pes)}
    return out


def fit_map(model, cond, choice, reward, valid, steps=500, starts=4, seed=0):
    """Batched multi-start MAP (prior + exact likelihood) for all participants at once. Returns z (B,D)."""
    g = torch.Generator().manual_seed(seed)
    dev = cond.device
    mu0, sd0 = torch.tensor(model.prior_mean, device=dev), torch.tensor(model.prior_sd, device=dev)
    best, blp = None, None
    for s in range(starts):
        z = (mu0 + 0.5 * sd0 * torch.randn(len(cond), len(mu0), generator=g).to(dev) * (1.0 if s else 0.0)).clone().requires_grad_()
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            lp = run(model, z, cond, choice, reward, valid)["loglik"] - 0.5 * (((z - mu0) / sd0) ** 2).sum(1)
            (-lp.sum()).backward(); opt.step()
        with torch.no_grad():
            lp = run(model, z, cond, choice, reward, valid)["loglik"] - 0.5 * (((z - mu0) / sd0) ** 2).sum(1)
        if best is None:
            best, blp = z.detach().clone(), lp
        else:
            m = lp > blp; best[m] = z.detach()[m]; blp = torch.where(m, lp, blp)
    return best, blp


def load_behaviour(root):
    """Learning-phase data of every participant (BIDS order sub-001..), from code/Beh*/<Original_ID>_RL_Learn_*.mat.
    Returns dict of arrays (S, 200): cond (0..3), choice (1 = better), reward (positive feedback), valid, plus ids."""
    import scipy.io as sio
    root = Path(root)
    import csv
    rows = list(csv.DictReader(open(root / "participants.tsv"), delimiter="\t"))
    out = {k: [] for k in ("cond", "choice", "reward", "valid", "side", "money", "delay", "flag")}
    subs = []
    for r in rows:
        fs = glob.glob(str(root / "code" / "*" / f"{r['Original_ID']}_RL_Learn_*.mat"))
        if not fs:
            continue
        m = sio.loadmat(fs[0])
        d = [v for k, v in m.items() if not k.startswith("__")][0].astype(float)
        if d.shape[0] not in (160, 200) or d.shape[1] < 9:
            continue
        out["cond"].append(d[:, 1].astype(int) - 1)
        out["choice"].append((d[:, 5] == d[:, 2]).astype(int))
        out["reward"].append(d[:, 5].astype(float))
        out["flag"].append(d[:, 2].astype(int))
        out["side"].append(d[:, 4].astype(int))
        out["money"].append(d[:, 6].astype(float))
        out["delay"].append(d[:, 8].astype(float) if d.shape[0] == 200 else np.full(len(d), np.nan))
        out["valid"].append(np.isfinite(d[:, 5]))
        subs.append(int(re.sub(r"\D", "", r["participant_id"])))
    R = {k: np.stack(v) for k, v in out.items()}
    R["sub"] = np.array(subs)
    R["p_good"] = np.array([R["flag"][R["cond"] == c].mean() for c in range(N_COND)])     # valid-feedback rate
    return R
