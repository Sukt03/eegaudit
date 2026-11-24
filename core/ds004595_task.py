"""ds004595 (Campbell & Cavanagh; 3-armed bandit with novel options): behaviour, task schedule and RW-with-decay
with value reset at novel-option insertion.

Task (code/BEH_EXPLAIN.m): three option slots A, B, C with fixed reward rates (0.2 / 0.5 / 0.8); every 5-9 trials a
new stimulus replaces one slot, with a rate drawn from {0.2, 0.5, 0.8}. 350 trials per participant.
Trial-wise behaviour: code/Model/model_dat_aud_bandit.csv (stimulus IDs and rates per slot, chosen ID, feedback);
row k aligns with the k-th feedback event (FB: Win / Loss / No Resp) in the BIDS events file (checked in load()).

Model: RW with decay (cogmodels.models.RWDecay) where a slot's value resets to the prior value 0.5 when a new
stimulus is inserted in it (the learner has no knowledge of a new stimulus). run_reset() is models.run plus this reset;
install() patches models.run / models.reward_schedule so exp_rate.py and exp_exact_joint.py simulate this task.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cogmodels import models as M

ROOT = Path("data_raw/ds004595")
_CSV = None


def _csv():
    global _CSV
    if _CSV is None:
        rows = [l.rstrip("\n").split(",") for l in open(ROOT / "code/Model/model_dat_aud_bandit.csv")]
        h = rows[0]
        _CSV = {}
        for r in rows[1:]:
            _CSV.setdefault(r[0], []).append({k: r[i] for i, k in enumerate(h)})
    return _CSV


def pid_of(sub):
    rows = [l.rstrip("\n").split("\t") for l in open(ROOT / "participants.tsv")]
    return rows[sub][1].strip()


def events(sub):
    ev = [l.rstrip("\n").split("\t") for l in open(ROOT / f"sub-{sub:03d}/eeg/sub-{sub:03d}_task-ThreeArmedBandit_events.tsv")][1:]
    fb, cue_t, stim = [], None, []
    for row in ev:
        t, v = float(row[0]), row[3]
        if v.startswith("CUE"):
            cue_t = t
        elif v.startswith("FB"):
            fb.append((t, v)); stim.append(cue_t)
    return fb, stim


def load(sub):
    """Per-trial arrays for one participant (T = 350): choice slot (0..2), reward, missing, resets (T,3), rates (3,T),
    feedback and stimulus onsets (s). Raises if the CSV and the EEG feedback events disagree."""
    R = sorted(_csv()[pid_of(sub)], key=lambda r: int(r["Trial"]))
    fb, stim = events(sub)
    T = len(R)
    if len(fb) != T:
        raise ValueError(f"sub {sub}: {len(fb)} feedback events vs {T} behaviour rows")
    ids = np.array([[float(r[f"stim{s}_ID"]) for s in "ABC"] for r in R])
    rates = np.array([[float(r[f"stim{s}_p"]) for s in "ABC"] for r in R]).T
    chosen = np.array([float(r["chosen_ID"]) for r in R])
    fbv = np.array([float(r["chosen_fb"]) for r in R])
    missing = chosen == 0
    choice = np.zeros(T, int)
    for t in range(T):
        if not missing[t]:
            hit = np.where(ids[t] == chosen[t])[0]
            if len(hit) != 1:
                missing[t] = True
            else:
                choice[t] = hit[0]
    reward = np.where(missing, 0.0, fbv)
    resets = np.zeros((T, 3))
    resets[1:] = (ids[1:] != ids[:-1]).astype(float)
    ev_win = np.array([1.0 if v == "FB: Win" else 0.0 for _, v in fb])
    ev_miss = np.array([v == "FB: No Resp" for _, v in fb])
    agree = float(np.mean((ev_win == reward)[~missing])) if (~missing).any() else 0.0
    miss_agree = float(np.mean(ev_miss == missing))
    return {"T": T, "choice": choice, "reward": reward, "missing": missing, "resets": resets, "rates": rates,
            "fb_onset": np.array([t for t, _ in fb]), "stim_onset": np.array([s if s is not None else np.nan for s in stim]),
            "agree_fb": agree, "agree_missing": miss_agree}


def run_reset(model, z, probs, resets, choices=None, rewards=None, missing=None, p_miss=0.02, record=False, count=None):
    """cogmodels.models.run with reset at novel-option insertion: before trial t, every per-arm state entry of arm k
    returns to the model's initial value where resets[..., t, k] = 1 (RW/RWD: Q = 0.5; HGF: mu2, sa2, mu3, sa3;
    Bayes-adaptive: Beta counts a, b). resets: (T,3) shared or (B,T,3) per participant."""
    B, T, dev = z.shape[0], probs.shape[1], z.device
    rs = torch.as_tensor(resets, dtype=torch.float32, device=dev)
    rs = rs[None].expand(B, -1, -1) if rs.dim() == 2 else rs
    p = model.params(z)
    s = model.init(B, dev)
    s0 = model.init(B, dev)
    simulate = choices is None
    if simulate:
        missing = torch.rand(B, T, device=dev) < p_miss
        choices = torch.zeros(B, T, dtype=torch.long, device=dev)
        rewards = torch.zeros(B, T, device=dev)
    ll = torch.zeros(B, device=dev)
    rec = {}
    kappa = z[:, -1:]
    prev = torch.zeros(B, M.N_ARMS, device=dev)
    for t in range(T):
        s = {k: (v * (1 - rs[:, t]) + s0[k] * rs[:, t]) if (k in s0 and v.shape[-1] == M.N_ARMS) else v for k, v in s.items()}
        s = model.pre(s, p)
        lg = model.logits(s, p) + kappa * prev
        valid = (~missing[:, t]).float()
        if simulate:
            c = torch.multinomial(F.softmax(lg, -1), 1).squeeze(1)
            r = (torch.rand(B, device=dev) < probs[c, t]).float()
            c = torch.where(valid.bool(), c, torch.zeros_like(c))
            r = r * valid
            choices[:, t], rewards[:, t] = c, r
        else:
            c, r = choices[:, t].clamp(min=0), rewards[:, t]
        ll = ll + valid * (1.0 if count is None else count[:, t].float()) * M._sel(F.log_softmax(lg, -1), c)
        lp_rep = M._sel(F.log_softmax(lg, -1), prev.argmax(1)) * (prev.sum(1) > 0).float()
        prev = F.one_hot(c, M.N_ARMS).float() * valid[:, None] + prev * (1 - valid[:, None])
        s, q = model.update(s, p, c, r, valid)
        if record:
            q = dict(q)
            q["logp_rep"] = lp_rep
            lsm = F.log_softmax(lg, -1)
            q["entropy"] = -(lsm.exp() * lsm).sum(-1)
            for k, v in q.items():
                rec.setdefault(k, []).append(v)
    out = {"choices": choices, "rewards": rewards, "missing": missing, "loglik": ll}
    if record:
        out["q"] = {k: torch.stack(v, 1) for k, v in rec.items()}
    return out


def novelty_schedule(T=480, seed=0, init=(0.8, 0.2, 0.5)):
    """The task's generative algorithm: start rates as in the real sessions; every 5-9 trials a random slot gets a new
    stimulus with rate in {0.2, 0.5, 0.8}. Returns rates (3,T) and resets (T,3)."""
    rng = np.random.default_rng(seed)
    rates = np.zeros((3, T)); resets = np.zeros((T, 3))
    cur = np.array(init, float)
    nxt = rng.integers(5, 10)
    for t in range(T):
        if t == nxt:
            k = rng.integers(3); cur[k] = rng.choice([0.2, 0.5, 0.8]); resets[t, k] = 1.0
            nxt = t + rng.integers(5, 10)
        rates[:, t] = cur
    return rates, resets


def install(T=480, seed=0):
    """Make models.run / models.reward_schedule simulate and evaluate the ds004595 task (shared schedule)."""
    rates, resets = novelty_schedule(T, seed)
    orig = M.run

    def run(model, z, probs, *args, **kw):
        return run_reset(model, z, probs, resets[:probs.shape[1]], *args, **kw)

    def reward_schedule(T_=480, *a, device="cpu", **k):
        return torch.tensor(rates[:, :T_], dtype=torch.float32, device=device)

    M.run = run
    M.reward_schedule = reward_schedule
    return rates, resets
