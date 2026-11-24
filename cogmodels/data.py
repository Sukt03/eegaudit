"""ds003458 (Cavanagh three-armed bandit EEG) behaviour parsing from BIDS events.tsv.

Stimulus codes (README _ Code.txt): concatenation of stim ids [X=1, Y=2, Z=3] in positions
[up, left, right], minus 100 (so code + 100 gives the three digits). Response 1=up, 2=left,
3=right, 4=no response. Feedback 5 = "+1" (win), 6 = "~" (loss).
The BIDS trial_type text labels the same digits as (Left, Right, Up), which contradicts the
README, so both decodings are scored against the known reward schedule and the better one used.
"""
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://s3.amazonaws.com/openneuro.org/ds003458"
STIM = {23, 32, 113, 131, 212, 221}
VARIANTS = {"readme": ("up", "left", "right"), "text": ("left", "right", "up")}
RESP = {1: "up", 2: "left", 3: "right"}


def events_path(root, sub):
    return Path(root) / f"sub-{sub:03d}" / "eeg" / f"sub-{sub:03d}_task-ThreeArmedBandit_events.tsv"


def fetch_events(root, sub):
    p = events_path(root, sub)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/sub-{sub:03d}/eeg/{p.name}", p)
    return p


def parse_trials(path):
    df = pd.read_csv(path, sep="\t")
    trials, cur = [], None
    for onset, val in zip(df["onset"], df["value"].astype(str)):
        m = re.match(r"S\s*(\d+)", val)
        if not m:
            continue
        code = int(m.group(1))
        if code in STIM:
            if cur:
                trials.append(cur)
            cur = {"stim": code, "resp": None, "fb": None, "fb_onset": np.nan, "stim_onset": onset,
                   "resp_onset": np.nan}
        elif cur is None:
            continue
        elif code in (1, 2, 3, 4) and cur["resp"] is None:
            cur["resp"] = code
            cur["resp_onset"] = onset
        elif code in (5, 6) and cur["fb"] is None:
            cur["fb"], cur["fb_onset"] = code, onset
    if cur:
        trials.append(cur)
    return trials


def decode(trials, variant):
    order = VARIANTS[variant]
    T = len(trials)
    choice = np.full(T, -1)
    reward = np.zeros(T)
    missing = np.ones(T, bool)
    fb_onset = np.full(T, np.nan)
    stim_onset = np.array([tr["stim_onset"] for tr in trials], float)
    resp_onset = np.array([tr["resp_onset"] for tr in trials], float)
    for t, tr in enumerate(trials):
        if tr["resp"] in RESP and tr["fb"] in (5, 6):
            digits = str(tr["stim"] + 100)
            pos = dict(zip(order, digits))
            choice[t] = int(pos[RESP[tr["resp"]]]) - 1
            reward[t] = 1.0 if tr["fb"] == 5 else 0.0
            missing[t] = False
            fb_onset[t] = tr["fb_onset"]
    return {"choice": choice, "reward": reward, "missing": missing, "fb_onset": fb_onset,
            "stim_onset": stim_onset, "resp_onset": resp_onset}


def reward_loglik(d, probs):
    """Log-likelihood of observed outcomes under the known schedule for the decoded choices."""
    v = ~d["missing"]
    t = np.arange(len(d["choice"]))[v]
    p = probs[d["choice"][v], t].clip(1e-3, 1 - 1e-3)
    r = d["reward"][v]
    return float((r * np.log(p) + (1 - r) * np.log(1 - p)).sum())


def load_subject(root, sub, probs):
    trials = parse_trials(fetch_events(root, sub))[: probs.shape[1]]
    T = len(trials)
    pr = probs[:, :T]
    decs = {v: decode(trials, v) for v in VARIANTS}
    lls = {v: reward_loglik(d, pr) for v, d in decs.items()}
    best = max(lls, key=lls.get)
    out = decs[best]
    out.update({"sub": sub, "T": T, "variant": best, "variant_ll": lls})
    return out
