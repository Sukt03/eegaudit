"""Shared helpers for the methodology pilots (read-only on existing project files).
Loads frozen encoders via experiments/eegbits/run_acq_controls.load_encoder and scores a similarity matrix with
several yield estimators at once: top-1, 2-way, Wolpaw/Fano bits, InfoNCE soft bound (LOO temperature), and the
nearest-neighbour 2-way (hard pool N=2 by CLIP cosine, audit C)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import wolpaw_bits  # noqa
from run_acq_controls import load_encoder, load_test  # noqa

OUT = Path("/path/to/eegaudit/results/method_pilots"); OUT.mkdir(exist_ok=True, parents=True)
LOG2 = np.log(2)
TAUS = np.logspace(-2.5, 0, 40)


def _norm(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def sim(e, Y):
    return _norm(e) @ _norm(Y).T


def soft_bits(S):
    """InfoNCE / Barber-Agakov lower bound: log2 N - LOO cross-entropy of a softmax over cosine rows, with the
    temperature chosen per query on the other queries (audit A estimator)."""
    n = S.shape[0]; rows = np.arange(n)
    def ce_all(tau):
        Z = S / tau; Z = Z - Z.max(1, keepdims=True)
        lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -lp[rows, rows] / LOG2            # per-row cross-entropy, bits
    CE = np.stack([ce_all(t) for t in TAUS])      # (n_tau, n)
    tot = CE.sum(1, keepdims=True)
    loo_ce = (tot - CE) / (n - 1)                 # mean CE over the other rows, per tau
    t_idx = loo_ce.argmin(0)                      # best tau for each held-out row
    return float(np.log2(n) - CE[t_idx, rows].mean())


def score(S, nn_idx=None):
    """S: (n_queries, n_candidates) cosine with the true candidate on the diagonal; nn_idx: for each query the
    index of its CLIP-nearest other candidate (hard pool N=2)."""
    n = S.shape[0]; diag = np.diag(S)
    rank = (S > diag[:, None]).sum(1) + 1
    top1 = float((rank == 1).mean())
    out = dict(top1=top1, two_way=float(((S < diag[:, None]).sum(1) / (n - 1)).mean()),
               bits=wolpaw_bits(top1, n), soft_bits=soft_bits(S), mean_rank=float(rank.mean()))
    if nn_idx is not None:
        out["nn_two_way"] = float((diag > S[np.arange(n), nn_idx]).mean())
    return out


def nn_index(Y):
    C = sim(Y, Y); np.fill_diagonal(C, -2); return C.argmax(1)


def save(name, obj):
    p = OUT / name; p.write_text(json.dumps(obj, indent=1)); print("->", p, flush=True)
