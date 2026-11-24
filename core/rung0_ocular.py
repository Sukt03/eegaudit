"""Rung 0 (is it brain?): decoding beyond the eyes, using the recorded EOG channel before ICA.

Ocular regressors (pre-ICA, 0.1-30 Hz, 150-600 ms after feedback, 50 Hz): vertical EOG, Fp1-Fp2 and F7-F8 (horizontal
proxies), and a blink indicator (EOG peak-to-peak > 75 uV). Three decoders share the nuisance set (outcome, arm,
outcome x arm, cubic time), PCA-16 and ridge, 10 contiguous folds:
  full            all channels (ICA-cleaned epochs)
  interior_eogreg 49 interior channels (no Fp/AF/F7/F8/temporal ring), ocular regressors removed from every feature in-fold
  eog_only        the ocular regressors as features
Pre-stated pass rule (per latent): P1 interior_eogreg excess > 0 (Wilcoxon one-sided p < .025) and >= 50% of full;
P2 interior_eogreg > eog_only (paired, one-sided p < .025); P3 decoder pattern at 250-350 ms peaks off Fp/AF and
|spatial r| with the ocular map < 0.5 (computed in the figure script from rung-2 Haufe patterns and the map saved here).
Usage: python rung0_ocular.py --sub 3 --out results/rung0_ocular
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

from cogmodels import models as M
from cogmodels.data import load_subject

ROOT = Path("data_raw/ds003458")
E = Path("results/eeg_measures")
QUANT = "results/full_36318/behaviour_quantities_flow.npz"
RING = ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "AFz", "F7", "F8", "FT9", "FT10", "T7", "T8", "TP9", "TP10"]
LATS = ("pe", "abs_pe", "kg")


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def resid(v, cols):
    C = np.c_[np.ones(len(v)), np.column_stack(cols)]
    b, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ b


def nuis(Q, j, tri):
    c, r = Q["choice"][j], Q["reward"][j]
    return [r[tri]] + [(c[tri] == k).astype(float) for k in range(3)] + \
           [(c[tri] == k).astype(float) * r[tri] for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]


def latent(Q, j, tag, tri):
    v = Q["BAMDP_kg" if tag == "kg" else "BAMDP_pe"][j][tri].astype(float)
    y = resid(np.abs(v) if tag == "abs_pe" else v, nuis(Q, j, tri))
    return (y - y.mean()) / (y.std() + 1e-9)


def cv(F, y, O=None, k=16, lam=10.0):
    n = len(y)
    f = np.minimum((np.arange(n) * 10) // n, 9)
    P = np.zeros(n)
    for b in range(10):
        tr, te = f != b, f == b
        A, B = F[tr].copy(), F[te].copy()
        if O is not None:                                   # remove ocular regressors from every feature, in-fold
            Oc = np.c_[np.ones(tr.sum()), O[tr]]
            beta = np.linalg.lstsq(Oc, A, rcond=None)[0]
            A = A - Oc @ beta
            B = B - np.c_[np.ones(te.sum()), O[te]] @ beta
        mu, sd = A.mean(0), A.std(0) + 1e-6
        A, B = (A - mu) / sd, (B - mu) / sd
        _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
        V = Vt[:min(k, len(Vt))].T
        Atr, Ate = A @ V, B @ V
        w = np.linalg.solve(Atr.T @ Atr + lam * len(Atr) * np.eye(V.shape[1]), Atr.T @ (y[tr] - y[tr].mean()))
        P[te] = Ate @ w
    return corr(P, y)


def ocular(sub, trials_keep):
    probs = M.reward_schedule(480).numpy()
    d = load_subject(ROOT, sub, probs)
    fb = d["fb_onset"][:d["T"]]
    raw = mne.io.read_raw_eeglab(ROOT / f"sub-{sub:03d}/eeg/sub-{sub:03d}_task-ThreeArmedBandit_eeg.set",
                                 preload=True, verbose="error")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    if "CPz" not in raw.ch_names:
        mne.add_reference_channels(raw, "CPz", copy=False)
    raw.set_eeg_reference("average", verbose="error")
    raw.filter(0.1, 30.0, picks="all", verbose="error")
    ok = np.where(~np.isnan(fb))[0]
    samp = np.round(fb[ok] * raw.info["sfreq"]).astype(int)
    ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    picks = eog + [c for c in ("Fp1", "Fp2", "F7", "F8") if c in raw.ch_names]
    ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=picks, preload=True, reject=None,
                    decim=int(round(raw.info["sfreq"] / 50)), verbose="error")
    X = ep.get_data(picks=picks) * 1e6
    tri = ok[ep.selection]
    w = (ep.times >= 0.15) & (ep.times <= 0.60)
    nm = list(picks)
    feats = [X[:, nm.index(eog[0]), w]] if eog else []
    if "Fp1" in nm and "Fp2" in nm:
        feats.append(X[:, nm.index("Fp1"), w] - X[:, nm.index("Fp2"), w])
    if "F7" in nm and "F8" in nm:
        feats.append(X[:, nm.index("F7"), w] - X[:, nm.index("F8"), w])
    O = np.concatenate(feats, 1)
    if eog:
        O = np.c_[O, (np.ptp(X[:, nm.index(eog[0])], 1) > 75).astype(float)]
    pos = {t: i for i, t in enumerate(tri)}
    keep = np.array([t in pos for t in trials_keep])
    return O[[pos[t] for t in trials_keep[keep]]], keep, (X[:, nm.index(eog[0]), w].mean(1) if eog else None), tri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--n_null", type=int, default=12)
    ap.add_argument("--out", default="results/rung0_ocular")
    a = ap.parse_args()
    Q = np.load(QUANT)
    s, i = a.sub, a.sub - 1
    Ep = np.load(E / f"sub-{s:03d}_epochs.npz")
    good = ~Ep["bad"]
    chl = [str(c) for c in Ep["ch"]]
    Xa, tri, times = Ep["X"][good].astype(np.float64), Ep["trial"][good], Ep["times"]
    O, keep, eog_mean, tri_o = ocular(s, tri)
    Xa, tri = Xa[keep], tri[keep]
    win = (times >= 0.15) & (times <= 0.60)
    n = len(tri)
    full = Xa[:, :, win].reshape(n, -1)
    inner = [chl.index(c) for c in chl if c not in RING]
    interior = Xa[:, inner][:, :, win].reshape(n, -1)
    rng = np.random.default_rng(500 + s)
    nulls = rng.choice([j for j in range(Q["choice"].shape[0]) if j != i], a.n_null, replace=False)
    res = {"n": n, "n_ocular_features": int(O.shape[1]), "n_interior_channels": len(inner)}
    for tag in LATS:
        y = latent(Q, i, tag, tri)
        ys = [latent(Q, j, tag, tri) for j in nulls]
        row = {}
        for arm, F, Oreg in (("full", full, None), ("interior_eogreg", interior, O), ("eog_only", O, None)):
            real = cv(F, y, Oreg)
            nl = [cv(F, yn, Oreg) for yn in ys]
            row[arm] = {"r": real, "null": float(np.mean(nl)), "excess": real - float(np.mean(nl))}
        res[tag] = row
    # ocular topography: correlation of each channel's mean 150-600 ms amplitude (ICA-cleaned) with pre-ICA EOG
    if eog_mean is not None:
        pos = {t: k for k, t in enumerate(tri_o)}
        e = eog_mean[[pos[t] for t in tri]]
        res["ocular_map_ch"] = chl
        res["ocular_map"] = [corr(Xa[:, c, win].mean(1), e) for c in range(len(chl))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"sub-{s:03d}.json").write_text(json.dumps(res))
    print(s, {t: {k: round(v["excess"], 3) for k, v in res[t].items()} for t in LATS}, flush=True)


if __name__ == "__main__":
    main()
