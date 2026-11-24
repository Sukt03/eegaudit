"""Belief bottleneck curves: how much of single-trial feedback EEG does a belief decoder use? (rungs 0-2 of the usefulness protocol)

Inspired by bottleneck analyses of neural decoding (Mayo et al., NeurIPS 2024): bottleneck curves with a floor and ceiling, applied to science-derived
feature families of single-trial EEG and to the belief state of a Bayes-adaptive learner.

Targets (belief state, per trial; KG-BA flow-posterior latents): pe, abs_pe, sd (chosen-arm uncertainty), kg, logp_rep.
Each target is residualized on outcome, arm, outcome x arm and cubic time, then z-scored within participant.

Families (all features fit inside 10 contiguous-block CV folds):
  raw      64 ch x 150-600 ms ERP                       (reference)
  csd      surface Laplacian ERP                         (physics: undoes volume conduction)
  ged      in-fold generalized-eigen spatial filters     (physics: source-like directions)
  spec     log band power delta/theta/alpha/beta         (maths: Fourier)
  riem     Riemannian tangent-space covariance           (maths: geometry of channel covariance)
  canon    canonical components FRN/RewP, P300, theta, delta at literature sites (biology)
  eye      frontopolar channels Fp1/Fp2/AF7/AF8/AFz ERP  (artefact control)
  ens      raw-free ensemble: csd + spec + riem + canon stacked (each family block-standardized)

Curves:
  (a) unsupervised bottleneck: PCA-L of the family features (L = 1,2,4,8,16,32,64), ridge to each target
  (b) supervised multi-target bottleneck: reduced-rank ridge to all 5 targets jointly, rank L = 1..5
Score: cross-validated r minus mean session-permutation null r (same EEG, other participants' latents), and
Gaussian-copula MI in bits per trial. Shapley shares of the science families on the full-size ensemble.
Haufe forward patterns (raw family, L=full) are saved for scalp maps.
Usage: python belief_bottleneck.py --subs 1,2,3 --n_null 8 --out results/bottleneck
"""
import argparse
import itertools
import json
from math import factorial
from pathlib import Path

import numpy as np
from scipy import stats

E = Path("results/eeg_measures")
QUANT = "results/full_36318/behaviour_quantities_flow.npz"
TARGETS = ["pe", "abs_pe", "sd", "kg", "logp_rep"]
EYE = ["Fp1", "Fp2", "AF7", "AF8", "AFz"]
CANON = {"rewp": (["FCz", "Fz", "Cz"], 0.25, 0.35), "p300": (["Pz", "CPz", "Cz"], 0.30, 0.50)}
SCIENCE = ["csd", "spec", "riem", "canon"]
PLAYERS = SCIENCE + ["eye"]
PCA_L = [1, 2, 4, 8, 16, 32, 64]
LAM = 10.0


def resid(v, cols):
    C = np.c_[np.ones(len(v)), np.column_stack(cols)]
    b, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ b


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def gc_mi(x, y):
    cop = lambda v: stats.norm.ppf(stats.rankdata(v) / (len(v) + 1))
    r = np.corrcoef(cop(x), cop(y))[0, 1]
    return float(-0.5 * np.log2(1 - min(r * r, 0.999999)))


def folds(n, k=10):
    return np.minimum((np.arange(n) * k) // n, k - 1)


def bandpow(X, times, lo=0.15, hi=0.60, sf=50.0):
    w = (times >= lo) & (times <= hi)
    seg = X[:, :, w] * np.hanning(w.sum())[None, None]
    sp = np.abs(np.fft.rfft(seg, n=64, axis=2)) ** 2
    f = np.fft.rfftfreq(64, 1 / sf)
    return np.concatenate([np.log10(sp[:, :, (f >= a) & (f <= b)].mean(2) + 1e-12)
                           for a, b in ((1.4, 3.2), (4, 8), (8, 13), (13, 25))], 1)


def csd(X, chans):
    import mne
    info = mne.create_info(list(chans), 50.0, "eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    ep = mne.EpochsArray(X * 1e-6, info, verbose="error")
    return mne.preprocessing.compute_current_source_density(ep, verbose="error").get_data() * 1e6


def riem(X, chans, times):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    sub = [c for c in ["Fz", "FCz", "Cz", "CPz", "Pz", "F3", "F4", "FC1", "FC2", "C3", "C4", "CP1", "CP2", "P3", "P4", "POz"]
           if c in chans]
    idx = [chans.index(c) for c in sub]
    w = (times >= 0.15) & (times <= 0.60)                    # post-feedback only (same window as the ERP families)
    C = Covariances(estimator="oas").fit_transform(X[:, idx][:, :, w])
    return TangentSpace(metric="riemann").fit_transform(C)   # reference = mean over all trials (label-free)


def canon(X, chans, times, BP=None):
    out = []
    for _, (cs, lo, hi) in CANON.items():
        w = (times >= lo) & (times <= hi)
        out += [X[:, chans.index(c), w].mean(1) for c in cs if c in chans]
    bp = bandpow(X, times) if BP is None else BP.transpose(0, 2, 1).reshape(len(BP), -1)   # (n, 4*ch) band-major
    nch = len(chans)
    for band, cs in ((1, ["FCz", "Fz"]), (0, ["Pz", "FCz"])):  # theta, delta
        out += [bp[:, band * nch + chans.index(c)] for c in cs if c in chans]
    return np.column_stack(out)


def family_features(X, chans, times, BP=None):
    """BP: true log band power (n, ch, 4) from eeg_v2.py (Morlet on 500 Hz epochs); falls back to a short FFT."""
    win = (times >= 0.15) & (times <= 0.60)
    n = len(X)
    Fam = {"raw": X[:, :, win].reshape(n, -1)}
    Fam["csd"] = csd(X, chans)[:, :, win].reshape(n, -1)
    Fam["spec"] = bandpow(X, times) if BP is None else BP.reshape(len(BP), -1)
    Fam["riem"] = riem(X, chans, times)
    Fam["canon"] = canon(X, chans, times, BP)
    eidx = [chans.index(c) for c in EYE if c in chans]
    Fam["eye"] = X[:, eidx][:, :, win].reshape(n, -1)
    Fam["pre"] = X[:, :, times < 0].reshape(n, -1)            # pre-feedback control: expected value exists before feedback
    return Fam


def zs(Ftr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    return (Ftr - mu) / sd, (Fte - mu) / sd


def ged_filters(Xtr, Ytr, times, k=4, reg=0.01):
    from scipy.linalg import eigh
    w = (times >= 0.15) & (times <= 0.60)
    Z = Xtr[:, :, w]
    Sig = np.einsum("ict,im->mct", Z, (Ytr - Ytr.mean(0)) / (Ytr.std(0) + 1e-9)) / len(Ytr)
    S = sum(s @ s.T for s in Sig)
    R = np.einsum("ict,jct->ij", Z.transpose(1, 0, 2), Z.transpose(1, 0, 2)) / (Z.shape[0] * Z.shape[2])
    R = R + reg * np.trace(R) / len(R) * np.eye(len(R))
    ev, V = eigh(S, R)
    return V[:, ::-1][:, :k]


def cv_predict(F, Y, mode, L, X=None, times=None, colw=None):
    """Cross-validated predictions of all targets Y (n, m) from features F under a bottleneck.
    mode 'pca': unsupervised PCA-L then per-target ridge. mode 'rrr': reduced-rank ridge of rank L on PCA-64 features."""
    n, m = Y.shape
    f = folds(n)
    P = np.zeros_like(Y)
    for b in range(10):
        tr, te = f != b, f == b
        if F is None:                                        # GED family: filters learned in-fold from targets
            V = ged_filters(X[tr], Y[tr], times)
            w = (times >= 0.15) & (times <= 0.60)
            Fc = np.einsum("ict,ck->ikt", X, V)[:, :, w].reshape(n, -1)
            A, B = zs(Fc[tr], Fc[te])
        else:
            A, B = zs(F[tr], F[te])
            if colw is not None:                             # block weights applied after in-fold scaling
                A, B = A * colw, B * colw
        U, S, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
        k = min(L if mode == "pca" else 64, Vt.shape[0])
        Vk = Vt[:k].T
        Atr, Ate = A @ Vk, B @ Vk
        ym = Y[tr].mean(0)
        W = np.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * np.eye(k), Atr.T @ (Y[tr] - ym))
        if mode == "rrr":
            Fit = Atr @ W
            _, _, Vr = np.linalg.svd(Fit, full_matrices=False)
            Vr = Vr[:L].T
            W = W @ Vr @ Vr.T
        P[te] = Ate @ W                                      # no fold-mean term (avoids negative CV bias under drift)
    return P


def targets(Q, j, tri, cols):
    out = []
    for t in TARGETS:
        key = {"pe": "pe", "abs_pe": "pe", "sd": "sd", "kg": "kg", "logp_rep": "logp_rep"}[t]
        v = Q[f"BAMDP_{key}"][j][tri].astype(float)
        v = np.abs(v) if t == "abs_pe" else v
        y = resid(v, cols)
        out.append((y - y.mean()) / (y.std() + 1e-9))
    return np.column_stack(out)


def score(P, Y):
    return [corr(P[:, k], Y[:, k]) for k in range(Y.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(str(i) for i in range(1, 24)))
    ap.add_argument("--n_null", type=int, default=12)
    ap.add_argument("--out", default="results/bottleneck")
    a = ap.parse_args()
    Q = np.load(QUANT)
    nS = Q["choice"].shape[0]
    common = None
    for s in range(1, 24):
        c = [str(x) for x in np.load(E / f"sub-{s:03d}_epochs.npz")["ch"]]
        common = c if common is None else [x for x in common if x in c]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.subs.split(",")]:
        i = s - 1
        Ep = np.load(E / f"sub-{s:03d}_epochs.npz")
        keep = ~Ep["bad"]
        chl = [str(x) for x in Ep["ch"]]
        X = Ep["X"][keep][:, [chl.index(c) for c in common]].astype(np.float64)
        tri, times = Ep["trial"][keep], Ep["times"]
        ch, rw = Q["choice"][i], Q["reward"][i]
        cols = [rw[tri]] + [(ch[tri] == k).astype(float) for k in range(3)] + \
               [(ch[tri] == k).astype(float) * rw[tri] for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
        Y = targets(Q, i, tri, cols)
        rng = np.random.default_rng(1000 + s)
        null_ids = rng.choice([j for j in range(nS) if j != i], a.n_null, replace=False)
        def own_cols(j):                                     # null latents residualized on THEIR OWN session's outcomes/choices
            cj, rj = Q["choice"][j], Q["reward"][j]
            return [rj[tri]] + [(cj[tri] == k).astype(float) for k in range(3)] + \
                   [(cj[tri] == k).astype(float) * rj[tri] for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
        Ynull = [targets(Q, j, tri, own_cols(j)) for j in null_ids]
        BPs = Ep["BP"][keep][:, [chl.index(c) for c in common]] if "BP" in Ep.files else None
        Fam = family_features(X, common, times, BPs)
        stack = lambda names: (np.concatenate([Fam[k] for k in names], 1),
                               np.concatenate([np.full(Fam[k].shape[1], np.sqrt(sum(Fam[x].shape[1] for x in names) / (len(names) * Fam[k].shape[1]))) for k in names]))
        Fam["ens"], W_ENS = stack(SCIENCE)
        res = {"n": int(len(tri)), "targets": TARGETS, "pca": {}, "rrr": {}, "mi_full": {}, "shapley": {}}

        def run(F, mode, L, fam=None, colw=None):
            Fx = None if fam == "ged" else F
            colw = W_ENS if fam == "ens" else colw
            P = cv_predict(Fx, Y, mode, L, X, times, colw)
            real = score(P, Y)
            NP = [cv_predict(Fx, Yn, mode, L, X, times, colw) for Yn in Ynull]
            nl = np.array([score(Pn, Yn) for Pn, Yn in zip(NP, Ynull)])
            mi_null = np.array([[gc_mi(Pn[:, k], Yn[:, k]) for k in range(len(TARGETS))] for Pn, Yn in zip(NP, Ynull)])
            return real, nl.mean(0).tolist(), P, nl.std(0).tolist(), mi_null.mean(0)

        for fam in ["raw", "csd", "ged", "spec", "riem", "canon", "eye", "pre", "ens"]:
            F = Fam.get(fam)
            res["pca"][fam] = {}
            for L in (PCA_L if fam != "ged" else [4, 16, 64]):
                real, nul, P, nsd, mnull = run(F, "pca", L, fam)
                res["pca"][fam][L] = {"r": real, "null": nul, "null_sd": nsd, "excess": (np.array(real) - nul).tolist()}
                if L == 64:
                    mi = np.array([gc_mi(P[:, k], Y[:, k]) for k in range(len(TARGETS))])
                    res["mi_full"][fam] = {"bits": mi.tolist(), "bits_null": mnull.tolist(), "bits_excess": (mi - mnull).tolist()}
            res["rrr"][fam] = {}
            if fam in ("raw", "ens", "canon", "eye"):
                for L in range(1, len(TARGETS) + 1):
                    real, nul, _, _, _ = run(F, "rrr", L, fam)
                    res["rrr"][fam][L] = {"r": real, "null": nul, "excess": (np.array(real) - nul).tolist()}
        # Shapley shares (null-corrected excess r, PCA-64 on each stacked subset); eye family is a player
        val = {}
        for k in range(len(PLAYERS) + 1):
            for sub in itertools.combinations(PLAYERS, k):
                if not sub:
                    val[sub] = np.zeros(len(TARGETS)); continue
                F, cw = stack(list(sub))
                real = np.array(score(cv_predict(F, Y, "pca", 64, colw=cw), Y))
                nul = np.mean([score(cv_predict(F, Yn, "pca", 64, colw=cw), Yn) for Yn in Ynull[:6]], 0)
                val[sub] = real - nul
        nF = len(PLAYERS)
        for fam in PLAYERS:
            phi = np.zeros(len(TARGETS))
            others = [x for x in PLAYERS if x != fam]
            for k in range(nF):
                for sub in itertools.combinations(others, k):
                    w = factorial(k) * factorial(nF - k - 1) / factorial(nF)
                    withf = tuple(x for x in PLAYERS if x in sub or x == fam)
                    phi += w * (val[withf] - val[tuple(x for x in PLAYERS if x in sub)])
            res["shapley"][fam] = phi.tolist()
        res["shapley_total"] = val[tuple(PLAYERS)].tolist()
        # Haufe forward pattern (raw ERP, ridge on all trials) averaged over the window -> scalp map per target
        win = (times >= 0.15) & (times <= 0.60)
        A = (Fam["raw"] - Fam["raw"].mean(0))
        W = np.linalg.solve(A.T @ A + LAM * len(A) * np.eye(A.shape[1]), A.T @ Y)
        Pat = (A.T @ (A @ W)) / len(A)
        res["haufe_ch"] = common
        tw = times[win]
        res["haufe"] = Pat.reshape(len(common), win.sum(), len(TARGETS))[:, (tw >= 0.25) & (tw <= 0.35)].mean(1).T.tolist()
        Path(a.out, f"sub-{s:03d}.json").write_text(json.dumps(res))
        ex = {f: round(res["pca"][f][64 if f != "ged" else 64]["excess"][0], 3) for f in res["pca"]}
        print(s, "PE excess @64:", ex, flush=True)


if __name__ == "__main__":
    main()
