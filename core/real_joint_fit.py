"""Real-data verification of EEG's predicted worth (ds003458, 23 participants, Rescorla-Wagner with decay).

For each participant, the read-out y_t is a decoder's prediction of the trial-wise signed prediction error from feedback
EEG: eegnet = EEGNet trained on the other 22 participants (the participant's own labels are never used; not circular);
ens = the science-ensemble features with in-participant cross-validated ridge (targets at the behavioural MAP; mild
circularity, reported as such). The joint likelihood is the exact behavioural log-likelihood (models.run) plus the
Gaussian read-out model of exp_exact_joint.py,
    y_t ~ N(rho * q~_t(theta), 1 - rho^2),
q~(theta) = signed PE under theta, residualized on [1, outcome, arm dummies, arm x outcome, t, t^2, t^3] and z-scored over
the participant's EEG trials in the fitting window; y is residualized and z-scored identically. rho = the participant's
null-corrected decoding coupling tanh(d_i) from group_stats.py (--rho subject) or the pooled random-effects value
(--rho pooled). Posteriors are importance-sampled (two-stage Student-t proposal around the MAP, defensive prior mixture;
ESS reported). Only the likelihood changes between conditions: model dynamics always run on all observed trials.

Tests
 A  held-out choice prediction: fit on trials 1-240 (EEG only there), score the posterior-predictive log-likelihood of
    choices 241-480 (one-step-ahead); and the reverse. Behaviour-only vs joint, paired across participants
    (Wilcoxon signed-rank and DerSimonian-Laird random effects).
 B  split-half reliability: odd vs even 40-trial blocks, posterior means, Pearson r across participants,
    Spearman-Brown corrected, bootstrap CI over participants; joint minus behaviour-only.
 C  observed posterior variance ratio (joint / behaviour-only, first-half fit) vs the exact-curve prediction at the
    participant's rho (results/ceiling_RWD_v2/results.json, 240-trial joint).
 Control: y circularly shifted by >= 40 EEG trials (breaks trial-wise correspondence): must give no gain.
Usage: python real_joint_fit.py --decoder eegnet --rho subject --out results/real_verify_eegnet
       python real_joint_fit.py --summarize results/real_verify_eegnet     (recompute C once the curve exists)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from cogmodels import models as M
from group_stats import dersimonian_laird

QUANT = "results/full_36318/behaviour_quantities_flow.npz"
CURVE = "results/ceiling_RWD_v2/results.json"
T = 480
BLOCK = 40


def regressors(c, r, tri):
    oh = np.eye(3)[c[tri]]
    t = tri / 480.0
    return np.column_stack([np.ones(len(tri)), r[tri], oh, oh * r[tri, None], t, t ** 2, t ** 3])


def projector(C, dev):
    C = torch.tensor(C, dtype=torch.float32, device=dev)
    G = C.T @ C + 1e-6 * torch.eye(C.shape[1], device=dev)
    return C @ torch.linalg.solve(G, C.T)


def resid_z(q, P):
    """q (M, n) rows; remove projection on the regressors, z-score each row."""
    q = q - q @ P.T
    q = q - q.mean(1, keepdim=True)
    return q / (q.std(1, keepdim=True) + 1e-6)


class Subject:
    def __init__(self, i, Q, dev, model, probs, y_all, tri):
        self.i, self.dev, self.model, self.probs = i, dev, model, probs
        self.c = Q["choice"][i].astype(int); self.r = Q["reward"][i].astype(float); self.miss = Q["missing"][i].astype(bool)
        self.ch = torch.tensor(self.c, device=dev)[None]
        self.rw = torch.tensor(self.r, dtype=torch.float32, device=dev)[None]
        self.ms = torch.tensor(self.miss, device=dev)[None]
        self.y_all, self.tri = y_all, tri

    def run(self, z, count):
        """Behaviour log-likelihood on the trials in `count` (dynamics on all trials) and the PE series."""
        Mn = z.shape[0]
        o = M.run(self.model, z, self.probs, self.ch.expand(Mn, -1), self.rw.expand(Mn, -1), self.ms.expand(Mn, -1),
                  record=True, count=count.expand(Mn, -1))
        return o["loglik"], o["q"]["pe"]

    def readout(self, window, shift=0):
        """y and projector for the EEG trials inside a trial window (bool mask over T)."""
        keep = window[self.tri]
        tri = self.tri[keep]
        y = self.y_all[keep]
        if shift:
            y = np.roll(y, shift)
        C = regressors(self.c, self.r, tri)
        P = projector(C, self.dev)
        yt = resid_z(torch.tensor(y, dtype=torch.float32, device=self.dev)[None], P)[0]
        return torch.tensor(tri, device=self.dev), yt, P


def logprior(z, mu, sd):
    return (-0.5 * ((z - mu) / sd) ** 2).sum(-1)


def is_fit(S, count, center, rho, ro, mu, sd, m1, m2, chunk, test=None, defensive=(0.2, 0.1)):
    """Two-stage importance sampling. Returns posterior mean, var, ESS, and log p(test | fit) if `test` given."""
    dev = S.dev
    D = len(mu)

    def target(z):
        lls, lts = [], []
        for k in range(0, len(z), chunk):
            zk = z[k:k + chunk]
            lb, pe = S.run(zk, count)
            l = lb
            if ro is not None and rho > 0:
                tri, yt, P = ro
                q = resid_z(pe[:, tri], P)
                l = l + (-0.5 * (yt[None] - rho * q) ** 2 / (1 - rho ** 2)).sum(1)
            lls.append(l)
            if test is not None:
                lts.append(S.run(zk, test)[0])
        return torch.cat(lls), (torch.cat(lts) if test is not None else None)

    def stage(cen, scale, m, wdef):
        prop = torch.distributions.StudentT(5.0, cen.expand(m, D), scale.expand(m, D))
        pri = torch.distributions.Normal(mu.expand(m, D), sd.expand(m, D))
        pick = torch.rand(m, 1, device=dev) < wdef
        z = torch.where(pick, pri.sample(), prop.sample())
        lq = torch.logaddexp(math.log(1 - wdef) + prop.log_prob(z).sum(-1), math.log(wdef) + pri.log_prob(z).sum(-1))
        lt, ltest = target(z)
        lw = torch.nan_to_num(lt + logprior(z, mu, sd) - lq, nan=-float("inf"))
        w = torch.softmax(lw, 0)
        mean = (w[:, None] * z).sum(0)
        var = (w[:, None] * (z - mean) ** 2).sum(0)
        ess = float(1 / (w ** 2).sum()) / m
        lpred = float(torch.logsumexp(torch.log(w + 1e-300) + ltest, 0)) if ltest is not None else None
        return mean, var, ess, lpred

    m_, v_, _, _ = stage(center, 0.6 * sd, m1, defensive[0])
    mean, var, ess, lpred = stage(m_, 2.0 * v_.sqrt() + 1e-3, m2, defensive[1])
    return mean, var, ess, lpred


def decoder_readouts(dec, Q, Zmap, model, probs):
    """y per participant (on its EEG trials) and its trial indices."""
    out = []
    for i in range(Q["choice"].shape[0]):
        fz = np.load(Path(f"results/features/{dec}") / f"sub-{i + 1:03d}.npz")
        tri = fz["tri"].astype(int)
        if fz["F"].shape[1] == 1:                         # trained decoder: its prediction is the read-out
            y = fz["F"][:, 0].astype(float)
        else:                                             # feature set: in-participant cross-validated prediction
            from group_stats import cv_predict_gpu       # identical to belief_bottleneck.cv_predict(mode='pca'), on GPU
            with torch.no_grad():
                pe = M.run(model, torch.tensor(Zmap[i][None]).float(), probs, torch.tensor(Q["choice"][i])[None],
                           torch.tensor(Q["reward"][i])[None], torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy()
            C = regressors(Q["choice"][i].astype(int), Q["reward"][i].astype(float), tri)
            v = pe[tri].astype(float); v = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; v = (v - v.mean()) / (v.std() + 1e-9)
            P = cv_predict_gpu(fz["F"].astype(np.float64), v[:, None], 64, colw=fz["colw"].astype(np.float64))
            y = np.asarray(P.cpu().numpy() if hasattr(P, "cpu") else P)[:, 0]
        out.append((y, tri))
    return out


def reliability(a, b, rng, nboot=2000):
    r = np.corrcoef(a, b)[0, 1]
    bs = []
    n = len(a)
    for _ in range(nboot):
        k = rng.integers(n, size=n)
        bs.append(np.corrcoef(a[k], b[k])[0, 1])
    sb = lambda x: 2 * x / (1 + x)
    return float(r), float(sb(r)), np.array(bs)


def summarize_C(R, names):
    p = Path(CURVE)
    if not p.exists():
        return {"status": f"pending: {CURVE} not found"}
    C = json.loads(p.read_text())
    rhos = sorted(float(k) for k in C["joint"])
    key = lambda r: next(k for k in C["joint"] if abs(float(k) - r) < 1e-9)
    out = {}
    for j, nm in enumerate(names):
        c = np.array([C["joint"][key(r)]["exact"][nm]["contraction"] for r in rhos])
        pred_ratio = lambda rho: (1 - np.interp(rho, rhos, c)) / (1 - c[0])          # var(joint)/var(behaviour)
        obs, pred = [], []
        for s in R["subjects"].values():
            obs.append(s["h1_joint"]["var"][j] / s["h1_beh"]["var"][j]); pred.append(pred_ratio(max(s["rho"], 0.0)))
        obs, pred = np.array(obs), np.array(pred)
        out[nm] = {"observed_ratio_mean": float(obs.mean()), "observed_ratio_median": float(np.median(obs)),
                   "predicted_ratio_mean": float(pred.mean()), "corr_obs_pred": float(np.corrcoef(obs, pred)[0, 1]),
                   "wilcoxon_obs_lt_1_p": float(stats.wilcoxon(obs - 1, alternative="less").pvalue)}
    return out


def summarize(R, names, rng):
    S = R["subjects"]
    out = {}
    # A: held-out prediction, nats per test trial, joint minus behaviour-only (and shuffled control)
    for tag in ("A_h1_to_h2", "A_h2_to_h1"):
        for cond in ("joint", "shuffled"):
            d = np.array([(s[tag][cond] - s[tag]["beh"]) / s[tag]["n_test"] for s in S.values()])
            out[f"{tag}_{cond}"] = {"mean_gain_nats_per_trial": float(d.mean()), "n_positive": int((d > 0).sum()),
                                     "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                                     "meta": dersimonian_laird(d, np.full(len(d), 1e-8))}   # between-person spread enters via tau^2
    # B: split-half reliability
    for j, nm in enumerate(names):
        res = {}
        for cond in ("beh", "joint"):
            a = np.array([s["odd_" + cond]["mean"][j] for s in S.values()]); b = np.array([s["even_" + cond]["mean"][j] for s in S.values()])
            r, rsb, bs = reliability(a, b, np.random.default_rng(0))
            res[cond] = {"r": r, "spearman_brown": rsb, "ci95": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]}
        a0 = np.array([s["odd_beh"]["mean"][j] for s in S.values()]); b0 = np.array([s["even_beh"]["mean"][j] for s in S.values()])
        a1 = np.array([s["odd_joint"]["mean"][j] for s in S.values()]); b1 = np.array([s["even_joint"]["mean"][j] for s in S.values()])
        diff = []
        g = np.random.default_rng(1)
        for _ in range(2000):
            k = g.integers(len(a0), size=len(a0))
            diff.append(np.corrcoef(a1[k], b1[k])[0, 1] - np.corrcoef(a0[k], b0[k])[0, 1])
        res["joint_minus_beh"] = {"r_diff": res["joint"]["r"] - res["beh"]["r"],
                                  "ci95": [float(np.nanquantile(diff, .025)), float(np.nanquantile(diff, .975))]}
        out[f"B_{nm}"] = res
    out["C"] = summarize_C(R, names)
    out["ess_min"] = float(min(min(s["ess"].values()) for s in S.values()))
    out["ess_median"] = float(np.median([np.median(list(s["ess"].values())) for s in S.values()]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="eegnet")
    ap.add_argument("--rho", default="subject", choices=["subject", "pooled"])
    ap.add_argument("--rho_from", default=None, help="group_stats decoder name giving rho (default: --decoder)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subs", default=None, help="comma list of 1-based participants (smoke test)")
    ap.add_argument("--m1", type=int, default=8000)
    ap.add_argument("--m2", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--summarize", default=None, help="recompute the summary of an existing results dir")
    a = ap.parse_args()
    model = M.EXTRA_MODELS["RWD"]
    names = model.param_names
    if a.summarize:
        p = Path(a.summarize) / "results.json"
        R = json.loads(p.read_text()); R["summary"] = summarize(R, names, None)
        p.write_text(json.dumps(R, indent=1)); print(json.dumps(R["summary"], indent=1)); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    out = Path(a.out or f"results/real_verify_{a.decoder}"); out.mkdir(parents=True, exist_ok=True)
    Q = np.load(QUANT)
    Zmap = np.load("results/map_RWD.npy")
    probs = M.reward_schedule(T)
    G = json.loads(Path("results/group_stats/results.json").read_text())["decoders"][a.rho_from or a.decoder]
    rho_sub = np.clip(np.tanh(np.array(G["d"])), 0.0, 0.95)
    rho_pool = float(np.clip(G["rho_eff_pooled"], 0.0, 0.95))
    RO = decoder_readouts(a.decoder, Q, Zmap, model, probs)
    mu = torch.tensor(model.prior_mean, device=dev); sd = torch.tensor(model.prior_sd, device=dev)
    probs = probs.to(dev)
    tt = np.arange(T)
    h1 = tt < 240; h2 = ~h1
    blk = (tt // BLOCK) % 2
    odd, even = blk == 0, blk == 1
    mask = lambda b: torch.tensor(b.astype(np.float32), device=dev)[None]
    subs = [int(x) - 1 for x in a.subs.split(",")] if a.subs else list(range(Q["choice"].shape[0]))
    R = {"decoder": a.decoder, "rho_mode": a.rho, "m1": a.m1, "m2": a.m2, "subjects": {}}
    rng = np.random.default_rng(7)
    for i in subs:
        y, tri = RO[i]
        S = Subject(i, Q, dev, model, probs, y, tri)
        rho = float(rho_sub[i] if a.rho == "subject" else rho_pool)
        cen = torch.tensor(Zmap[i], dtype=torch.float32, device=dev)
        rec = {"rho": rho, "n_eeg": int(len(tri)), "ess": {}}
        n_eeg_h1 = int(h1[tri].sum())
        shift = int(rng.integers(40, max(41, n_eeg_h1 - 40)))
        with torch.no_grad():
            for fitw, testw, tag in ((h1, h2, "A_h1_to_h2"), (h2, h1, "A_h2_to_h1")):
                ro = S.readout(fitw)
                ro_sh = S.readout(fitw, shift=shift)
                res = {"n_test": int((testw & ~S.miss).sum())}
                for cond, r_, ro_ in (("beh", 0.0, None), ("joint", rho, ro), ("shuffled", rho, ro_sh)):
                    m, v, ess, lp = is_fit(S, mask(fitw), cen, r_, ro_, mu, sd, a.m1, a.m2, a.chunk, test=mask(testw))
                    res[cond] = lp; rec["ess"][f"{tag}_{cond}"] = ess
                    if tag == "A_h1_to_h2" and cond != "shuffled":
                        rec[f"h1_{cond}"] = {"mean": m.tolist(), "var": v.tolist()}
                    if tag == "A_h1_to_h2" and cond == "shuffled":
                        rec["h1_shuffled"] = {"mean": m.tolist(), "var": v.tolist()}
                rec[tag] = res
            for win, wt in ((odd, "odd"), (even, "even")):
                ro = S.readout(win)
                for cond, r_, ro_ in (("beh", 0.0, None), ("joint", rho, ro)):
                    m, v, ess, _ = is_fit(S, mask(win), cen, r_, ro_, mu, sd, a.m1, a.m2, a.chunk)
                    rec[f"{wt}_{cond}"] = {"mean": m.tolist(), "var": v.tolist()}; rec["ess"][f"{wt}_{cond}"] = ess
        R["subjects"][str(i + 1)] = rec
        A1 = rec["A_h1_to_h2"]
        print(f"sub {i + 1} rho={rho:.3f} A(h1->h2) joint-beh {A1['joint'] - A1['beh']:+.2f} nats, shuffled-beh "
              f"{A1['shuffled'] - A1['beh']:+.2f} | var ratio alpha {rec['h1_joint']['var'][0] / rec['h1_beh']['var'][0]:.2f} "
              f"(shuffled {rec['h1_shuffled']['var'][0] / rec['h1_beh']['var'][0]:.2f}) | ESS min {min(rec['ess'].values()):.3f}", flush=True)
        (out / "results.json").write_text(json.dumps(R, indent=1))
    if len(subs) >= 5:
        R["summary"] = summarize(R, names, rng)
        (out / "results.json").write_text(json.dumps(R, indent=1))
        print(json.dumps(R["summary"], indent=1))


if __name__ == "__main__":
    main()
