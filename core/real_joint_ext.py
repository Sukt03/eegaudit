"""Diagnostics for the real-data verification (real_joint_fit.py): why does the EEG term contract the posterior without
improving held-out prediction?

--part real   (per real participant; decoder read-out as in real_joint_fit.py)
   eeg_only   posterior from the read-out likelihood + prior alone (no behaviour), full session; compared with the
              behaviour-only posterior (full session)
   linked     6-parameter model: behaviour under (alpha, d, beta, kappa); read-out q~ under separate (alpha_n, d_n)
              (same priors); posterior of alpha_n - alpha and d_n - d; held-out prediction (A, h1 -> h2) and odd/even
              reliability (B) of the behavioural parameters under this model
   ar1        joint model with AR(1)-prewhitened read-out errors (phi from consecutive-trial residuals at the MAP);
              A (h1 -> h2) and C (first-half variance ratio)
--part twin   simulated twins: k per participant, theta drawn from the participant's behaviour-only posterior, choices
              simulated on the real schedule, the participant's EEG trial set and measured rho, read-out generated from
              the correctly specified model; the identical A (h1 -> h2), B (odd/even) and C pipeline -> expected effects
              and power under correct specification.
Usage: python real_joint_ext.py --part real --decoder eegnet --out results/real_ext_eegnet
       python real_joint_ext.py --part twin --decoder eegnet --k 4 --subs 1-12 --out results/real_twin_eegnet_a
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
from real_joint_fit import QUANT, T, BLOCK, regressors, projector, resid_z, decoder_readouts, summarize_C

NAMES = ["logit_alpha", "logit_d", "log_beta", "kappa"]


class Data:
    """One participant's (real or simulated) behaviour + read-out."""

    def __init__(self, c, r, miss, y, tri, dev, model, probs):
        self.c, self.r, self.miss, self.y_all, self.tri = c.astype(int), r.astype(float), miss.astype(bool), y, tri
        self.dev, self.model, self.probs = dev, model, probs
        self.ch = torch.tensor(self.c, device=dev)[None]
        self.rw = torch.tensor(self.r, dtype=torch.float32, device=dev)[None]
        self.ms = torch.tensor(self.miss, device=dev)[None]

    def run(self, z, count):
        Mn = z.shape[0]
        o = M.run(self.model, z, self.probs, self.ch.expand(Mn, -1), self.rw.expand(Mn, -1), self.ms.expand(Mn, -1),
                  record=True, count=count.expand(Mn, -1))
        return o["loglik"], o["q"]["pe"]

    def readout(self, window, shift=0, phi=0.0):
        keep = window[self.tri]
        tri = self.tri[keep]
        y = np.roll(self.y_all[keep], shift) if shift else self.y_all[keep]
        P = projector(regressors(self.c, self.r, tri), self.dev)
        yt = resid_z(torch.tensor(y, dtype=torch.float32, device=self.dev)[None], P)[0]
        cons = torch.tensor(np.r_[False, np.diff(tri) == 1], device=self.dev)
        return {"tri": torch.tensor(tri, device=self.dev), "y": yt, "P": P, "cons": cons, "phi": phi}


def readout_ll(pe, ro, rho):
    q = resid_z(pe[:, ro["tri"]], ro["P"])
    e = ro["y"][None] - rho * q
    s2 = 1 - rho ** 2
    phi = ro["phi"]
    if not phi:
        return (-0.5 * e ** 2 / s2).sum(1)
    innov = torch.where(ro["cons"][None], e - phi * torch.roll(e, 1, 1), e)
    var = torch.where(ro["cons"][None], torch.full_like(e, s2 * (1 - phi ** 2)), torch.full_like(e, s2))
    return (-0.5 * innov ** 2 / var - 0.5 * torch.log(var)).sum(1)


def is_generic(target, center, scale0, mu, sd, m1, m2, dev, wdef=(0.2, 0.1), keep=False):
    """Two-stage defensive IS. target(z) -> (log-lik on fit data, log-lik on test data or None)."""
    D = len(mu)

    def stage(cen, scale, m, wd):
        prop = torch.distributions.StudentT(5.0, cen.expand(m, D), scale.expand(m, D))
        pri = torch.distributions.Normal(mu.expand(m, D), sd.expand(m, D))
        z = torch.where(torch.rand(m, 1, device=dev) < wd, pri.sample(), prop.sample())
        lq = torch.logaddexp(math.log(1 - wd) + prop.log_prob(z).sum(-1), math.log(wd) + pri.log_prob(z).sum(-1))
        lt, ltest = target(z)
        lw = torch.nan_to_num(lt + (-0.5 * ((z - mu) / sd) ** 2).sum(-1) - lq, nan=-float("inf"))
        w = torch.softmax(lw, 0)
        mean = (w[:, None] * z).sum(0); var = (w[:, None] * (z - mean) ** 2).sum(0)
        lpred = float(torch.logsumexp(torch.log(w + 1e-300) + ltest, 0)) if ltest is not None else None
        return z, w, mean, var, float(1 / (w ** 2).sum()) / m, lpred

    _, _, m_, v_, _, _ = stage(center, scale0, m1, wdef[0])
    z, w, mean, var, ess, lpred = stage(m_, 2.0 * v_.sqrt() + 1e-3, m2, wdef[1])
    out = {"mean": mean, "var": var, "ess": ess, "lpred": lpred}
    if keep:
        out["z"], out["w"] = z, w
    return out


def chunked(fn, z, chunk):
    a, b = [], []
    for k in range(0, len(z), chunk):
        x, y = fn(z[k:k + chunk]); a.append(x); b.append(y)
    return torch.cat(a), (torch.cat(b) if b[0] is not None else None)


def fits_one(Dt, rho, cen, mu4, sd4, masks, m1, m2, chunk, dev, parts, phi=0.0):
    """Returns dict of results for one participant (or twin)."""
    h1, h2, odd, even, full, zero = masks
    res = {}

    def std(win, test, ro=None, ro_rho=0.0):
        def f(z):
            def g(zk):
                lb, pe = Dt.run(zk, win)
                l = lb + (readout_ll(pe, ro, ro_rho) if ro is not None and ro_rho > 0 else 0)
                return l, (Dt.run(zk, test)[0] if test is not None else None)
            return chunked(g, z, chunk)
        return is_generic(f, cen, 0.6 * sd4, mu4, sd4, m1, m2, dev)

    def pack(o):
        return {"mean": o["mean"].tolist(), "var": o["var"].tolist(), "ess": o["ess"], "lpred": o["lpred"]}

    win_np = lambda m: m[0].cpu().numpy() > 0.5
    if "A" in parts:
        ro = Dt.readout(win_np(h1))
        res["A_beh"] = pack(std(h1, h2))
        res["A_joint"] = pack(std(h1, h2, ro, rho))
        res["n_test"] = int((win_np(h2) & ~Dt.miss).sum())
    if "B" in parts:
        for nm, w in (("odd", odd), ("even", even)):
            res[f"B_{nm}_beh"] = pack(std(w, None))
            res[f"B_{nm}_joint"] = pack(std(w, None, Dt.readout(win_np(w)), rho))
    if "ar1" in parts:
        ro = Dt.readout(win_np(h1), phi=phi)
        res["A_joint_ar1"] = pack(std(h1, h2, ro, rho))
    if "eeg_only" in parts:
        ro = Dt.readout(win_np(full))
        res["full_beh"] = pack(std(full, None))
        res["eeg_only"] = pack(std(zero, None, ro, rho))
    if "linked" in parts or "linkedfull" in parts:
        mu6, sd6 = torch.cat([mu4, mu4[:2]]), torch.cat([sd4, sd4[:2]])
        cen6 = torch.cat([cen, cen[:2]])

        def lk(win, test, ro):
            def f(z):
                def g(zk):
                    lb, _ = Dt.run(zk[:, :4], win)
                    _, pen = Dt.run(torch.cat([zk[:, 4:6], zk[:, 2:4]], 1), zero)
                    return lb + readout_ll(pen, ro, rho), (Dt.run(zk[:, :4], test)[0] if test is not None else None)
                return chunked(g, z, chunk)
            return is_generic(f, cen6, 0.6 * sd6, mu6, sd6, m1, 2 * m2, dev, keep=True)

        o = lk(full, None, Dt.readout(win_np(full)))
        z, w = o["z"], o["w"]
        for j, nm in ((0, "alpha"), (1, "d")):
            dlt = z[:, 4 + j] - z[:, j]
            m = float((w * dlt).sum()); v = float((w * (dlt - m) ** 2).sum())
            res[f"linked_{nm}_n_minus_b"] = {"mean": m, "var": v}
        res["linked_full"] = {"mean": o["mean"].tolist(), "var": o["var"].tolist(), "ess": o["ess"]}
        if "linked" in parts:
            res["A_linked"] = pack(lk(h1, h2, Dt.readout(win_np(h1))))
            for nm, wm in (("odd", odd), ("even", even)):
                res[f"B_{nm}_linked"] = pack(lk(wm, None, Dt.readout(win_np(wm))))
    return res


def ar1_phi(Dt, zmap, rho, dev):
    with torch.no_grad():
        _, pe = Dt.run(zmap[None], torch.ones(1, T, device=dev))
    ro = Dt.readout(np.ones(T, bool))
    q = resid_z(pe[:, ro["tri"]], ro["P"])[0]
    e = (ro["y"] - rho * q).cpu().numpy(); c = ro["cons"].cpu().numpy()
    return float(np.corrcoef(e[1:][c[1:]], e[:-1][c[1:]])[0, 1])


def rel(a, b, seed=0, nboot=2000):
    g = np.random.default_rng(seed); n = len(a)
    bs = [np.corrcoef(a[k], b[k])[0, 1] for k in (g.integers(n, size=n) for _ in range(nboot))]
    r = np.corrcoef(a, b)[0, 1]
    return {"r": float(r), "sb": float(2 * r / (1 + r)), "ci95": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]}


def summarize(R):
    S = list(R["subjects"].values())
    out = {}
    if "A_beh" in S[0]:
        for cond in [c for c in ("joint", "linked", "joint_ar1") if f"A_{c}" in S[0]]:
            d = np.array([(s[f"A_{cond}"]["lpred"] - s["A_beh"]["lpred"]) / s["n_test"] for s in S])
            out[f"A_{cond}"] = {"gain_nats_per_trial": float(d.mean()), "n_pos": int((d > 0).sum()), "n": len(d),
                                "wilcoxon_p": float(stats.wilcoxon(d).pvalue), "ttest_p": float(stats.ttest_1samp(d, 0).pvalue),
                                "dz": float(d.mean() / (d.std(ddof=1) + 1e-12))}
    if "B_odd_beh" in S[0]:
        for cond in [c for c in ("beh", "joint", "linked") if f"B_odd_{c}" in S[0]]:
            out[f"B_{cond}"] = {nm: rel(np.array([s[f"B_odd_{cond}"]["mean"][j] for s in S]),
                                        np.array([s[f"B_even_{cond}"]["mean"][j] for s in S])) for j, nm in enumerate(NAMES)}
    if "eeg_only" in S[0]:
        e = {}
        for j, nm in enumerate(NAMES[:2]):
            a = np.array([s["eeg_only"]["mean"][j] for s in S]); b = np.array([s["full_beh"]["mean"][j] for s in S])
            dd = a - b; g = np.random.default_rng(0)
            bs = [dd[g.integers(len(dd), size=len(dd))].mean() for _ in range(2000)]
            e[nm] = {"corr_eeg_beh": float(np.corrcoef(a, b)[0, 1]), "corr_p": float(stats.pearsonr(a, b)[1]),
                     "mean_diff_eeg_minus_beh": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                     "eeg_only_var_over_prior": float(np.mean([s["eeg_only"]["var"][j] for s in S]) / [1.5, 1.5][j] ** 2)}
        out["eeg_only"] = e
    if "linked_alpha_n_minus_b" in S[0]:
        out["linked_diff"] = {}
        for nm in ("alpha", "d"):
            d = np.array([s[f"linked_{nm}_n_minus_b"]["mean"] for s in S]); v = np.array([s[f"linked_{nm}_n_minus_b"]["var"] for s in S])
            out["linked_diff"][nm] = {"meta": dersimonian_laird(d, v), "n_ci_excl_0": int(np.sum(np.abs(d) > 1.96 * np.sqrt(v)))}
    keys = list(R["subjects"])
    if "_" in keys[0] and "A_joint" in S[0]:              # twins: per replicate set of participants -> power
        reps = sorted({k.split("_")[1] for k in keys})
        pa, pb, gains, rdiff = [], [], [], []
        for rp in reps:
            Ss = [R["subjects"][k] for k in keys if k.endswith("_" + rp)]
            d = np.array([(s["A_joint"]["lpred"] - s["A_beh"]["lpred"]) / s["n_test"] for s in Ss])
            gains.append(d.mean()); pa.append(stats.wilcoxon(d, alternative="greater").pvalue)
            a0 = np.array([s["B_odd_beh"]["mean"][0] for s in Ss]); b0 = np.array([s["B_even_beh"]["mean"][0] for s in Ss])
            a1 = np.array([s["B_odd_joint"]["mean"][0] for s in Ss]); b1 = np.array([s["B_even_joint"]["mean"][0] for s in Ss])
            rdiff.append(np.corrcoef(a1, b1)[0, 1] - np.corrcoef(a0, b0)[0, 1])
        out["twin_power"] = {"n_sets": len(reps), "A_mean_gain_per_set": [float(x) for x in gains],
                             "A_power_wilcoxon_greater_.05": float(np.mean(np.array(pa) < .05)),
                             "B_alpha_r_diff_per_set": [float(x) for x in rdiff]}
    if "A_joint" in S[0] and "rho" in S[0]:
        o = np.array([s["A_joint"]["var"][0] / s["A_beh"]["var"][0] for s in S])
        out["C_alpha_var_ratio"] = {"mean": float(o.mean()), "median": float(np.median(o))}
        if "A_joint_ar1" in S[0]:
            o2 = np.array([s["A_joint_ar1"]["var"][0] / s["A_beh"]["var"][0] for s in S])
            out["C_alpha_var_ratio_ar1"] = {"mean": float(o2.mean()), "median": float(np.median(o2)),
                                            "phi_median": float(np.median([s["phi"] for s in S]))}
    return out


LAMS = (0.0, 0.1, 0.25, 0.5, 1.0)


def temper_one(Dt, rho, cen, mu4, sd4, m1, m2, chunk, dev, shift):
    """Generalized-Bayes tempering: log-lik = behaviour + lambda * read-out. lambda chosen per direction by nested
    held-out choice prediction inside the fitting half (inner fit = its first 180 trials, inner validation = its last
    60), then the whole half is refit with the selected lambda and scored on the other half. Same for shifted EEG."""
    tt = np.arange(T)
    mk = lambda b: torch.tensor(b.astype(np.float32), device=dev)[None]

    def fit(win_b, test_b, lam, sh):
        ro = Dt.readout(win_b, shift=sh) if lam > 0 else None
        win, test = mk(win_b), (mk(test_b) if test_b is not None else None)

        def f(z):
            def g(zk):
                lb, pe = Dt.run(zk, win)
                l = lb + (lam * readout_ll(pe, ro, rho) if ro is not None and rho > 0 else 0)
                return l, (Dt.run(zk, test)[0] if test is not None else None)
            return chunked(g, z, chunk)
        return is_generic(f, cen, 0.6 * sd4, mu4, sd4, m1, m2, dev)

    res = {}
    for tag, (lo, hi), (tlo, thi) in (("h1_to_h2", (0, 240), (240, 480)), ("h2_to_h1", (240, 480), (0, 240))):
        fitw = (tt >= lo) & (tt < hi); testw = (tt >= tlo) & (tt < thi)
        inner_fit = (tt >= lo) & (tt < lo + 180); inner_val = (tt >= lo + 180) & (tt < hi)
        r = {"n_test": int((testw & ~Dt.miss).sum())}
        for cond, sh in (("eeg", 0), ("shifted", shift)):
            val = {lam: fit(inner_fit, inner_val, lam, sh)["lpred"] for lam in LAMS}
            lam_sel = max(LAMS, key=lambda l: val[l])
            o = fit(fitw, testw, lam_sel, sh)
            r[cond] = {"inner_val": {str(k): v for k, v in val.items()}, "lam": lam_sel, "lpred": o["lpred"],
                       "mean": o["mean"].tolist(), "var": o["var"].tolist()}
        r["beh"] = {"lpred": fit(fitw, testw, 0.0, 0)["lpred"]}
        r["lam1"] = {"lpred": fit(fitw, testw, 1.0, 0)["lpred"]}
        res[tag] = r
    lam_b = res["h1_to_h2"]["eeg"]["lam"]                      # odd/even reliability at the lambda chosen on half 1
    blk = (tt // BLOCK) % 2
    for nm, w in (("odd", blk == 0), ("even", blk == 1)):
        res[f"B_{nm}_temper"] = {"mean": fit(w, None, lam_b, 0)["mean"].tolist()}
    res["lam_B"] = lam_b
    return res


def summarize_temper(R):
    S = list(R["subjects"].values())
    out = {}
    for tag in ("h1_to_h2", "h2_to_h1"):
        lam = [s[tag]["eeg"]["lam"] for s in S]; lams = [s[tag]["shifted"]["lam"] for s in S]
        o = {"lambda_counts_eeg": {str(l): lam.count(l) for l in LAMS}, "lambda_counts_shifted": {str(l): lams.count(l) for l in LAMS}}
        for cond, get in (("selected_vs_beh", lambda s: s[tag]["eeg"]["lpred"] - s[tag]["beh"]["lpred"]),
                          ("selected_vs_lam1", lambda s: s[tag]["eeg"]["lpred"] - s[tag]["lam1"]["lpred"]),
                          ("lam1_vs_beh", lambda s: s[tag]["lam1"]["lpred"] - s[tag]["beh"]["lpred"]),
                          ("shifted_selected_vs_beh", lambda s: s[tag]["shifted"]["lpred"] - s[tag]["beh"]["lpred"])):
            d = np.array([get(s) / s[tag]["n_test"] for s in S])
            o[cond] = {"gain_nats_per_trial": float(d.mean()), "n_pos": int((d > 0).sum()), "n_zero": int((d == 0).sum()),
                       "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        out[tag] = o
    out["B_temper"] = {nm: rel(np.array([s["B_odd_temper"]["mean"][j] for s in S]), np.array([s["B_even_temper"]["mean"][j] for s in S]))
                       for j, nm in enumerate(NAMES)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["real", "twin", "temper"])
    ap.add_argument("--decoder", default="eegnet")
    ap.add_argument("--rho_from", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--parts", default="A,B", help="twin: which analyses (A,B,eeg_only,linkedfull,linked)")
    ap.add_argument("--noise", default="gauss", choices=["gauss", "real"],
                    help="twin read-out noise: Gaussian, or the real decoder's own residuals (y minus its fit on q~ at the MAP)")
    ap.add_argument("--subs", default="1-23")
    ap.add_argument("--m1", type=int, default=6000)
    ap.add_argument("--m2", type=int, default=15000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        R = json.loads((out / "results.json").read_text()); R["summary"] = summarize_temper(R) if R["part"] == "temper" else summarize(R)
        (out / "results.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R["summary"], indent=1)); return
    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    model = M.EXTRA_MODELS["RWD"]
    Q = np.load(QUANT); Zmap = np.load("results/map_RWD.npy")
    probs_cpu = M.reward_schedule(T); probs = probs_cpu.to(dev)
    G = json.loads(Path("results/group_stats/results.json").read_text())["decoders"][a.rho_from or a.decoder]
    rho_sub = np.clip(np.tanh(np.array(G["d"])), 0.0, 0.95)
    mu4 = torch.tensor(model.prior_mean, device=dev); sd4 = torch.tensor(model.prior_sd, device=dev)
    tt = np.arange(T); blk = (tt // BLOCK) % 2
    mk = lambda b: torch.tensor(b.astype(np.float32), device=dev)[None]
    masks = (mk(tt < 240), mk(tt >= 240), mk(blk == 0), mk(blk == 1), mk(np.ones(T, bool)), mk(np.zeros(T, bool)))
    lo, hi = (int(x) for x in a.subs.split("-"))
    subs = list(range(lo - 1, hi))
    RO = decoder_readouts(a.decoder, Q, Zmap, model, probs_cpu)
    R = {"part": a.part, "decoder": a.decoder, "subjects": {}}
    g = np.random.default_rng(11 + lo)
    with torch.no_grad():
        for i in subs:
            y, tri = RO[i]
            rho = float(rho_sub[i]); cen = torch.tensor(Zmap[i], dtype=torch.float32, device=dev)
            if a.part == "temper":
                Dt = Data(Q["choice"][i], Q["reward"][i], Q["missing"][i], y, tri, dev, model, probs)
                rec = temper_one(Dt, rho, cen, mu4, sd4, a.m1, a.m2, a.chunk, dev, shift=int(g.integers(40, 160)))
                rec["rho"] = rho
                R["subjects"][str(i + 1)] = rec
                h = rec["h1_to_h2"]
                print(f"sub {i + 1} lam {h['eeg']['lam']} (shifted {h['shifted']['lam']}) | sel-beh {h['eeg']['lpred'] - h['beh']['lpred']:+.2f} "
                      f"lam1-beh {h['lam1']['lpred'] - h['beh']['lpred']:+.2f} shifted-beh {h['shifted']['lpred'] - h['beh']['lpred']:+.2f}", flush=True)
                (out / "results.json").write_text(json.dumps(R, indent=1))
                continue
            if a.part == "real":
                Dt = Data(Q["choice"][i], Q["reward"][i], Q["missing"][i], y, tri, dev, model, probs)
                phi = ar1_phi(Dt, cen, rho, dev)
                rec = fits_one(Dt, rho, cen, mu4, sd4, masks, a.m1, a.m2, a.chunk, dev, ("A", "ar1", "eeg_only", "linked"), phi=phi)
                rec.update(rho=rho, phi=phi)
                R["subjects"][str(i + 1)] = rec
                print(f"sub {i + 1} rho={rho:.2f} phi={phi:.2f} | A joint {rec['A_joint']['lpred'] - rec['A_beh']['lpred']:+.2f} "
                      f"ar1 {rec['A_joint_ar1']['lpred'] - rec['A_beh']['lpred']:+.2f} linked {rec['A_linked']['lpred'] - rec['A_beh']['lpred']:+.2f} | "
                      f"alpha_n-alpha {rec['linked_alpha_n_minus_b']['mean']:+.2f} d_n-d {rec['linked_d_n_minus_b']['mean']:+.2f} | "
                      f"eeg-only alpha {rec['eeg_only']['mean'][0]:+.2f} vs beh {rec['full_beh']['mean'][0]:+.2f}", flush=True)
            else:
                Dr = Data(Q["choice"][i], Q["reward"][i], Q["missing"][i], y, tri, dev, model, probs)
                if a.noise == "real":                       # real decoder residual structure
                    _, pe_r = Dr.run(cen[None], masks[4])
                    P_r = projector(regressors(Dr.c, Dr.r, tri), dev)
                    q_r = resid_z(pe_r[:, torch.tensor(tri, device=dev)], P_r)[0].cpu().numpy()
                    y_r = resid_z(torch.tensor(y, dtype=torch.float32, device=dev)[None], P_r)[0].cpu().numpy()
                    e_r = y_r - (y_r @ q_r) / (q_r @ q_r) * q_r
                    e_r = (e_r - e_r.mean()) / e_r.std()
                    e_map = dict(zip(tri.tolist(), e_r))
                post = is_generic(lambda z: chunked(lambda zk: (Dr.run(zk, masks[4])[0], None), z, a.chunk),
                                  cen, 0.6 * sd4, mu4, sd4, a.m1, a.m2, dev, keep=True)
                for k in range(a.k):
                    th = post["z"][torch.multinomial(post["w"], 1)][0]
                    sim = M.run(model, th[None], probs)
                    c = sim["choices"][0].cpu().numpy(); r = sim["rewards"][0].cpu().numpy(); ms = sim["missing"][0].cpu().numpy()
                    tri_k = np.array([t for t in tri if not ms[t]])
                    _, pe = Data(c, r, ms, np.zeros(len(tri_k)), tri_k, dev, model, probs).run(th[None], masks[4])
                    P = projector(regressors(c.astype(int), r.astype(float), tri_k), dev)
                    qt = resid_z(pe[:, torch.tensor(tri_k, device=dev)], P)[0].cpu().numpy()
                    noise = np.array([e_map[t] for t in tri_k]) if a.noise == "real" else g.standard_normal(len(tri_k))
                    yk = rho * qt + math.sqrt(1 - rho ** 2) * noise
                    Dt = Data(c, r, ms, yk, tri_k, dev, model, probs)
                    rec = fits_one(Dt, rho, th, mu4, sd4, masks, a.m1, a.m2, a.chunk, dev, tuple(a.parts.split(",")))
                    rec.update(rho=rho, theta=th.tolist(), real=i + 1)
                    R["subjects"][f"{i + 1}_{k}"] = rec
                    msg = f"twin {i + 1}.{k} rho={rho:.2f} true alpha {th[0]:+.2f}"
                    if "A_joint" in rec:
                        msg += f" | A joint-beh {rec['A_joint']['lpred'] - rec['A_beh']['lpred']:+.2f}"
                    if "eeg_only" in rec:
                        msg += f" | eeg-only alpha {rec['eeg_only']['mean'][0]:+.2f} beh {rec['full_beh']['mean'][0]:+.2f}"
                    if "linked_alpha_n_minus_b" in rec:
                        msg += f" | alpha_n-alpha {rec['linked_alpha_n_minus_b']['mean']:+.2f}"
                    print(msg, flush=True)
            (out / "results.json").write_text(json.dumps(R, indent=1))
    R["summary"] = summarize_temper(R) if a.part == "temper" else summarize(R)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R["summary"], indent=1))


if __name__ == "__main__":
    main()
