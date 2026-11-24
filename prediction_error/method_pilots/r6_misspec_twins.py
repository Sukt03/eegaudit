"""r6: which read-out misspecification reproduces BOTH symptoms of the real joint fit?

Real data (results/real_verify_eegnet): the joint fit contracts the learning-rate posterior to 0.51 of its behavioural
variance where the exact analysis predicts 0.70, and held-out choice prediction does not improve. Correctly specified twins
(results/real_twin_eegnet_a/b/c) give 0.71-0.73 and a gain of about +0.0005 nats/trial.

Here: twins of real participants (theta from the behaviour-only posterior, choices simulated on the real schedule, the
participant's EEG trial set and target coupling), but the read-out is generated under a misspecified linking function.
The fit is the one of real_joint_fit.py / real_joint_ext.py (Gaussian signed-PE read-out, rho fixed), test A (fit on trials
1-240, score 241-480) and test C (first-half alpha variance ratio). As in the real pipeline, rho entering the likelihood is
the MEASURED correlation between the read-out and the signed-PE regressor q~ (here under the true theta), not the
generative coupling; the generative weight a is chosen so that this measured rho equals the participant's real rho when
the misspecified signal allows it.

Families (--families):
  correct        y = rho q~ + sqrt(1-rho^2) eps                                    (baseline, = existing twins)
  unsigned       signal = |PE| (residualised, z-scored like q~)                     (i)
  mix_unsigned   signal = z(q~) + z(|PE|~)                                          (i)
  t3             eps ~ Student-t(3), unit variance                                  (ii)
  hetero         eps sd proportional to (1 + |q~|), unit mean variance              (ii)
  ar1_0.3/0.5    eps AR(1) over consecutive EEG trials                              (iii)
  alpha_n-1.0 / alpha_n-0.6 / alpha_n+1.0   PE computed under logit alpha + delta   (iv)
  alpha_n_rand   PE computed under logit alpha_n ~ N(0, 1.5) independent of theta  (iv)
  prev_1.0 / prev_0.5   signal = PE_t + w PE_{t-1}                                  (v)
Usage: python r6_misspec_twins.py --subs 1,3,4,6,8,12,16,19 --k 2 --families correct,t3 --tag smoke
Output: results/method_pilots/r6_misspec_<tag>.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path("/path/to/eegaudit")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cogmodels import models as M
from real_joint_fit import QUANT, T, BLOCK, CURVE, regressors, projector, resid_z, decoder_readouts
from real_joint_ext import Data, fits_one, is_generic, chunked

OUT = ROOT / "results/method_pilots"
NAMES = ["logit_alpha", "logit_d", "log_beta", "kappa"]


def pred_ratio_fn():
    C = json.loads(Path(CURVE).read_text())
    rhos = sorted(float(k) for k in C["joint"])
    key = lambda r: next(k for k in C["joint"] if abs(float(k) - r) < 1e-9)
    c = np.array([C["joint"][key(r)]["exact"]["logit_alpha"]["contraction"] for r in rhos])
    return lambda rho: float((1 - np.interp(rho, rhos, c)) / (1 - c[0]))


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def resid_np(v, C):
    v = np.asarray(v, float)
    return zs(v - C @ np.linalg.lstsq(C, v, rcond=None)[0])


def ar1_noise(n, phi, g):
    e = np.empty(n)
    e[0] = g.standard_normal()
    for t in range(1, n):
        e[t] = phi * e[t - 1] + math.sqrt(1 - phi ** 2) * g.standard_normal()
    return e


def make_signal(fam, pe_true, tri_k, C, g, run_pe, th):
    """Returns (signal on tri_k, standardised and residualised; noise generator or None for Gaussian; info dict)."""
    info = {}
    if fam in ("correct", "t3", "hetero") or fam.startswith("ar1_"):
        s = resid_np(pe_true[tri_k], C)
    elif fam == "unsigned":
        s = resid_np(np.abs(pe_true[tri_k]), C)
    elif fam == "mix_unsigned":
        s = resid_np(resid_np(pe_true[tri_k], C) + resid_np(np.abs(pe_true[tri_k]), C), C)
    elif fam.startswith("alpha_n"):
        thn = th.clone()
        if fam == "alpha_n_rand":
            thn[0] = float(g.normal(0.0, 1.5))
        else:
            thn[0] = thn[0] + float(fam[len("alpha_n"):])
        info["logit_alpha_n"] = float(thn[0]); info["logit_alpha_b"] = float(th[0])
        s = resid_np(run_pe(thn)[tri_k], C)
    elif fam.startswith("prev_"):
        w = float(fam[len("prev_"):])
        pe_prev = np.r_[0.0, pe_true[:-1]]
        s = resid_np((pe_true + w * pe_prev)[tri_k], C)
    else:
        raise ValueError(fam)
    return s, info


def make_noise(fam, n, q, g):
    if fam == "t3":
        return g.standard_t(3, size=n) / math.sqrt(3.0)
    if fam == "hetero":
        sdv = 1 + np.abs(q)
        return g.standard_normal(n) * sdv / math.sqrt(np.mean(sdv ** 2))
    if fam.startswith("ar1_"):
        return ar1_noise(n, float(fam[len("ar1_"):]), g)
    return g.standard_normal(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="1,3,4,6,8,12,16,19")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--families", default="correct")
    ap.add_argument("--decoder", default="eegnet")
    ap.add_argument("--rho_use", default="measured", choices=["measured", "target"])
    ap.add_argument("--m1", type=int, default=6000)
    ap.add_argument("--m2", type=int, default=15000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    g = np.random.default_rng(100 + a.seed)
    model = M.EXTRA_MODELS["RWD"]
    Q = np.load(QUANT); Zmap = np.load("results/map_RWD.npy")
    probs_cpu = M.reward_schedule(T); probs = probs_cpu.to(dev)
    G = json.loads(Path("results/group_stats/results.json").read_text())["decoders"][a.decoder]
    rho_sub = np.clip(np.tanh(np.array(G["d"])), 0.0, 0.95)
    mu4 = torch.tensor(model.prior_mean, device=dev); sd4 = torch.tensor(model.prior_sd, device=dev)
    tt = np.arange(T); blk = (tt // BLOCK) % 2
    mk = lambda b: torch.tensor(b.astype(np.float32), device=dev)[None]
    masks = (mk(tt < 240), mk(tt >= 240), mk(blk == 0), mk(blk == 1), mk(np.ones(T, bool)), mk(np.zeros(T, bool)))
    full = masks[4]
    pred_ratio = pred_ratio_fn()
    RO = decoder_readouts(a.decoder, Q, Zmap, model, probs_cpu)
    subs = [int(x) - 1 for x in a.subs.split(",")]
    fams = a.families.split(",")
    out = OUT / f"r6_misspec_{a.tag}.json"
    R = {"tag": a.tag, "decoder": a.decoder, "rho_use": a.rho_use, "k": a.k, "m1": a.m1, "m2": a.m2, "seed": a.seed,
         "families": fams, "subs": [s + 1 for s in subs], "twins": {}}
    t0 = time.time()
    with torch.no_grad():
        for i in subs:
            y_real, tri = RO[i]
            rho = float(rho_sub[i]); cen = torch.tensor(Zmap[i], dtype=torch.float32, device=dev)
            Dr = Data(Q["choice"][i], Q["reward"][i], Q["missing"][i], y_real, tri, dev, model, probs)
            post = is_generic(lambda z: chunked(lambda zk: (Dr.run(zk, full)[0], None), z, a.chunk),
                              cen, 0.6 * sd4, mu4, sd4, a.m1, a.m2, dev, keep=True)
            for k in range(a.k):
                th = post["z"][torch.multinomial(post["w"], 1)][0]
                torch.manual_seed(a.seed * 1000 + i * 10 + k)
                sim = M.run(model, th[None], probs)
                c = sim["choices"][0].cpu().numpy(); r = sim["rewards"][0].cpu().numpy(); ms = sim["missing"][0].cpu().numpy()
                tri_k = np.array([t for t in tri if not ms[t]])
                D0 = Data(c, r, ms, np.zeros(len(tri_k)), tri_k, dev, model, probs)
                run_pe = lambda z: D0.run(z[None], full)[1][0].cpu().numpy()
                pe_true = run_pe(th)
                C = regressors(c.astype(int), r.astype(float), tri_k)
                q = resid_np(pe_true[tri_k], C)                       # the regressor the fit uses (at the true theta)
                for fam in fams:
                    gf = np.random.default_rng(g.integers(1 << 31))
                    s, info = make_signal(fam, pe_true, tri_k, C, gf, run_pe, th)
                    eps = make_noise(fam, len(tri_k), q, gf)
                    csq = float(np.corrcoef(s, q)[0, 1])
                    amp = min(1.0, rho / csq) if csq > 0.05 else rho
                    y = amp * s + math.sqrt(1 - amp ** 2) * eps
                    rho_meas = float(np.clip(np.corrcoef(resid_np(y, C), q)[0, 1], 0.0, 0.95))
                    rho_fit = rho_meas if a.rho_use == "measured" else rho
                    Dt = Data(c, r, ms, y, tri_k, dev, model, probs)
                    res = fits_one(Dt, rho_fit, th, mu4, sd4, masks, a.m1, a.m2, a.chunk, dev, ("A",))
                    vr = [res["A_joint"]["var"][j] / res["A_beh"]["var"][j] for j in range(4)]
                    gain = (res["A_joint"]["lpred"] - res["A_beh"]["lpred"]) / res["n_test"]
                    rec = {"real": i + 1, "twin": k, "family": fam, "theta": th.tolist(), "rho_target": rho, "corr_signal_q": csq,
                           "amp": amp, "rho_measured": rho_meas, "rho_fit": rho_fit, "pred_ratio_at_rho_fit": pred_ratio(rho_fit),
                           "var_ratio": vr, "gain_nats_per_trial": gain, "n_test": res["n_test"],
                           "ess": [res["A_beh"]["ess"], res["A_joint"]["ess"]],
                           "mean_beh": res["A_beh"]["mean"], "mean_joint": res["A_joint"]["mean"],
                           "var_beh": res["A_beh"]["var"], "var_joint": res["A_joint"]["var"], **info}
                    R["twins"][f"{i + 1}_{k}_{fam}"] = rec
                    print(f"[{time.time() - t0:6.0f}s] sub {i + 1}.{k} {fam:13s} rho_t {rho:.2f} c(s,q) {csq:+.2f} amp {amp:.2f} "
                          f"rho_fit {rho_fit:.2f} pred {rec['pred_ratio_at_rho_fit']:.2f} | alpha ratio {vr[0]:.2f} | "
                          f"gain {gain:+.4f} | ess {res['A_joint']['ess']:.3f}", flush=True)
                out.write_text(json.dumps(R, indent=1))
    R["elapsed_s"] = time.time() - t0
    out.write_text(json.dumps(R, indent=1))
    print("->", out, f"{R['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
