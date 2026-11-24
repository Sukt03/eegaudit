"""MI retest replicate (, 23 Nov 2026).
Within-participant (blocked) yield R of the SAME read-outs as the MI card LPGO rows, EEGMMIDB 4-class imagery, bits per trial.
Design = mi_card.py secondary_within (unchanged splits): per participant, leave one run PAIR out (pairs (4,6), (8,10), (12,14)); the other two
pairs are the training set (inner: each once as validation for (lambda,T) / (step,T)); final fit on both. Floors: noise input (the v2 noise draw,
same generator seed, regressed on its own proxies for the regressed rows) and within-participant shuffled training labels.
No existing code is edited: mi_card_v3 / mi_fm_eogreg functions are imported; the only copied piece is run_design (adds per-trial bits for split halves).
Arms:
  logistic   rows raw, raw_eogreg  (kernel-form logistic as v2 secondary_within, 56 training trials)
  eegnet     --regress 0/1 -> row eegnet / eegnet_eogreg (600 steps, every 20, batch 32 as v2 --within_eegnet)
  fm         --model cbramod|labram|eegpt|reve, frozen pretrained, input regressed (--regress 1, as mi_fm_eogreg) or not; noise floor = FM on unregressed noise
Usage: --arm A --out DIR [--subjects 1,2,3 | --shard i --nshard n] [--regress 1] [--model reve]
Outputs: DIR/sub-XXX.json (per participant bits/acc per row x cond), DIR/pertrial.npz (per-trial bits + participant index), DIR/meta.json."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/path/to/eegaudit/experiments/ext")
import mi_card_v3 as M

CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
DEV = "cuda"; ROOT = M.ROOT; log = M.log
LAMS = M.LAMS


def run_design_pt(units, fits, Fdict, y, ylab_by_cond, sid, nP, eeg_members, eeg_units, Xin, yin, eeg_cfg, eeg_name):
    """M.run_design (within branch) + per-trial bits. Returns R[k][row][cond] = dict(bits, acc, n_test), PT['row|cond'] = np [N], diag."""
    N = len(y); R = {}; PT = {}; diag = {}
    for cond, FS in Fdict.items():
        ylab = ylab_by_cond[cond]
        for fname, Xf in FS.items():
            Z, gm = M.kernel_fits(Xf, fits, ylab, LAMS)
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV); ne = [0, 0]
            for u in units:
                Zi = torch.cat([Z[i] for i in u["inner"]], 1); yi = torch.cat([y[fits[i][1]] for i in u["inner"]])
                l, T, e = M.select(Zi, yi); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]
                f = u["final"]; ev = fits[f][1]; Zf = Z[f][l]
                bt[ev] = M.bits(Zf, y[ev], T, M.prior_of(ylab[f])); ok[ev] = (Zf.argmax(-1) == y[ev]).double()
            assert not torch.isnan(bt).any(), f"{cond} {fname}: untested trials"
            diag[f"{cond}|{fname}"] = dict(max_grad=gm, lambda_edge=ne[0], T_edge=ne[1], n_units=len(units)); PT[f"{fname}|{cond}"] = bt.cpu().numpy()
            for k in range(nP):
                r = sid == k; R.setdefault(k, {}).setdefault(fname, {})[cond] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), n_test=int(r.sum()))
            log(f"logistic {cond} {fname}: max|grad| {gm:.1e}; mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    if eeg_members:
        Zn = M.eegnet_members(Xin, yin, eeg_members, **eeg_cfg)
        for tw in ("eeg", "noise"):
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV); ne = [0, 0]
            for u in [u for u in eeg_units if u["twin"] == tw]:
                isel, ifin = u["inner"][0], u["final"]
                c, T, e = M.select(Zn[isel], yin[eeg_members[isel][1]]); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]
                ev = eeg_members[ifin][1]; evy = ev % N; Zf = Zn[ifin][c]
                bt[evy] = M.bits(Zf, yin[ev], T, M.prior_of(yin[eeg_members[ifin][0]])); ok[evy] = (Zf.argmax(-1) == yin[ev]).double()
            assert not torch.isnan(bt).any(), f"{eeg_name} {tw}: untested trials"
            diag[f"{tw}|{eeg_name}"] = dict(step_edge=ne[0], T_edge=ne[1]); PT[f"{eeg_name}|{tw}"] = bt.cpu().numpy()
            for k in range(nP):
                r = sid == k; R.setdefault(k, {}).setdefault(eeg_name, {})[tw] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), n_test=int(r.sum()))
            log(f"{eeg_name} {tw}: mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    return R, PT, diag


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["logistic", "eegnet", "fm"]); ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default=""); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--regress", type=int, default=1); ap.add_argument("--model", default="reve"); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mchunk", type=int, default=72)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "results/ext/mi_card_cache"; D = {}; excl = {}
    for s in range(1, 110):                                    # same QC as mi_card
        d, why = M.load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d["y"], minlength=4)
        if c.max() - c.min() > 2: excl[s] = f"class counts {c.tolist()}"; continue
        D[s] = d
    P = sorted(D); nP0 = len(P); ch = D[P[0]]["ch"]; assert nP0 == 105, nP0
    subs = {int(x) for x in a.subjects.split(",")} if a.subjects else None
    act = [k for k in range(nP0) if (P[k] in subs if subs else k % a.nshard == a.shard)]; Pa = [P[k] for k in act]; nA = len(Pa)
    log(f"arm {a.arm} regress {a.regress}: {nA} participants {Pa[:3]}..{Pa[-1]}")
    keep_nofp = torch.tensor([c not in M.FP for c in ch], device=DEV)
    Xall = np.concatenate([D[s]["X"] for s in P]); sid_all = np.concatenate([[k] * len(D[s]["y"]) for k, s in enumerate(P)])
    N0 = len(Xall); rows_a = np.where(np.isin(sid_all, act))[0]
    g = torch.Generator(device=DEV); g.manual_seed(20261121)
    Rn = torch.randn((N0, 64, 960), generator=g, device=DEV)    # = v2 noise draw (full cohort, then sliced)
    Nz = torch.empty((len(rows_a), 64, 960), dtype=torch.float32)
    pos = {int(r): i for i, r in enumerate(rows_a)}; sel = torch.tensor(rows_a, device=DEV)
    for i in range(0, len(rows_a), 1500): Nz[i:i + 1500] = M.bandpass(Rn[sel[i:i + 1500]], 1, 40).cpu()
    del Rn; torch.cuda.empty_cache()
    X = torch.tensor(Xall[rows_a], dtype=torch.float32); del Xall
    y = torch.tensor(np.concatenate([D[s]["y"] for s in Pa]), device=DEV); pair = torch.tensor(np.concatenate([D[s]["pair"] for s in Pa]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D[s]["y"]) for k, s in enumerate(Pa)]), device=DEV); N = len(y); sidc = sid.cpu()
    Xb = torch.empty_like(X)
    for i in range(0, N, 1500): Xb[i:i + 1500] = M.bandpass(X[i:i + 1500].to(DEV), 1, 40).cpu()
    for k in range(nA):                                          # v2 noise scaling: per participant per channel SD of the 1-40 Hz EEG
        r = sidc == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * Xb[r].std((0, 2), keepdim=True)
    # within design (mi_card.py secondary_within, unchanged)
    fits, units = [], []
    for k in range(nA):
        for oo in range(3):
            pa, pb = [q for q in range(3) if q != oo]
            ia, ib, io = [torch.where((sid == k) & (pair == q))[0] for q in (pa, pb, oo)]
            fits += [(ia, ib), (ib, ia), (torch.cat([ia, ib]), io)]; units.append(dict(inner=[len(fits) - 3, len(fits) - 2], final=len(fits) - 1))
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7)
    ylab = {"eeg": [y[f[0]] for f in fits], "noise": [y[f[0]] for f in fits],
            "shuffle": [y[f[0]][torch.randperm(len(f[0]), generator=gperm, device=DEV)] for f in fits]}
    share = None
    if a.regress:
        Xe, share = M.eog_regress(Xb, sidc, nA, ch)               # cpu in -> cpu out (same call as mi_fm_eogreg)
    else:
        Xe = Xb
    Fdict, Xin, yin, mem, eu, cfg, info = {}, None, None, [], [], None, {}
    if a.arm == "logistic":
        FE, _ = M.features(X.to(DEV), keep_nofp); FN, _ = M.features(Nz.to(DEV), keep_nofp)
        Rr, share = M.eog_regress(M.bandpass(X.to(DEV), 1, 40), sid, nA, ch); FEr, _ = M.features(Rr, keep_nofp); del Rr
        Rn_, _ = M.eog_regress(M.bandpass(Nz.to(DEV), 1, 40), sid, nA, ch); FNr, _ = M.features(Rn_, keep_nofp); del Rn_
        Fe = {"raw": FE["raw"], "raw_eogreg": FEr["raw"]}; Fn = {"raw": FN["raw"], "raw_eogreg": FNr["raw"]}
        Fdict = {"eeg": Fe, "noise": Fn, "shuffle": Fe}; eeg_name = None; del FE, FN, FEr, FNr
    elif a.arm == "eegnet":
        eeg_name = "eegnet_eogreg" if a.regress else "eegnet"
        XinE = M.bandpass(Xe.to(DEV), 4, 40)[:, :, M.W0:M.W1].float()
        if a.regress:
            Rn_, _ = M.eog_regress(M.bandpass(Nz.to(DEV), 1, 40), sid, nA, ch); XinN = M.bandpass(Rn_, 4, 40)[:, :, M.W0:M.W1].float(); del Rn_
        else:
            XinN = M.bandpass(Nz.to(DEV), 4, 40)[:, :, M.W0:M.W1].float()
        Xin = torch.cat([XinE, XinN]); del XinE, XinN
        for k in range(nA):
            r = torch.where(sid == k)[0]; sc = Xin[r].std(); Xin[r] /= sc; Xin[r + N] /= sc
        yin = torch.cat([y, y])
        for k in range(nA):
            for oo in range(3):
                pa, pb = [q for q in range(3) if q != oo]
                ia, ib, io = [torch.where((sid == k) & (pair == q))[0] for q in (pa, pb, oo)]
                for tw, off in (("eeg", 0), ("noise", N)):
                    mem.append((ia + off, ib + off)); mem.append((torch.cat([ia, ib]) + off, io + off)); eu.append(dict(inner=[len(mem) - 2], final=len(mem) - 1, twin=tw))
        cfg = dict(steps=a.steps, every=20, batch=32, mchunk=a.mchunk, seed=13)
    else:
        import mi_fm_eogreg as FMG
        eeg_name = None; zst = {}
        for nm, XX in (("eeg", Xe), ("noise", Nz)):
            mu = torch.stack([XX[sidc == k][:, :, M.W0:M.W1].mean((0, 2)) for k in range(nA)]).to(DEV)
            sd = torch.stack([XX[sidc == k][:, :, M.W0:M.W1].std((0, 2)) for k in range(nA)]).to(DEV) + 1e-6; zst[nm] = (mu, sd)
        Fe, ie = FMG.extract(a.model, True, Xe, sid, zst["eeg"], ch); Fn, inn = FMG.extract(a.model, True, Nz, sid, zst["noise"], ch)
        info = dict(eeg=ie, noise=inn, label=FMG.ELIG[a.model][0]); row = f"fm_{a.model}"
        Fdict = {"eeg": {row: Fe}, "noise": {row: Fn}, "shuffle": {row: Fe}}
    R, PT, diag = run_design_pt(units, fits, Fdict, y, ylab, sid, nA, mem, eu, Xin, yin, cfg, eeg_name or "")
    for k, s in enumerate(Pa): (out / f"sub-{s:03d}.json").write_text(json.dumps(dict(participant=s, **R[k]), indent=1))
    np.savez(out / "pertrial.npz", sid=sidc.numpy(), participants=np.array(Pa), y=y.cpu().numpy(), **{k.replace("|", "__"): v for k, v in PT.items()})
    (out / "meta.json").write_text(json.dumps(dict(arm=a.arm, regress=a.regress, model=a.model if a.arm == "fm" else None, participants=Pa, n=nA,
        design="mi_card.py secondary_within: leave one run pair out within participant (3 folds), inner = the two training pairs", steps=a.steps if a.arm == "eegnet" else None,
        eog_regression_variance_share_mean=float(np.mean(share)) if share else None, code_sha256=CODE_SHA256, diag=diag, info=info,
        status="", runtime_s=round(time.time() - M.t0)), indent=1, default=str))
    log("DONE", {k: round(float(np.mean([R[q][k][c]["bits"] for q in range(nA)])), 4) for k in R[0] for c in R[0][k]} if False else "")


if __name__ == "__main__":
    main()
