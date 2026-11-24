"""N-D. REVE PE pretraining dividend: generality (ds004595) and eye control (event-regressed ds003458). FIXER-2, 21 Nov 2026.
Spec: EXTENSIONS_LOOP_20261121 "N-D" + the bug pre-review (N-D bullets). m7_latent.py is imported UNCHANGED (load path, reve_layers,
  nested_pe, ridge_dual_pe, gain_rows_t, bandpower, density_gain_t, boot_ci, signflip_p, holm); the M7 PE loop is re-written here
  without the INLP rows (not part of N-D) so that it can take a second dataset and per-fold inputs.
Modes:
  --data ds004595             the M7 PE pipeline on ds004595 (results/fm_epochs_ds004595: same cleaning family, 200 Hz, -1..1 s,
                              uV/100, post-feedback REVE patch -0.1..0.9 s; target = s13b targets_ds004595, the verified adapter;
                              trials = results/features_ds004595/raw trials). Arms: raw (X7-style features_ds004595/raw), raw_win,
                              bandpower, reve, reve_twin, each with its noise-input floor; layer chosen per outer fold on inner CV.
                              Dedicated reference = the ds004595 EEGNet read-outs (features_ds004595/eegnet and eegnet_post; density gain).
  --data ds003458             same loop on ds003458 = G2: every arm reproduces results/ext/m7/pe_sub-XXX.json (registered M7 PE).
  --data ds003458 --eventreg  before REVE, the per-trial blink / saccade EVENT predictors (pe_eogreg.ocular_v2 events: VEOG re-referenced
                              to the average; blink = VEOG p2p > median + 3 MAD, saccade = (F7-F8) p2p > median + 3 MAD) are regressed
                              out of every channel x sample of the standardized epoch IN-FOLD: for each outer fold b, OLS of X[:, c, t] on
                              [1, blink, saccade] over fold b's training rows; X' = X - (E - mean_train(E)) B_E (the event deviation is
                              removed, the across-trial mean = the ERP is kept, so REVE still sees an epoch, not a mean-removed residual);
                              applied to training and held-out rows; then embedded (REVE and twin, 10 embeddings per model). Inner calibration
                              fits of fold b use fold b's regressed inputs (inner rows are a subset of fold b's training rows).
                              The same per-fold regression is applied to the raw_win / bandpower inputs and to the X7 raw features.
                              Cost choice (bug pre-review): regression on the RAW epoch, per outer fold, as specified (not on the features).
Floors: target circularly shifted by n/4, n/2, 3n/4 (all modes); N(0,1) noise epochs (seed 3000 + s) through REVE / twin, Gaussian raw
  features (seed 1000 + s) and noise raw_win / bandpower (plain modes). G3: |floor| <= 0.02 per arm (group means printed).
Contrasts (summary; participant bootstrap 10,000; one-sided sign-flip p; Holm over the contrasts of one mode): reve - reve_twin
  (the dividend), reve - raw_win, reve_twin - raw_win; for --eventreg also the paired change of the dividend vs the registered plain M7
  files (eventreg dividend - plain dividend, same participants).
Falsifier (pre-stated, N-D): REVE - twin CI includes 0 on ds004595 OR on event-regressed ds003458 -> "dividend specific to ds003458 /
  not robust to eye events". Context (GAP FINDER round 3): ds004595's own yield is small (~0.006), which bounds the replication.
Outputs: results/ext/nd_reve/{pe4595,pe3458_g2,pe_eventreg}/pe_sub-XXX.json (+ summary.json), every file with code_sha256.
Usage: python nd_reve_generality.py --data ds004595|ds003458 [--eventreg] [--subjects 1,2] --out DIR | --summarize --out DIR
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="ds004595"); ap.add_argument("--eventreg", action="store_true")
ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
A = ap.parse_args()
import d2_pe_tasks as D2                                                     # adds experiments/loop etc. to sys.path
_argv = sys.argv; sys.argv = ["m7_latent.py"]; import m7_latent as M7; sys.argv = _argv   # m7 parses its own argv at import (defaults only)
from belief_bottleneck import LAM, folds
DEV = M7.DEV; LAYERS = M7.LAYERS; WIN = slice(180, 380)


def nested_fold(F, colw, Ys, fo, lam, b):
    """= m7_latent.nested_pe restricted to outer fold b (same algebra, same order): gains of fold b's held-out rows (len te, 4), inner r."""
    tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; trt, tet = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
    Po = M7.ridge_dual_pe(F, colw, Ys, trt, tet, lam); fi = folds(len(tr), 9); Pin = torch.zeros(len(tr), 4, dtype=torch.float64, device=DEV)
    for c in range(9):
        itr, ite = torch.tensor(tr[fi != c], device=DEV), torch.tensor(tr[fi == c], device=DEV); Pin[torch.tensor(fi == c, device=DEV)] = M7.ridge_dual_pe(F, colw, Ys, itr, ite, lam)
    return M7.gain_rows_t(Ys[trt], Pin, Ys[tet], Po), float(torch.corrcoef(torch.stack([Pin[:, 0], Ys[trt, 0]]))[0, 1])


def one(F):
    return torch.ones(F.shape[1], dtype=torch.float64, device=DEV)


def rwin(X):
    n = len(X); return X[:, :, WIN].reshape(n, X.shape[1], 50, 4).mean(-1).reshape(n, -1).double()


def models():
    from transformers import AutoModel, AutoConfig
    positions = AutoModel.from_pretrained(str(ROOT / "external/reve/positions"), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(DEV).eval()
    mods = {}
    for name, pre in (("reve", True), ("reve_twin", False)):                # = m7_latent.run_pe model construction (twin: seed 0, from_config)
        if pre: m = AutoModel.from_pretrained(str(ROOT / "external/reve/base"), trust_remote_code=True, local_files_only=True, use_safetensors=True)
        else:
            torch.manual_seed(0); m = AutoModel.from_config(AutoConfig.from_pretrained(str(ROOT / "external/reve/base"), trust_remote_code=True, local_files_only=True), trust_remote_code=True)
        mods[name] = m.to(DEV).eval()
    return positions, mods


def load(ds, s, qtil, fpath, known):
    if ds == "ds003458":
        tri = np.load(fpath("fm", s))["tri"]; E = np.load(ROOT / f"results/fm_epochs/sub-{s:03d}.npz")
    else:
        E = np.load(ROOT / f"results/fm_epochs_ds004595/sub-{s:03d}.npz"); have = set(E["trial"].tolist())
        tri0 = np.load(fpath("raw", s))["tri"]; tri = np.array([t for t in tri0 if t in have])
    pidx = {t: k for k, t in enumerate(E["trial"])}; names = [str(c) for c in E["ch"]]; keep = [k for k, c in enumerate(names) if c in known]
    X = E["X"][[pidx[t] for t in tri]][:, keep].reshape(len(tri), len(keep), -1).astype(np.float32)
    X = torch.from_numpy(X).to(DEV); X = (X - X.mean((0, 2), keepdim=True)) / (X.std((0, 2), unbiased=False, keepdim=True) + 1e-6)
    raw = np.load(fpath("raw", s)); rpos = {t: k for k, t in enumerate(raw["tri"])}; ri = [rpos[t] for t in tri]
    Fraw = torch.tensor(raw["F"][ri], dtype=torch.float64, device=DEV); cw = torch.tensor(raw["colw"], dtype=torch.float64, device=DEV)
    return tri, X, [names[k] for k in keep], len(names), Fraw, cw


def subject(s, qtil, fpath, positions, mods, out):
    t0 = time.time(); ds = A.data; tri, X, chans, nch_all, Fraw, cw = load(ds, s, qtil, fpath, set(positions.mapping)); pos = positions(chans).to(DEV)
    r = dict(subject=f"sub-{s:03d}", dataset=ds, mode="eventreg" if A.eventreg else "plain", code_sha256=CODE_SHA256, channels_with_positions=len(chans), channels=nch_all,
             input_check="200 Hz, -1..1 s = 400 samples -> 2 REVE patches; post-feedback patch (-0.1..0.9 s) used (m7 PE convention)", arms={})
    if A.eventreg:
        import pe_eogreg as EG
        _, Ev, kp = EG.ocular_v2(s, tri); r["n_lost_no_ocular_epoch"] = int((~kp).sum()); r["blink_rate"] = float(Ev[:, 0].mean()); r["saccade_rate"] = float(Ev[:, 1].mean())
        tri, X, Fraw = tri[kp], X[torch.tensor(kp, device=DEV)], Fraw[torch.tensor(kp, device=DEV)]
    y = torch.tensor(qtil(s, tri), dtype=torch.float64, device=DEV); n = len(y); r["n_trials"] = n
    Ys = torch.stack([y] + [torch.roll(y, round(n * k / 4)) for k in (1, 2, 3)], 1); fo = folds(n)

    def summ(name, G, extra=None):
        d = dict(gain=float(G[:, 0].mean()), shifted=[float(v) for v in G[:, 1:].mean(0)], shift_mean=float(G[:, 1:].mean()), **(extra or {})); r["arms"][name] = d
        print(f"sub-{s:03d} {name}: gain {d['gain']:.4f} shift {d['shift_mean']:.4f}", f"({time.time()-t0:.0f}s)", flush=True)

    if not A.eventreg:
        Xn = torch.randn(X.shape, generator=torch.Generator().manual_seed(3000 + s)).to(DEV)
        Fraw_n = torch.tensor(np.random.default_rng(1000 + s).standard_normal(tuple(Fraw.shape)), device=DEV)
        summ("raw", M7.nested_pe(Fraw, cw, Ys, fo, LAM)[2]); summ("raw_noise", M7.nested_pe(Fraw_n, cw, Ys, fo, LAM)[2])
        rw, rwn = rwin(X), rwin(Xn); summ("raw_win", M7.nested_pe(rw, one(rw), Ys, fo, LAM)[2]); summ("raw_win_noise", M7.nested_pe(rwn, one(rwn), Ys, fo, LAM)[2])
        G0, ir0 = nested_fold(rw, one(rw), Ys, fo, LAM, 0); r["check_nested_fold_vs_nested_pe"] = float((G0 - M7.nested_pe(rw, one(rw), Ys, fo, LAM)[2][torch.tensor(fo == 0, device=DEV)]).abs().max())
        bp = M7.bandpower(X[:, :, WIN], 200).double(); bpn = M7.bandpower(Xn[:, :, WIN], 200).double()
        summ("bandpower", M7.nested_pe(bp, one(bp), Ys, fo, LAM)[2]); summ("bandpower_noise", M7.nested_pe(bpn, one(bpn), Ys, fo, LAM)[2])
        for name, m in mods.items():
            for inp, XX in (("", X), ("_noise", Xn)):
                Fl = M7.reve_layers(m, pos, XX, patch=-1, bs=64); outs = {l: M7.nested_pe(Fl[l].double(), one(Fl[l]), Ys, fo, LAM) for l in LAYERS}
                Gs = torch.full((n, 4), float("nan"), dtype=torch.float64, device=DEV); chosen = {}
                for b in range(10):
                    lb = max(LAYERS, key=lambda l: outs[l][1][b]); chosen[b] = lb; m_ = torch.tensor(fo == b, device=DEV); Gs[m_] = outs[lb][2][m_]
                summ(name + inp, Gs, dict(layers=chosen, by_layer={l: float(outs[l][2][:, 0].mean()) for l in LAYERS})); del Fl, outs; torch.cuda.empty_cache()
        decs = ["eegnet"] if ds == "ds003458" else ["eegnet", "eegnet_post"]
        r["dedicated"] = {}
        for dname in decs:
            ez = np.load(fpath(dname, s)); r["dedicated"][dname] = M7.density_gain_t(qtil(s, ez["tri"]), ez["F"][:, 0], folds(len(ez["tri"])))
        if ds == "ds003458":                                                    # G2 vs the registered M7 PE file
            reg = json.loads((ROOT / f"results/ext/m7/pe_sub-{s:03d}.json").read_text())["arms"]
            r["g2"] = {k: abs(v["gain"] - reg[k]["gain"]) for k, v in r["arms"].items() if k in reg}; r["g2_max_abs_diff"] = max(r["g2"].values())
    else:
        Ev_t = torch.tensor(Ev, dtype=torch.float64, device=DEV); Ds = torch.cat([torch.ones(n, 1, dtype=torch.float64, device=DEV), Ev_t], 1)
        G = {k: torch.full((n, 4), float("nan"), dtype=torch.float64, device=DEV) for k in ("raw", "raw_win", "bandpower", "reve", "reve_twin")}; chosen = {"reve": {}, "reve_twin": {}}; byl = {k: {l: [] for l in LAYERS} for k in chosen}
        for b in range(10):
            trm = torch.tensor(fo != b, device=DEV); tem = torch.tensor(fo == b, device=DEV); Dt = Ds[trm]; Ec = Ev_t - Ev_t[trm].mean(0)
            def reg(Z):                                                          # Z (n, ...) -> event deviation removed (coefficients on fold-b training rows)
                Zf = Z.reshape(n, -1).double(); Bc = (torch.linalg.pinv(Dt) @ Zf[trm])[1:]; return (Zf - Ec @ Bc).reshape(Z.shape)
            Xb = reg(X).float(); Fr = reg(Fraw); rw = rwin(Xb); bp = M7.bandpower(Xb[:, :, WIN], 200).double()
            for k, F_, c_ in (("raw", Fr, cw), ("raw_win", rw, one(rw)), ("bandpower", bp, one(bp))): G[k][tem] = nested_fold(F_, c_, Ys, fo, LAM, b)[0]
            for name, m in mods.items():
                Fl = M7.reve_layers(m, pos, Xb, patch=-1, bs=64); res_ = {l: nested_fold(Fl[l].double(), one(Fl[l]), Ys, fo, LAM, b) for l in LAYERS}
                lb = max(LAYERS, key=lambda l: res_[l][1]); chosen[name][b] = lb; G[name][tem] = res_[lb][0]
                for l in LAYERS: byl[name][l].append(float(res_[l][0][:, 0].sum()))
                del Fl, res_; torch.cuda.empty_cache()
            print(f"sub-{s:03d} fold {b} done ({time.time()-t0:.0f}s)", flush=True)
        for k in G: summ(k, G[k], dict(layers=chosen[k], by_layer={l: sum(v) / n for l, v in byl[k].items()}) if k in chosen else None)
        reg_m7 = json.loads((ROOT / f"results/ext/m7/pe_sub-{s:03d}.json").read_text())["arms"]
        r["plain_m7"] = {k: reg_m7[k]["gain"] for k in ("raw", "raw_win", "bandpower", "reve", "reve_twin") if k in reg_m7}
    r["runtime_s"] = round(time.time() - t0, 1); (out / f"pe_sub-{s:03d}.json").write_text(json.dumps(r, indent=1))


def summarize(out):
    R = [json.loads(f.read_text()) for f in sorted(out.glob("pe_sub-*.json"))]; n = len(R)
    S = dict(code_sha256=CODE_SHA256, n=n, participants=[r["subject"] for r in R], dataset=R[0]["dataset"], mode=R[0]["mode"], code_sha256_per_participant=sorted({r["code_sha256"] for r in R}), arms={}, contrasts={})
    arms = [k for k in R[0]["arms"] if all(k in r["arms"] for r in R)]
    for k in arms:
        g = [r["arms"][k]["gain"] for r in R]; sh = [r["arms"][k]["shift_mean"] for r in R]
        S["arms"][k] = dict(gain=float(np.mean(g)), ci=M7.boot_ci(g), shift_floor=float(np.mean(sh)), n_shift_abs_gt_002=int(sum(abs(v) > 0.02 for v in sh)), n_above_shift=int(sum(a > b for a, b in zip(g, sh))))
    fl = [v["shift_floor"] for v in S["arms"].values()] + [v["gain"] for k, v in S["arms"].items() if k.endswith("_noise")]
    S["g3_floors_max_abs_group_mean"] = float(max(abs(x) for x in fl)); S["g3_pass"] = bool(S["g3_floors_max_abs_group_mean"] <= 0.02)
    pairs = [("reve", "reve_twin"), ("reve", "raw_win"), ("reve_twin", "raw_win")]; fam = []
    for a_, b_ in pairs:
        d = [r["arms"][a_]["gain"] - r["arms"][b_]["gain"] for r in R]; p = M7.signflip_p(d); fam.append((f"{a_}_minus_{b_}", p))
        S["contrasts"][f"{a_}_minus_{b_}"] = dict(mean=float(np.mean(d)), ci=M7.boot_ci(d), n_pos=int(sum(x > 0 for x in d)), p_signflip_one_sided=p)
    for (k, p), q in zip(fam, M7.holm([p for _, p in fam])): S["contrasts"][k]["p_holm"] = q
    if S["mode"] == "eventreg":
        d = [(r["arms"]["reve"]["gain"] - r["arms"]["reve_twin"]["gain"]) - (r["plain_m7"]["reve"] - r["plain_m7"]["reve_twin"]) for r in R]
        S["contrasts"]["dividend_eventreg_minus_plain"] = dict(mean=float(np.mean(d)), ci=M7.boot_ci(d), n_pos=int(sum(x > 0 for x in d)))
        S["plain_m7_dividend_same_participants"] = float(np.mean([r["plain_m7"]["reve"] - r["plain_m7"]["reve_twin"] for r in R]))
    if "dedicated" in R[0]:
        S["dedicated"] = {k: dict(gain=float(np.mean([r["dedicated"][k] for r in R])), ci=M7.boot_ci([r["dedicated"][k] for r in R])) for k in R[0]["dedicated"]}
    if "g2_max_abs_diff" in R[0]: S["g2_max_abs_diff"] = float(max(r["g2_max_abs_diff"] for r in R))
    ci = S["contrasts"]["reve_minus_reve_twin"]["ci"]
    S["falsifier"] = dict(rule="N-D: REVE - twin CI includes 0 on ds004595 OR on event-regressed ds003458 -> dividend specific to ds003458 / not robust to eye events",
                          ci_includes_0=bool(ci[0] <= 0 <= ci[1]), this_arm_triggers=bool(ci[0] <= 0 <= ci[1]))
    (out / "summary.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1)[:5000])


def main():
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    if A.summarize: return summarize(out)
    assert not (A.eventreg and A.data != "ds003458"), "--eventreg is defined for ds003458 only"
    ns = D2.s13b_targets(); qtil, subs, fpath, _ = ns["targets_" + A.data]()
    positions, mods = models()
    for s in ([int(x) for x in A.subjects.split(",")] if A.subjects else subs):
        if not (out / f"pe_sub-{s:03d}.json").exists(): subject(s, qtil, fpath, positions, mods, out)


if __name__ == "__main__":
    main()
