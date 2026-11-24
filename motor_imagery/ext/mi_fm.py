"""MI-FM: frozen EEG foundation-model rows for the MI card (EEGMMIDB, 4-class imagery), same primary design as mi_card v2.

Models  : CBraMod (external/CBraMod, torch.load weights_only=True), LaBraM and EEGPT (fm_zoo.load, braindecode ports, safetensors),
          REVE-base (external/reve, safetensors; loader convention of m7_latent.load_reve). Each with a RANDOM-INIT twin: same
          architecture, default init, torch.manual_seed(0), no weights loaded.
Input   : the mi_card cache (uV, -1..5 s at 160 Hz), mi_card.bandpass 1-40 Hz on the whole epoch (same band as the raw rung and the
          noise floor), FFT-resampled to the model rate on the whole epoch, then cropped to 0.5-4.0 s after the cue.
          CBraMod / LaBraM: 200 Hz, 1-s patches; the 3.5-s window is covered by two 3-patch crops (0.5-3.5 s, 1.0-4.0 s), uV/100.
          EEGPT: 256 Hz, 896 samples = 14 patches of 64, checkpoint 62-channel layout filled by name (zeros elsewhere), uV/100 (fm_zoo).
          REVE: 200 Hz, patch 200 / overlap 20, two crops of 560 samples (0.5-3.3 s, 1.2-4.0 s), per-participant per-channel z-score
          (label-free), clipped at +-15; electrode positions by name.
          Channels by name (LaBraM canonical list / EEGPT layout / REVE position bank); unmatched channels dropped (counts printed).
Pooling : all output tokens of all crops -> mean and s.d. over tokens (2 x embed dim).
Read-out: identical to mi_card primary_lpgo: 10 groups = index mod 10 over the sorted QC-passing list; per-participant column
          z-scoring (label-free), PCA-256 on outer-training rows, exact-Newton multinomial logistic (mi_card.primal_fits), (lambda, T)
          jointly on the inner validation group (g+1) mod 10, refit on all 9 groups; bits = log2 p(y) - log2 train prior.
Floors  : noise input = mi_card's noise (Gaussian 1-40 Hz, per-participant per-channel SD, same generator seed) through the same
          frozen model (and through the twin).
Outputs : <out>/<model>/sub-XXX.json, meta.json, summary.json (code_sha256, rows, FM - twin dividend, FM - raw when mi_card's
          per-participant files for the same participants exist); <out>/gate.json (smoke gate: floors within +-0.02, twin rows exist).
Usage   : python mi_fm.py --out results/ext/mi_fm [--subjects 1,2,3] [--models cbramod,labram,eegpt,reve] [--raw_dir DIR]
          python mi_fm.py --out results/ext/mi_fm --resummarize      (re-reads per-participant files; adds FM - raw when available)
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "external/CBraMod"):
    sys.path.insert(0, str(p))
os.chdir(ROOT)
from mi_card import (load_sub, bandpass, newton_logistic, primal_fits, select, bits, prior_of, boot, signflip_p, holm, LAMS, TGRID,  # noqa: F401,E402
                     W0, W1)

DEV = "cuda"
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:7.1f}s]", *m, flush=True)
ELIG = {
    "cbramod": ("disjoint", "Wang et al., CBraMod (ICLR 2025), arXiv:2412.07236, Sec. 3.1: pretrained on TUEG only; PhysioNet-MI is a "
                "downstream benchmark (also used in their ablations: model-selection exposure, not pretraining)"),
    "reve": ("disjoint", "El Ouahidi et al., REVE (NeurIPS 2025), arXiv:2510.21585, App. B exhaustive list (Table 7): MOABB list has no "
             "PhysionetMI; Physionet = Siena + I-CARE; OpenNeuro list has no EEGMMIDB; Other = NMT, HMS, SparrKULee, Inria Large, THINGS2, "
             "TDBRAIN, TUH. PhysioNet-MI is a downstream benchmark (also used in their ablations: model-selection exposure)"),
    "labram": ("transductive (pretrained on EEGMMIDB)", "Jiang et al., LaBraM (ICLR 2024), App. D: EEG Motor Movement/Imagery in the corpus"),
    "eegpt": ("transductive (pretrained on EEGMMIDB)", "Wang et al., EEGPT (NeurIPS 2024), Table 1: PhysioNet MI in the corpus"),
}
SPEC = {  # sample rate, crops (start, length) inside the 0.5-4.0 s window at that rate
    "cbramod": (200, [(0, 600), (100, 600)]), "labram": (200, [(0, 600), (100, 600)]),
    "eegpt": (256, [(0, 896)]), "reve": (200, [(0, 560), (140, 560)]),
}


# ---------------------------------------------------------------- models
def load_model(name, pretrained, ch):
    """-> (forward(x [B, C_in, T] model-rate crop) -> tokens [B, n, d], channel index list into ch, info dict)."""
    up = [c.upper() for c in ch]
    if name == "cbramod":
        from models.cbramod import CBraMod
        torch.manual_seed(0); m = CBraMod()
        if pretrained:
            m.load_state_dict(torch.load(ROOT / "external/CBraMod/pretrained_weights/pretrained_weights.pth", map_location="cpu", weights_only=True))
        m.proj_out = torch.nn.Identity(); m = m.to(DEV).eval()
        def fwd(x):
            B, C, T = x.shape; f = m(x.reshape(B, C, T // 200, 200)); return f.reshape(B, -1, f.shape[-1])
        return fwd, list(range(len(ch))), dict(channels=len(ch))
    if name in ("labram", "eegpt"):
        import fm_zoo
        import braindecode.models as bm
        n_times = SPEC[name][1][0][1]
        if pretrained:
            m, _ = fm_zoo.load(name, ch, n_times, DEV)
        else:                                              # same constructor arguments as fm_zoo.load, no weights
            cls, path, sf = fm_zoo.CFG[name]; cfg = json.load(open(f"{path}/config.json"))
            kw = {k: v for k, v in cfg.items() if k in ("chs_info", "n_chans", "n_outputs") and v is not None}
            kw.update(n_times=n_times, sfreq=sf)
            if name == "eegpt": kw.update(chan_proj_type="none")
            torch.manual_seed(0); m = getattr(bm, cls)(**kw).to(DEV).eval()
        if name == "labram":
            from braindecode.models.labram import _LABRAM_CANONICAL_INDEX
            idx = [i for i, c in enumerate(up) if c in _LABRAM_CANONICAL_INDEX]; names = [up[i] for i in idx]
            def fwd(x): return m(x, return_patch_tokens=True, ch_names=names)
            return fwd, idx, dict(channels=len(idx), dropped=[up[i] for i in range(len(up)) if i not in idx])
        names62 = [c["ch_name"].upper() for c in json.load(open(ROOT / "external/eegpt/config.json"))["chs_info"]]
        idx = [i for i, c in enumerate(up) if c in names62]; slot = [names62.index(up[i]) for i in idx]
        def fwd(x):
            X62 = torch.zeros(x.shape[0], len(names62), x.shape[2], device=DEV); X62[:, slot] = x
            t = fm_zoo.tokens(m, X62, "eegpt", None)
            if isinstance(t, (tuple, list)): t = t[0]
            return t.reshape(t.shape[0], -1, t.shape[-1])
        return fwd, idx, dict(channels=len(idx), of_layout=len(names62), dropped=[up[i] for i in range(len(up)) if i not in idx])
    if name == "reve":                                     # m7_latent.load_reve convention
        from transformers import AutoModel, AutoConfig
        positions = AutoModel.from_pretrained(str(ROOT / "external/reve/positions"), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(DEV).eval()
        if pretrained:
            m = AutoModel.from_pretrained(str(ROOT / "external/reve/base"), trust_remote_code=True, local_files_only=True, use_safetensors=True)
        else:
            torch.manual_seed(0); cfg = AutoConfig.from_pretrained(str(ROOT / "external/reve/base"), trust_remote_code=True, local_files_only=True)
            m = AutoModel.from_config(cfg, trust_remote_code=True)
        m = m.to(DEV).eval(); bank = {k.upper(): k for k in positions.mapping}
        idx = [i for i, c in enumerate(up) if c in bank]; pos = positions([bank[up[i]] for i in idx]).to(DEV); assert pos.shape == (len(idx), 3)
        def fwd(x):
            f = m(x, pos.expand(len(x), -1, -1)); return f.reshape(len(x), -1, f.shape[-1])
        return fwd, idx, dict(channels=len(idx), dropped=[up[i] for i in range(len(up)) if i not in idx])
    raise ValueError(name)


def resample(x, n_new):
    """FFT resampling of a whole (band-limited, 1-40 Hz) epoch to n_new samples."""
    n = x.shape[-1]; Y = torch.fft.rfft(x, dim=-1); Z = torch.zeros(*x.shape[:-1], n_new // 2 + 1, dtype=Y.dtype, device=x.device)
    k = min(Y.shape[-1], Z.shape[-1]); Z[..., :k] = Y[..., :k]
    return torch.fft.irfft(Z, n=n_new, dim=-1) * (n_new / n)


@torch.no_grad()
def extract(name, pretrained, Xs, sid, zstat, ch, bs=128):
    """Xs: CPU float32 [N, 64, 960] (1-40 Hz, uV). Returns pooled features [N, 2d] float64 on the GPU + info."""
    fwd, idx, info = load_model(name, pretrained, ch); sf, crops = SPEC[name]; n_new = 960 * sf // 160; s0 = int(1.5 * sf)
    out = []
    for b in range(0, len(Xs), bs):
        x = Xs[b:b + bs].to(DEV)
        if name == "reve":
            mu, sd = zstat[0][sid[b:b + bs]], zstat[1][sid[b:b + bs]]; x = (x - mu[..., None]) / sd[..., None]
        else:
            x = x / 100.0
        x = resample(x[:, idx], n_new)[:, :, s0:s0 + int(3.5 * sf)]
        if name == "reve": x = x.clamp(-15, 15)
        T = torch.cat([fwd(x[:, :, a:a + L].contiguous()) for a, L in crops], 1).double()
        out.append(torch.cat([T.mean(1), T.std(1)], 1))
    F = torch.cat(out); assert torch.isfinite(F).all(), f"{name}: non-finite features"
    info.update(tokens_per_trial=int(T.shape[1]), embed_dim=int(T.shape[2]), sample_rate=sf, crops_s=[[round(0.5 + a / sf, 3), round(0.5 + (a + L) / sf, 3)] for a, L in crops])
    del fwd; torch.cuda.empty_cache()
    return F, info


# ---------------------------------------------------------------- read-out (mi_card primary_lpgo, logistic rungs)
def score(F, y, sid, nP, fits, units, pca_rows):
    Z = torch.empty_like(F)
    for k in range(nP):
        r = sid == k; Z[r] = (F[r] - F[r].mean(0)) / (F[r].std(0) + 1e-8)
    ylab = [y[f[0]] for f in fits]
    Zs, gm, ev = primal_fits(Z, fits, ylab, LAMS, pca_rows)
    N = len(y); bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV)
    lr = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ne = [0, 0]
    for u in units:
        Zi = torch.cat([Zs[i] for i in u["inner"]], 1); yi = torch.cat([y[fits[i][1]] for i in u["inner"]])
        l, T, e = select(Zi, yi); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]
        f = u["final"]; evr = fits[f][1]; Zf = Zs[f][l]
        bt[evr] = bits(Zf, y[evr], T, prior_of(ylab[f])); ok[evr] = (Zf.argmax(-1) == y[evr]).double()
        m = y[evr] < 2; lr[evr[m]] = (Zf[m][:, :2].argmax(-1) == y[evr][m]).double()
    assert not torch.isnan(bt).any(), "untested trials"
    per = [dict(bits=float(bt[sid == k].mean()), acc=float(ok[sid == k].mean()), lr_acc=float(lr[sid == k].nanmean()), n_test=int((sid == k).sum())) for k in range(nP)]
    return per, dict(max_grad=gm, lambda_edge=ne[0], T_edge=ne[1], n_units=len(units), pca_ev=ev)


def paired(a, b):
    d = [x - z for x, z in zip(a, b)]
    return dict(d=float(np.mean(d)), ci=boot(d), n_pos=int(sum(x > 0 for x in d)), n=len(d), p_signflip=signflip_p(d))


def summarize(mdir, raw_dir):
    meta = json.loads((mdir / "meta.json").read_text()); P = meta["participants"]
    R = [json.loads((mdir / f"sub-{s:03d}.json").read_text()) for s in P]
    S = dict(meta, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows={})
    for arm in ("fm", "twin"):
        for cond in ("eeg", "noise"):
            v = [r[arm][cond]["bits"] for r in R]
            row = dict(mean=float(np.mean(v)), ci=boot(v), acc=float(np.mean([r[arm][cond]["acc"] for r in R])),
                       lr_acc=float(np.mean([r[arm][cond]["lr_acc"] for r in R])))
            if cond == "noise": row["g3"] = dict(n_abs_le_0p02=int(sum(abs(x) <= 0.02 for x in v)), n=len(v))
            S["rows"].setdefault(arm, {})[cond] = row
    S["dividend_fm_minus_twin"] = paired([r["fm"]["eeg"]["bits"] for r in R], [r["twin"]["eeg"]["bits"] for r in R])
    S["floors_ok"] = all(abs(S["rows"][a]["noise"]["mean"]) <= 0.02 for a in ("fm", "twin"))
    rd = Path(raw_dir); rs = rd / "summary_mi.json"
    if rs.exists() and json.loads(rs.read_text()).get("participants") == P and all((rd / f"sub-{s:03d}.json").exists() for s in P):
        raw = [json.loads((rd / f"sub-{s:03d}.json").read_text())["raw"]["eeg"]["bits"] for s in P]
        S["fm_minus_raw"] = dict(paired([r["fm"]["eeg"]["bits"] for r in R], raw), raw_source=str(rd), raw_code_sha256=json.loads(rs.read_text()).get("code_sha256"))
        S["twin_minus_raw"] = dict(paired([r["twin"]["eeg"]["bits"] for r in R], raw), raw_source=str(rd))
        S["raw_mean"] = float(np.mean(raw))
    else:
        S["fm_minus_raw"] = dict(status=f"pending: no mi_card primary_lpgo results for the same participants in {rd}")
    (mdir / "summary.json").write_text(json.dumps(S, indent=1))
    return S


def gate(out, models, raw_dir):
    G = dict(models={}, pass_=True)
    for mname in models:
        f = out / mname / "summary.json"
        if not f.exists(): G["models"][mname] = dict(exists=False); G["pass_"] = False; continue
        S = json.loads(f.read_text()); tw = "twin" in S["rows"] and "eeg" in S["rows"]["twin"]
        G["models"][mname] = dict(exists=True, floors_ok=S["floors_ok"], twin_rows=tw, fm_eeg=S["rows"]["fm"]["eeg"]["mean"],
                                  twin_eeg=S["rows"]["twin"]["eeg"]["mean"], fm_noise=S["rows"]["fm"]["noise"]["mean"],
                                  twin_noise=S["rows"]["twin"]["noise"]["mean"], dividend=S["dividend_fm_minus_twin"]["d"],
                                  fm_minus_raw=S["fm_minus_raw"].get("d"), label=S["label"])
        G["pass_"] &= bool(S["floors_ok"] and tw)
    done = [m for m in models if G["models"][m].get("exists")]
    if done:
        for key in ("dividend_fm_minus_twin", "fm_minus_raw"):
            ps = [json.loads((out / m / "summary.json").read_text())[key].get("p_signflip") for m in done]
            if all(p is not None for p in ps):
                for m, ph in zip(done, holm(ps)): G["models"][m][f"{key}_p_holm4"] = ph
    G["pass"] = G.pop("pass_"); G["criteria"] = "every model: summary exists, |FM noise| and |twin noise| group means <= 0.02 bits, twin rows exist"
    (out / "gate.json").write_text(json.dumps(G, indent=1)); log("GATE", json.dumps(G, indent=1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--subjects", default="")
    ap.add_argument("--models", default="cbramod,labram,eegpt,reve"); ap.add_argument("--raw_dir", default=str(ROOT / "results/ext/mi_card/primary_lpgo"))
    ap.add_argument("--resummarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); models = a.models.split(",")
    if a.resummarize:
        for mname in models:
            if (out / mname / "meta.json").exists(): summarize(out / mname, a.raw_dir)
        gate(out, models, a.raw_dir)
        if all("d" in json.loads((out / m / "summary.json").read_text())["fm_minus_raw"] for m in models if (out / m / "summary.json").exists()):
            (out / "raw_merged.json").write_text(json.dumps(dict(raw_dir=a.raw_dir, models=models)))
        return
    cache = ROOT / "results/ext/mi_card_cache"; subs = [int(x) for x in a.subjects.split(",")] if a.subjects else list(range(1, 110))
    D, excl = {}, {}
    for s in subs:                                         # same QC as mi_card
        d, why = load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d["y"], minlength=4)
        if c.max() - c.min() > 2: excl[s] = f"class counts {c.tolist()} differ by > 2"; continue
        D[s] = d
    P = sorted(D); nP = len(P); ch = D[P[0]]["ch"]; assert all(D[s]["ch"] == ch for s in P); log(f"{nP} participants, excluded {excl}")
    X = torch.tensor(np.concatenate([D[s]["X"] for s in P]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([D[s]["y"] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D[s]["y"]) for k, s in enumerate(P)]), device=DEV); N = len(y); sidc = sid.cpu()
    Xb = torch.empty_like(X); Nz = torch.empty_like(X)
    for i in range(0, N, 1500): Xb[i:i + 1500] = bandpass(X[i:i + 1500].to(DEV), 1, 40).cpu()
    del X
    g = torch.Generator(device=DEV); g.manual_seed(20261121); R_ = torch.randn((N, 64, 960), generator=g, device=DEV)   # = mi_card noise draw
    for i in range(0, N, 1500): Nz[i:i + 1500] = bandpass(R_[i:i + 1500], 1, 40).cpu()
    del R_; torch.cuda.empty_cache()
    for k in range(nP):
        r = sidc == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * Xb[r].std((0, 2), keepdim=True)
    zst = {}
    for nm, XX in (("eeg", Xb), ("noise", Nz)):            # per-participant per-channel window stats (label-free; REVE input)
        mu = torch.stack([XX[sidc == k][:, :, W0:W1].mean((0, 2)) for k in range(nP)]).to(DEV)
        sd = torch.stack([XX[sidc == k][:, :, W0:W1].std((0, 2)) for k in range(nP)]).to(DEV) + 1e-6
        zst[nm] = (mu, sd)
    G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
    for k in range(nP):
        c = torch.bincount(y[sid == k], minlength=4); assert int(c.max() - c.min()) <= 2
    fits, units, pca_rows = [], [], {}
    for gg in range(G):
        v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
        fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg))
        units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
    for mname in models:
        mdir = out / mname; mdir.mkdir(parents=True, exist_ok=True); per = [dict(participant=s, group=k % G) for k, s in enumerate(P)]; diag, info = {}, {}
        for arm, pre in (("fm", True), ("twin", False)):
            for cond, XX in (("eeg", Xb), ("noise", Nz)):
                F, inf = extract(mname, pre, XX, sid, zst[cond], ch); info[f"{arm}|{cond}"] = inf
                pp, dg = score(F, y, sid, nP, fits, units, pca_rows); diag[f"{arm}|{cond}"] = dg; del F; torch.cuda.empty_cache()
                for k in range(nP): per[k].setdefault(arm, {})[cond] = pp[k]
                log(f"{mname} {arm} {cond}: dim {2 * inf['embed_dim']}, tokens {inf['tokens_per_trial']}, mean bits {np.mean([q['bits'] for q in pp]):+.4f} "
                    f"acc {np.mean([q['acc'] for q in pp]):.3f}; edges lambda {dg['lambda_edge']}/{dg['n_units']} T {dg['T_edge']}")
        for k, s in enumerate(P): (mdir / f"sub-{s:03d}.json").write_text(json.dumps(per[k], indent=1))
        meta = dict(experiment="MI-FM: frozen EEG foundation model on the MI card (EEGMMIDB, 4-class imagery), primary_lpgo design",
                    model=mname, label=ELIG[mname][0], eligibility_source=ELIG[mname][1], n=nP, participants=P, groups=G,
                    excluded={str(k): v for k, v in excl.items()}, window="0.5-4.0 s after the cue", unit="bits per trial (log2 p - log2 train prior)",
                    input="mi_card.bandpass 1-40 Hz, FFT-resampled whole epoch, cropped; see SPEC/info", info=info, diag=diag,
                    pooling="mean and s.d. over all output tokens of all crops", twin="same architecture, default init, torch.manual_seed(0)",
                    readout="per-participant z-scoring -> PCA-256 (outer-training rows) -> Newton logistic; (lambda, T) on group (g+1) mod G; refit on all other groups",
                    noise="mi_card noise input (Gaussian 1-40 Hz, per-participant per-channel SD, seed 20261121) through the same frozen model",
                    mi_card_sha256=hashlib.sha256((ROOT / "experiments/ext/mi_card.py").read_bytes()).hexdigest(), runtime_s=round(time.time() - t0))
        (mdir / "meta.json").write_text(json.dumps(meta, indent=1)); S = summarize(mdir, a.raw_dir)
        log(mname, json.dumps({k: S[k] for k in ("label", "rows", "dividend_fm_minus_twin", "fm_minus_raw", "floors_ok")}, indent=1))
    gate(out, models, a.raw_dir)


if __name__ == "__main__":
    main()
