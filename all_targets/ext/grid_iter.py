"""GRID & THRESHOLD ITERATION, round 1 grid variants (METHOD 05:55; rung R5). EXECUTOR-2, 21 Nov 2026. GPU only (PLAN §10-12).
Dev participants only (image sub-01..03, PE sub-001..003); no test-row selection, no other participants.
Every variant reuses the M3 path (m3_yield_map: Morlet power on the whole epoch averaged over repetitions, zero-phase band-passed voltage,
train-stat standardisation, ridge, locked score, floors) and changes ONE axis:
  G1  window width 50 / 100 / 200 ms     (combined cells, 5 bands, 8 named regions)
  G2  cell kind voltage / power / combined (100 ms, 5 bands, 8 named regions)
  G3  band set 5 bands vs 3 (delta-theta 1-8, alpha 8-13, beta-gamma 13-45 Hz) vs broadband voltage only (100 ms, 8 named regions)
  G4  space 8 named regions vs the k-d tree depth-3 nodes (experiments/ext/regions_kdtree.json; 7-8 channels) (100 ms combined, 5 bands)
  (G5 CBraMod token grid: needs EXECUTOR-4's token cache; not in this script.)
Windows are defined by sample edges round(linspace(epoch start, end, n + 1)); voltage = 5 bins per window by the same rule. The 100-ms
5-band named-region combined variant is the M3 grid exactly (G2: equal to results/ext/m3 cells within 1e-4 image / 1e-9 PE where the M3
file exists). 3-band power = mean of the constituent bands' linear power (each band averages 3 frequencies, so equal frequency weights);
3-band voltage is re-filtered with the merged band edges.
Per variant and participant: every cell's yield and floor (image: noise + shuffled pairs; PE: circular shifts + noise), and the UNION
yield = one ridge on all cells' features concatenated (image: dual/kernel ridge when k > N, alpha by the M3 dev rule; PE: dual form of
pe_gain), against the broadband all-channel reference (the M3 broadband cell).
Selection rule (METHOD, fixed before the smokes): grid winner = the FINEST grid whose union keeps >= 95 % of the broadband all-channel
yield AND whose peak cell is stable (participant-bootstrap peak frequency >= 0.5 on the dev participants); ties -> simpler grid.
Output results/smoke_feasibility/grid_iter/<target>_<sub>.json and summary_<target>.json (with per-variant runtime).
ROUND 2 (METHOD ROUND 1 RULING 08:10): kind-matched broadband references (voltage = the M3 broadband cell; power = all-channel
whole-epoch band power, 63/64 ch x 5 bands; combined = both in one ridge); unions recomputed WITHOUT flagged cells (+ a flagged-only
union); within-participant map reliability (Spearman-Brown of the cell maps from odd vs even outer folds (PE) / test queries (image))
replaces the peak-frequency rule; variants G1 50/100/200 combined, G2 voltage, G3 3 vs 5 bands, named regions only.
Usage: python grid_iter.py --target image|pe --subjects 1,2,3 --out DIR ; python grid_iter.py --target image --out DIR --summarize
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import signal

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import m3_yield_map as M
import regions as RG
DEV = "cuda"
MERGED = [(1, 8), (8, 13), (13, 45)]


def edges(t0, t1, n):
    return np.round(np.linspace(t0, t1, n + 1)).astype(int)


def wfeats(pw, vo, e):
    """pw (N, C, B, T) linear power; vo (N, C, B, T) voltage; window edges e -> P (N, C, B, W) window-mean power, V (N, C, B, W, 5)."""
    P = torch.stack([pw[..., a:b].mean(-1) for a, b in zip(e[:-1], e[1:])], -1)
    V = torch.stack([torch.stack([vo[..., x:y].mean(-1) for x, y in zip(edges(a, b, 5)[:-1], edges(a, b, 5)[1:])], -1) for a, b in zip(e[:-1], e[1:])], 3)
    return P, V


def merged_gains(sf, nfft):
    f = np.fft.fftfreq(nfft, 1 / sf); g = []
    for lo, hi in MERGED:
        sos = signal.butter(4, [lo, hi], btype="band", fs=sf, output="sos"); _, h = signal.sosfreqz(sos, worN=np.abs(f), fs=sf); g.append(np.abs(h) ** 2)
    return torch.tensor(np.array(g), dtype=torch.float32, device=DEV)


def band_voltage_g(x, G, nfft):
    N, C, T = x.shape; P_ = T - 1; nb = G.shape[0]
    xp = torch.nn.functional.pad(x.reshape(N * C, 1, T), (P_, P_), mode="reflect").reshape(N * C, -1); X = torch.fft.fft(xp, nfft)
    return torch.stack([torch.fft.ifft(X * G[b][None], nfft).real[:, P_:P_ + T] for b in range(nb)], 1).reshape(N, C, nb, T)


def feature_sets(X_items, reps, sf, t0, t1, specs):
    """Returns {spec_ms: dict(P5, V5, P3, V3, Vbb)} on the GPU (float32); X_items: (items, reps, C, T) numpy (reps = 1 for PE)."""
    T = X_items.shape[-1]; Mw, nfft, Wf = M.morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); G5 = M.bandpass_gains(sf, nfv); G3 = merged_gains(sf, nfv)
    out = {ms: dict(P5=[], V5=[], P3=[], V3=[], Vbb=[]) for ms in specs}; step = max(1, (800 if reps > 1 else 64) // reps)   # PE: 64-trial chunks (shared GPU)
    for i in range(0, len(X_items), step):
        xb = torch.tensor(np.ascontiguousarray(X_items[i:i + step]), device=DEV, dtype=torch.float32); nI = xb.shape[0]
        pw = M.band_power(xb.reshape(nI * reps, *xb.shape[2:]), sf, Mw, nfft, Wf).reshape(nI, reps, xb.shape[2], 5, T).mean(1)
        xm = xb.mean(1); v5 = M.band_voltage(xm, sf, G5, nfv); v3 = band_voltage_g(xm, G3, nfv)
        pw3 = torch.stack([pw[:, :, 0:2].mean(2), pw[:, :, 2], pw[:, :, 3:5].mean(2)], 2)
        for ms, n in specs.items():
            e = edges(t0, t1, n); P5, V5 = wfeats(pw, v5, e); P3, V3 = wfeats(pw3, v3, e); _, Vb = wfeats(pw[:, :, :1], xm[:, :, None], e)
            for k, v in (("P5", torch.log10(P5 + 1e-20)), ("V5", V5), ("P3", torch.log10(P3 + 1e-20)), ("V3", V3), ("Vbb", Vb)): out[ms][k].append(v)
    return {ms: {k: torch.cat(v) for k, v in d.items()} for ms, d in out.items()}


def variants(target, rnd=2):
    """(name, spec_ms, band set, kind, space). Round 2 (METHOD ROUND 1 RULING): broadband-only, power-only and the k-d grid are dropped."""
    if rnd == 3:   # METHOD ROUND 2 RULING (PE): final smoke w50 x 3 bands, combined; fallback w100 x 3 bands
        return [("R3_w50_3bands", 50, "3", "combined", "named"), ("R3_w100_3bands", 100, "3", "combined", "named")]
    v = [(f"G1_w{w}", w, "5", "combined", "named") for w in (50, 100, 200)]
    v += [("G2_voltage", 100, "5", "voltage", "named"), ("G3_3bands", 100, "3", "combined", "named")]
    if rnd == 1:
        v += [("G2_power", 100, "5", "power", "named"), ("G3_broadband_voltage", 100, "bb", "voltage", "named"), ("G4_kdtree_d3", 100, "5", "combined", "kd3")]
    return v


FWHM3 = None
ROUND = 2


def flag_of(bset, kind, b, t_ms):
    """Power-containing cell whose window starts < FWHM/2 after the epoch start (image) / feedback (PE); 3-band cells use the widest
    constituent band (delta for delta-theta, beta for beta-gamma). Voltage cells are never flagged."""
    fw = M.fwhm_ms(); f = (fw[M.BNAMES[b]] if bset == "5" else [fw["delta"], fw["alpha"], fw["beta"]][b])
    return bool(kind != "voltage" and t_ms < f / 2)


def cell_tensor(F, sets, bset, kind, W):
    """All cells of a variant as a list of (meta, (N, k) tensor) grouped later; F = feature dict for the spec."""
    P, V = (F["P5"], F["V5"]) if bset == "5" else (F["P3"], F["V3"]) if bset == "3" else (None, F["Vbb"])
    nb = V.shape[2]; cells = []
    for rname, chs in sets.items():
        ch = torch.as_tensor(chs, device=DEV)
        for b in range(nb):
            for w in range(W):
                if kind == "power": f = P[:, ch, b, w]
                elif kind == "voltage": f = V[:, ch, b, w, :].flatten(1)
                else: f = torch.cat([P[:, ch, b, w], V[:, ch, b, w, :].flatten(1)], 1)
                cells.append(((rname, b, w), f))
    return cells


def kd3_sets(target, ch):
    t = json.loads((ROOT / "experiments/ext/regions_kdtree.json").read_text())[target]["fixed"]
    return {f"kd{k}": RG.indices(ch, v) for k, v in t.items() if len(k) == 3}


# ------------------------------------------------------------------ image
def image_union(Ztr, Zte, Yg, Yng, R, chunk=2048):
    """Ridge on all features at once in dual/kernel form (k may exceed N), alpha by the M3 dev rule, test InfoNCE. Memory-lean: features in
    fp32, the kernel accumulated in fp64 over column chunks (same algebra as before: K = A A', predictions K_q,tr (K_tr + a I)^-1 (Y - ym))."""
    N = len(Ztr); idx = np.random.default_rng(0).permutation(N); cut = int(0.9 * N); tr, va = torch.as_tensor(idx[:cut], device=DEV), torch.as_tensor(idx[cut:], device=DEV)
    Ztr = torch.as_tensor(Ztr, device=DEV).float(); Zte = torch.as_tensor(Zte, device=DEV).float()
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, unbiased=False, keepdim=True) + 1e-6; A = (Ztr - mu) / sd; B = (Zte - mu) / sd; del Ztr, Zte
    K = torch.zeros((N, N), dtype=torch.float64, device=DEV); Kq = torch.zeros((len(B), N), dtype=torch.float64, device=DEV)
    for c in range(0, A.shape[1], chunk):
        a_ = A[:, c:c + chunk].double(); K += a_ @ a_.T; Kq += B[:, c:c + chunk].double() @ a_.T; del a_
    k_dim = int(A.shape[1]); del A, B; torch.cuda.empty_cache()
    def fitpred(Ktt, Kqt, Ytr, a):
        ym = Ytr.mean(0); return Kqt @ torch.linalg.solve(Ktt + a * torch.eye(len(Ktt), dtype=Ktt.dtype, device=DEV), Ytr - ym) + ym
    T = Yg[va]; T = T / T.norm(dim=1, keepdim=True); best, ba = -1, None; Ktt = K[tr][:, tr]; Kvt = K[va][:, tr]
    for a in R.ALPHAS:
        pv = fitpred(Ktt, Kvt, Yg[tr], a); pv = pv / pv.norm(dim=1, keepdim=True); S = pv @ T.T
        m = float((1.0 / ((S > S.diag()[:, None]).sum(1) + 1)).mean())
        if m > best: best, ba = m, a
    del Ktt, Kvt
    p = fitpred(K, Kq, Yg, ba); p = p / p.norm(dim=1, keepdim=True); del K
    y = float(M.soft_bits_gpu((p @ Yng.T)[None])[0]); sh = float(M.soft_bits_gpu((p @ Yng[torch.as_tensor(np.random.default_rng(200).permutation(200), device=DEV)].T)[None])[0])
    return dict(yield_=y, shuffled=sh, alpha=ba, k=k_dim)


def wait_gpu(gb=12.0):
    """Rule 7 guard: wait until the GPU has >= gb GB free."""
    while torch.cuda.mem_get_info()[0] < gb * 1e9:
        print("waiting for GPU memory", round(torch.cuda.mem_get_info()[0] / 1e9, 1), "GB free", flush=True); time.sleep(60)


def spearman_sb(a, b):
    """Spearman of two cell maps, Spearman-Brown corrected (split-half -> full-data map reliability)."""
    ra = torch.argsort(torch.argsort(torch.as_tensor(a))).double(); rb = torch.argsort(torch.argsort(torch.as_tensor(b))).double()
    ra, rb = ra - ra.mean(), rb - rb.mean(); r = float((ra @ rb) / torch.sqrt((ra @ ra) * (rb @ rb))); return r, 2 * r / (1 + r) if r > -1 else float("nan")


def image_subject(s, out):
    import run as R
    t0 = time.time(); sub = f"sub-{s:02d}"; C = ROOT / "data_raw/things_eeg2_250hz/cache_f32"
    if (C / f"{sub}_train.npy").exists():   # float32 memory-mapped cache (things_f32_cache.py): low host memory on cluster
        Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
        Xtr4, Xte80 = np.load(C / f"{sub}_train.npy", mmap_mode="r"), np.load(C / f"{sub}_test.npy", mmap_mode="r")
        ch = json.loads((ROOT / "results/eegbits/raw_sub-01.json").read_text())["raw_meta"]["ch_names"]
    else:
        sub, Ytr, Yte, Xtr4, Xte80, ch = M.image_load(s, R)
    specs = {50: 20, 100: 10, 200: 5, 1000: 1}
    Ftr = feature_sets(Xtr4, 4, 250.0, 0, 250, specs); Fte = feature_sets(Xte80, 80, 250.0, 0, 250, specs)
    print(sub, "features", f"{time.time()-t0:.0f}s", flush=True)
    _, full, Yn = M.image_scorer(Ytr, Yte, R); Yg = torch.tensor(Ytr, dtype=torch.float64, device=DEV); Yng = torch.as_tensor(Yn, dtype=torch.float64, device=DEV)
    bbtr, bbte = M.image_broadband(Xtr4, Xte80); del Xtr4, Xte80
    pw = lambda F: F[1000]["P5"].flatten(1)                                           # whole-epoch band power, 63 ch x 5 bands
    refs = dict(voltage=M.image_batch(torch.as_tensor(bbtr)[None], torch.as_tensor(bbte)[None], Yg, Yng, R, 1)[0],
                power=M.image_batch(pw(Ftr)[None], pw(Fte)[None], Yg, Yng, R, 1)[0])
    refs["combined"] = image_union(torch.cat([torch.as_tensor(bbtr, device=DEV), pw(Ftr)], 1), torch.cat([torch.as_tensor(bbte, device=DEV), pw(Fte)], 1), Yg, Yng, R)
    print(sub, "refs", {k: round(v["yield_"], 3) for k, v in refs.items()}, f"{time.time()-t0:.0f}s", flush=True)
    named = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}; res = dict(target="image", participant=sub, refs=refs, variants={})
    m3f = ROOT / f"results/ext/m3/image_{sub}.json"; m3 = {(c["region"], c["band"], c["t_ms"]): c["yield_"] for c in json.loads(m3f.read_text())["cells"] if c["kind"] == "combined"} if m3f.exists() else None
    odd = torch.arange(200) % 2 == 1
    for name, ms, bset, kind, space in variants("image", ROUND):
        tv = time.time(); sets = named; W = specs[ms]; e = edges(0, 250, W)
        ctr = cell_tensor(Ftr[ms], sets, bset, kind, W); cte = cell_tensor(Fte[ms], sets, bset, kind, W); cells = []
        for j in range(0, len(ctr), 40):
            r = M.image_batch(torch.stack([f for _, f in ctr[j:j + 40]]), torch.stack([f for _, f in cte[j:j + 40]]), Yg, Yng, R, 1, mem=6e8, per_query=True)
            for (meta, _), x in zip(ctr[j:j + 40], r):
                pq = x.pop("pq"); t_ms = 4 * int(e[meta[2]])
                cells.append(dict(region=meta[0], band=meta[1], w=meta[2], t_ms=t_ms, flag=flag_of(bset, kind, meta[1], t_ms), y_odd=float(pq[odd].mean()), y_even=float(pq[~odd].mean()), **x))
        fl = [i for i, c in enumerate(cells) if c["flag"]]; uf = [i for i, c in enumerate(cells) if not c["flag"]]
        un = image_union(torch.cat([ctr[i][1] for i in uf], 1), torch.cat([cte[i][1] for i in uf], 1), Yg, Yng, R)
        unf = image_union(torch.cat([ctr[i][1] for i in fl], 1), torch.cat([cte[i][1] for i in fl], 1), Yg, Yng, R) if fl else None
        rel = spearman_sb([c["y_odd"] for c in cells], [c["y_even"] for c in cells])
        g2 = max(abs(c["yield_"] - m3[(c["region"], M.BNAMES[c["band"]], c["t_ms"])]) for c in cells) if (name == "G1_w100" and m3 is not None) else None
        res["variants"][name] = dict(spec_ms=ms, bands=bset, kind=kind, space=space, n_cells=len(cells), n_flagged=len(fl), cells=cells, union_unflagged=un, union_flagged=unf,
                                     ref_kind=("voltage" if kind == "voltage" else "combined"), map_reliability=dict(r_half=rel[0], sb=rel[1]), g2_vs_m3=g2, runtime_s=round(time.time() - tv, 1))
        rk = refs["voltage" if kind == "voltage" else "combined"]["yield_"]
        print(sub, name, f"cells {len(cells)} (flagged {len(fl)}) union {un['yield_']:.3f} / ref {rk:.3f} = {un['yield_'] / rk:.2f}; flagged-only {unf['yield_'] if unf else None}; map rel SB {rel[1]:.2f}; g2 {g2}; {time.time()-tv:.0f}s", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"image_{sub}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ PE
def pe_union(F, Ys, LAM):
    """Dual-form pe_gain on all features (same folds, z-scoring, LAM * n_train, nested calibration)."""
    from belief_bottleneck import folds
    LN2 = np.log(2); Fg = torch.as_tensor(F, dtype=torch.float64, device=DEV); Y = torch.as_tensor(Ys, dtype=torch.float64, device=DEV); n = len(Y); fo = folds(n)
    def pred(tr, te):
        A = Fg[tr]; mu, sd = A.mean(0, keepdim=True), A.std(0, unbiased=False, keepdim=True) + 1e-6; Za = (A - mu) / sd; Zb = (Fg[te] - mu) / sd
        Yc = Y[tr] - Y[tr].mean(0); return Zb @ (Za.T @ torch.linalg.solve(Za @ Za.T + LAM * len(tr) * torch.eye(len(tr), dtype=torch.float64, device=DEV), Yc))
    sc = torch.zeros_like(Y)
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; P = pred(tr, te); inner = torch.zeros((len(tr), Y.shape[1]), dtype=torch.float64, device=DEV); fi = folds(len(tr), 9)
        for c in range(9): inner[torch.as_tensor(fi == c, device=DEV)] = pred(tr[fi != c], tr[fi == c])
        yt, ye = Y[tr], Y[te]; xm, ym = inner.mean(0), yt.mean(0); slope = ((inner - xm) * (yt - ym)).sum(0) / ((inner - xm) ** 2).sum(0); icpt = ym - slope * xm
        rv = ((yt - icpt - slope * inner) ** 2).mean(0); vv = ((yt - ym) ** 2).mean(0)
        sc[te] = (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * P) ** 2 / rv) / (2 * LN2)
    m = sc.mean(0).cpu().numpy(); return dict(yield_=float(m[0]), shift=float(m[1:].mean()), k=int(F.shape[1]))


def pe_subject(s, out, evaluate, fpath):
    from belief_bottleneck import LAM, folds
    t0 = time.time(); sub, fz, tri, n, Ys, X, ch = M.pe_load(s, evaluate, fpath)
    specs = {50: 12, 100: 6, 200: 3, 600: 1}; F = feature_sets(X[:, None], 1, 200.0, 200, 320, specs)
    bb = M.pe_broadband(X); pw = F[600]["P5"].flatten(1).double().cpu().numpy()
    refs = dict(voltage=pe_union(bb, Ys, LAM), power=pe_union(pw, Ys, LAM), combined=pe_union(np.concatenate([bb, pw], 1), Ys, LAM))
    print(sub, "refs", {k: round(v["yield_"], 4) for k, v in refs.items()}, flush=True)
    named = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}; res = dict(target="pe", participant=sub, refs=refs, variants={})
    m3f = ROOT / f"results/ext/m3/pe_{sub}.json"; m3 = {(c["region"], c["band"], c["t_ms"]): c["yield_"] for c in json.loads(m3f.read_text())["cells"] if c["kind"] == "combined"} if m3f.exists() else None
    rng = np.random.default_rng(1000 + s); odd = folds(n) % 2 == 1
    for name, ms, bset, kind, space in variants("pe", ROUND):
        tv = time.time(); W = specs[ms]; e = edges(200, 320, W)
        cl = cell_tensor(F[ms], named, bset, kind, W); Fb = torch.stack([f for _, f in cl]).double().cpu().numpy()
        E = M.pe_gain_batch(Fb, Ys, LAM); Nz = M.pe_gain_batch(rng.standard_normal(Fb.shape), Ys[:, :1], LAM); cells = []
        for k_, (meta, _) in enumerate(cl):
            sh, nv = float(E[k_, :, 1:].mean()), float(Nz[k_, :, 0].mean()); t_ms = 5 * int(e[meta[2]] - 200)
            cells.append(dict(region=meta[0], band=meta[1], w=meta[2], t_ms=t_ms, flag=flag_of(bset, kind, meta[1], t_ms), yield_=float(E[k_, :, 0].mean()), shift=sh, noise=nv,
                              floor=max(sh, nv), y_odd=float(E[k_, odd, 0].mean()), y_even=float(E[k_, ~odd, 0].mean())))
        fl = [i for i, c in enumerate(cells) if c["flag"]]; uf = [i for i, c in enumerate(cells) if not c["flag"]]
        un = pe_union(Fb[uf].transpose(1, 0, 2).reshape(n, -1), Ys, LAM); unf = pe_union(Fb[fl].transpose(1, 0, 2).reshape(n, -1), Ys, LAM) if fl else None
        rel = spearman_sb([c["y_odd"] for c in cells], [c["y_even"] for c in cells])
        g2 = max(abs(c["yield_"] - m3[(c["region"], M.BNAMES[c["band"]], c["t_ms"])]) for c in cells) if (name == "G1_w100" and m3 is not None) else None
        res["variants"][name] = dict(spec_ms=ms, bands=bset, kind=kind, space=space, n_cells=len(cells), n_flagged=len(fl), cells=cells, union_unflagged=un, union_flagged=unf,
                                     ref_kind=("voltage" if kind == "voltage" else "combined"), map_reliability=dict(r_half=rel[0], sb=rel[1]), g2_vs_m3=g2, runtime_s=round(time.time() - tv, 1))
        rk = refs["voltage" if kind == "voltage" else "combined"]["yield_"]
        print(sub, name, f"cells {len(cells)} (flagged {len(fl)}) union {un['yield_']:.4f} / ref {rk:.4f} = {un['yield_'] / rk:.2f}; flagged-only {unf['yield_'] if unf else None}; map rel SB {rel[1]:.2f}; g2 {g2}; {time.time()-tv:.0f}s", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"pe_{sub}.json").write_text(json.dumps(res, indent=1))


def summarize(target, out):
    """Round-2 rule (METHOD ROUND 1 RULING): finest window width whose UNFLAGGED combined union keeps >= 95 % of the kind-matched broadband
    reference, with within-participant map reliability (Spearman-Brown, odd vs even folds/queries) >= 0.5; ties -> fewer cells."""
    per = [json.loads(f.read_text()) for f in sorted(out.glob(f"{target}_sub-*.json"))]; n = len(per); rows = {}
    for name in per[0]["variants"]:
        V = [p["variants"][name] for p in per]; rk = V[0]["ref_kind"]
        u = np.array([v["union_unflagged"]["yield_"] for v in V]); r = np.array([p["refs"][rk]["yield_"] for p in per])
        rows[name] = dict(spec_ms=V[0]["spec_ms"], bands=V[0]["bands"], kind=V[0]["kind"], n_cells=V[0]["n_cells"], n_flagged=V[0]["n_flagged"], ref_kind=rk,
                          union_unflagged_mean=float(u.mean()), ref_mean=float(r.mean()), ratio_of_means=float(u.mean() / r.mean()), ratio_pp=(u / r).tolist(),
                          union_flagged_mean=float(np.mean([v["union_flagged"]["yield_"] for v in V])) if V[0]["union_flagged"] else None,
                          map_rel_sb_mean=float(np.mean([v["map_reliability"]["sb"] for v in V])), map_rel_sb_pp=[v["map_reliability"]["sb"] for v in V],
                          max_cell_minus_floor=float(np.mean([max(c["yield_"] - c["floor"] for c in v["cells"]) for v in V])),
                          floor_mean=float(np.mean([np.mean([c["floor"] for c in v["cells"]]) for v in V])), runtime_s=float(np.mean([v["runtime_s"] for v in V])), g2_vs_m3=[v["g2_vs_m3"] for v in V])
        rows[name]["passes"] = bool(rows[name]["ratio_of_means"] >= 0.95 and rows[name]["map_rel_sb_mean"] >= 0.5)
    comb = [k for k in rows if rows[k]["kind"] == "combined" and rows[k]["passes"] and (rows[k]["bands"] == "5" or k.startswith("R3_"))]
    winner = min(comb, key=lambda k: (rows[k]["spec_ms"], rows[k]["n_cells"])) if comb else None
    res = dict(target=target, n=n, participants=[p["participant"] for p in per], refs={k: float(np.mean([p["refs"][k]["yield_"] for p in per])) for k in per[0]["refs"]},
               variants=rows, width_winner=winner, rule=summarize.__doc__)
    (out / f"summary_{target}.json").write_text(json.dumps(res, indent=1))
    print("refs", res["refs"], "winner", winner)
    for k, r_ in rows.items(): print(k, {x: (round(r_[x], 3) if isinstance(r_[x], float) else r_[x]) for x in ("n_cells", "n_flagged", "union_unflagged_mean", "ref_mean", "ratio_of_means", "union_flagged_mean", "map_rel_sb_mean", "passes", "runtime_s")})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["image", "pe"]); ap.add_argument("--subjects", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--round", type=int, default=2)
    a = ap.parse_args(); global ROUND; ROUND = a.round; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(a.target, out)
    if a.target == "image":
        wait_gpu(12.0)
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"image_sub-{s:02d}.json").exists(): image_subject(s, out)
    else:
        from i2_complementarity import targets
        evaluate, subs, fpath, _ = targets()
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"pe_sub-{s:03d}.json").exists(): pe_subject(s, out, evaluate, fpath)


if __name__ == "__main__":
    main()
