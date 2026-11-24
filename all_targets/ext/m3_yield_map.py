"""M3 YIELD MAP (rung R5; METHOD CHANGE 8): where usable EEG lives, time x band x region, under the locked yield test.
EXECUTOR-2, 21 Nov 2026. Design = "## M3 PROPOSAL" + "## M3 REVIEW" binding changes (outputs/EXTENSIONS_LOOP_20261121.md).

Grid (fixed before outcomes):
  regions  = experiments/ext/regions.py REGIONS (8 named sets x 6 channels, both montages) + 3 size-matched random sets (reference only)
  bands    = delta 1-4, theta 4-8, alpha 8-13, beta 13-30, low-gamma 30-45 Hz
  windows  = 100 ms. Image: 0-1000 ms (10 windows; sample 0 = stimulus onset, METHOD TIME-AXIS RULING 03:05; the last window is
             900-996 ms; windows > 200 ms contain the next RSVP image). PE: 0-600 ms after feedback (6 windows).
  reference rows (not in the FWER family): image = 0-60 ms band-passed voltage (before image-specific signal; no pre-stimulus data
             exists in the files); PE = -200..0 ms pre-feedback (all kinds; expectation, not leakage).
  cell kinds: combined (power + voltage, the map and its FWER family), voltage-only, power-only. Power-containing cells whose window
             starts < FWHM/2 after the epoch start (image: "edge-affected") / after feedback (PE: "includes pre-event activity") are
             flagged; the timing prediction uses voltage cells and unflagged power cells only (G1 PASS M3 condition 2).
Cell features, per channel of the region:
  power   = complex Morlet power on the WHOLE epoch (reflect padding), 3 frequencies per band averaged, fixed cycles per band
            (delta 2, theta 3, alpha 5, beta 7, gamma 7), averaged inside the window, log10. Image: single-repetition power averaged over
            the repetitions of the query/training item (4 train, 80 test), so train and test power share their expectation.
  voltage = zero-phase band-passed voltage (|Butterworth-4|^2 applied in the frequency domain on the reflect-padded whole epoch),
            downsampled to 5 x 20-ms bins per window. Image: computed on the repetition mean (as the base ridge path).
  => 6 features per channel, 36 per cell.
Read-out and score:
  image = standardise (train stats) -> ridge -> CLIP ViT-H/14 (experiments/eegbits/run.py ridge_predict, pick_alpha rng seed 0),
          k = 80 test queries, InfoNCE soft bound over 200 candidates (run.soft_bits, LOO temperature). Floors per cell: noise input
          (N(0,1), seeds 100-102) and shuffled pairs (seeds 200-202); cell floor = max(mean noise, mean shuffled-pair).
          All-feature reference (G2 + share): run.py raw path (PCA-1024 of the broadband 63 x 250 epoch, the alpha sequence of run.py).
          Map cells are scored in batches of 40 (image_batch: same rule; asserted equal to the per-cell scorer on the first cell).
  PE    = the standard PE log-score (experiments/standard/pe_bottleneck_logscore.py): in-fold z-score, ridge (LAM * n), held-out
          Gaussian density gain with calibration fit on inner 9-fold read-outs of the training rows; 10 contiguous folds.
          Floors: target circularly shifted by n/4, n/2, 3n/4 (mean) and noise input; cell floor = max(shift mean, noise).
          Map cells are scored in batches (pe_gain_batch: same algebra; asserted equal to pe_gain on the first cell to 1e-8).
          G2: the same code with the X7 raw-ERP features (results/features/raw) reproduces results/standard/pe_bottleneck/sub-XXX.json
          ('eeg','pca','full') gain and shifts exactly. fm_epochs broadband cell (64 ch, 0-600 ms, 20-ms bins) reported beside X7.
Statistics (--summarize): D[s, cell] = yield - floor. FWER = max-t sign-flip permutation over the 8-region cells within target
  (exact 2^n when n <= 12, else 10,000 flips); Holm on one-sided t p-values (cross-check); cluster-mass sign-flip over adjacent
  time x band cells within region (secondary, cluster-forming one-sided p < 0.05). Peak cell = argmax group-mean D; prediction holds
  iff peak in the predicted set AND >= 50 % of 2,000 participant-bootstrap peaks fall in it. Share = cell yield / all-feature yield.
  Predicted sets: image {occipital, parieto_occipital} x {100-200, 200-300 ms} x {delta, theta}; PE {central, frontal} x
  {200-300, 300-400 ms} x {delta, theta}; peak taken over the timing-eligible cells. Effective time resolution per band (Morlet amplitude-envelope FWHM at band centre) is saved.
M3b (--tree, METHOD CHANGE 11 / M3b REVIEW): data-independent k-d partition (median split on the axis of largest range) of the
  standard_1005 unit-sphere coordinates, depths 0-4, written to experiments/ext/regions_kdtree.json (sha256 printed) before any yield.
  Per node: broadband voltage in 20-ms bins (image 0-1000 ms, repetition mean; PE 0-600 ms), same read-out/score/floors as the map.
  Null = the same algorithm after 50 random 3-D rotations (seeds 1000-1049); secondary = scattered random partitions with the fixed
  tree's node sizes (image 50, PE 20 draws). D_d = best fixed node - mean best rotated node; one-sided sign-flip test; fixed-sequence
  depths 1 -> 4; resolution = last depth passed; LOPO best-node magnitude and bootstrap winning-node frequency. G2: depth 0 = the
  map's broadband all-channel cell.
Usage: python m3_yield_map.py --target image|pe --subjects 1 --out DIR [--tree] [--summarize]
  (image + --tree runs the map AND the tree from one data load; PE --tree runs the tree only)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import signal, stats

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120"):
    sys.path.insert(0, str(p))
import regions as RG

BANDS = {"delta": (1, 4, [1.5, 2.5, 3.5], 2), "theta": (4, 8, [4.5, 6.0, 7.5], 3), "alpha": (8, 13, [8.5, 10.5, 12.5], 5),
         "beta": (13, 30, [14.0, 20.0, 28.0], 7), "gamma": (30, 45, [31.0, 37.0, 44.0], 7)}
BNAMES = list(BANDS)
DEV = "cuda"
PRED = {"image": dict(regions={"occipital", "parieto_occipital"}, t_ms={100, 200}, bands={"delta", "theta"}),
        "pe": dict(regions={"central", "frontal"}, t_ms={200, 300}, bands={"delta", "theta"})}


def fwhm_ms():
    out = {}
    for b, (lo, hi, fr, nc) in BANDS.items():
        fc = np.sqrt(lo * hi); sig = nc / (2 * np.pi * fc); out[b] = round(1000 * 2.3548 * sig, 1)
    return out


def morlet_bank(sf, T):
    """FFT of complex Morlet wavelets for all 15 frequencies (L1-normalised), and the reflect pad."""
    M = int(np.ceil(3 * 2 / (2 * np.pi * 1.5) * sf))            # half-length of the widest wavelet (delta 1.5 Hz, 2 cycles)
    M = min(M, T - 1); nfft = int(2 ** np.ceil(np.log2(T + 4 * M + 1)))
    Ws = []
    for b in BNAMES:
        for f in BANDS[b][2]:
            sig = BANDS[b][3] / (2 * np.pi * f); t = np.arange(-M, M + 1) / sf
            w = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sig ** 2)); w /= np.abs(w).sum()
            Ws.append(np.fft.fft(w, nfft))
    return M, nfft, torch.tensor(np.array(Ws), dtype=torch.complex64, device=DEV)


def band_power(x, sf, M, nfft, Wf):
    """x: (N, C, T) float32 GPU -> (N, C, 5 bands, T) power (mean over the band's 3 frequencies)."""
    N, C, T = x.shape
    xp = torch.nn.functional.pad(x.reshape(N * C, 1, T), (M, M), mode="reflect").reshape(N * C, T + 2 * M)
    X = torch.fft.fft(xp, nfft)                                   # (NC, nfft)
    out = torch.empty((N * C, 5, T), device=DEV)
    for bi in range(5):
        acc = 0
        for j in range(3):
            y = torch.fft.ifft(X * Wf[3 * bi + j][None], nfft)[:, M:M + T + 2 * M]   # 'full' conv shifted by M (centred kernel)
            acc = acc + y[:, M:M + T].abs() ** 2
        out[:, bi] = acc / 3
    return out.reshape(N, C, 5, T)


def bandpass_gains(sf, nfft):
    f = np.fft.fftfreq(nfft, 1 / sf); g = []
    for b in BNAMES:
        lo, hi = BANDS[b][:2]; sos = signal.butter(4, [lo, hi], btype="band", fs=sf, output="sos")
        _, h = signal.sosfreqz(sos, worN=np.abs(f), fs=sf); g.append(np.abs(h) ** 2)
    return torch.tensor(np.array(g), dtype=torch.float32, device=DEV)


def band_voltage(x, sf, G, nfft):
    """x: (N, C, T) GPU -> (N, C, 5, T) zero-phase band-passed voltage (reflect pad T-1 each side)."""
    N, C, T = x.shape; P = T - 1
    xp = torch.nn.functional.pad(x.reshape(N * C, 1, T), (P, P), mode="reflect").reshape(N * C, -1)
    X = torch.fft.fft(xp, nfft)
    return torch.stack([torch.fft.ifft(X * G[b][None], nfft).real[:, P:P + T] for b in range(5)], 1).reshape(N, C, 5, T)


def window_feats(pw, vo, t0, wlen, nwin):
    """pw, vo: (N, C, 5, T) -> power (N, C, 5, nwin) [window mean], voltage (N, C, 5, nwin, 5) [5 bins]."""
    P = torch.stack([pw[..., t0 + w * wlen:t0 + (w + 1) * wlen].mean(-1) for w in range(nwin)], -1)
    V = torch.stack([vo[..., t0 + w * wlen:t0 + (w + 1) * wlen].reshape(*vo.shape[:3], 5, wlen // 5).mean(-1) for w in range(nwin)], 3)
    return P, V


def gpu_ridge(Ztr, Ytr, Zte, alphas):
    """Same algebra as run.ridge_predict (W = (Z'Z + a I)^-1 Z'(Y - ym), pred = Z W + ym), float64 on the GPU, all alphas at once."""
    Z = torch.as_tensor(Ztr, dtype=torch.float64, device=DEV); Y = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV); ym = Y.mean(0)
    G = Z.T @ Z; C = Z.T @ (Y - ym); Zt = torch.as_tensor(Zte, dtype=torch.float64, device=DEV); I = torch.eye(G.shape[0], dtype=torch.float64, device=DEV)
    return [Zt @ torch.linalg.solve(G + a * I, C) + ym for a in alphas]


def fast_pick_alpha(Z, Y, rng, R):
    """Same rule and split as run.pick_alpha (90/10 split from rng, max MRR over run.ALPHAS, first maximum); GPU ridge, MRR on the GPU; Y = GPU float64 tensor."""
    idx = rng.permutation(len(Z)); cut = int(0.9 * len(idx)); tr, va = idx[:cut], idx[cut:]
    tr_t, va_t = torch.as_tensor(tr, device=DEV), torch.as_tensor(va, device=DEV); Zg = torch.as_tensor(Z, dtype=torch.float64, device=DEV)
    T = Y[va_t]; T = T / T.norm(dim=1, keepdim=True); best, ba = -1, None
    for a, p in zip(R.ALPHAS, gpu_ridge(Zg[tr_t], Y[tr_t], Zg[va_t], R.ALPHAS)):
        p = p / p.norm(dim=1, keepdim=True); S = p @ T.T; rank = (S > S.diag()[:, None]).sum(1) + 1; m = float((1.0 / rank).mean())
        if m > best: best, ba = m, a
    return ba


def cell_feats(P, V, chs, b, w, kind):
    """P: (N, C, 5, W) log-power; V: (N, C, 5, W, 5). kind: combined (6 / ch) | voltage (5 / ch) | power (1 / ch)."""
    if kind == "power": return P[:, chs, b, w]
    if kind == "voltage": return V[:, chs, b, w, :].reshape(len(P), -1)
    return np.concatenate([P[:, chs, b, w], V[:, chs, b, w, :].reshape(len(P), -1)], 1)


def region_sets(ch):
    sets = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}
    for i, r in enumerate(RG.random_regions(ch)):
        sets[f"random{i}"] = r
    return sets


def edge_flag(kind, b, t_ms):
    """Power-containing cells whose window starts < FWHM/2 after the epoch start (image: edge-affected) / after feedback (PE: includes
    pre-event activity). Voltage cells are never flagged."""
    return bool(kind != "voltage" and t_ms < fwhm_ms()[BNAMES[b]] / 2)


# ------------------------------------------------------------------ image
def image_load(s, R):
    t0 = time.time(); sub = f"sub-{s:02d}"
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
    Xtr4, Xte80, meta = R.load_raw(sub); assert Xtr4.shape[-1] == 250
    print(sub, "loaded", f"{time.time()-t0:.0f}s", flush=True)
    return sub, Ytr, Yte, Xtr4, Xte80, meta["ch_names"]


def image_scorer(Ytr, Yte, R):
    Yg = torch.tensor(Ytr, dtype=torch.float64, device=DEV); Yn = Yte / np.linalg.norm(Yte, axis=1, keepdims=True)
    def score(Ftr, Fte, alpha=None):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; A, B = (Ftr - mu) / sd, (Fte - mu) / sd
        if alpha is None: alpha = fast_pick_alpha(A, Yg, np.random.default_rng(0), R)
        p = gpu_ridge(A, Yg, B, [alpha])[0].cpu().numpy(); p = p / np.linalg.norm(p, axis=1, keepdims=True)
        return alpha, p
    def full(Ftr, Fte, n_noise=3):
        alpha, p = score(Ftr, Fte); y = R.soft_bits(p @ Yn.T); nz = []
        for k in range(n_noise):
            r = np.random.default_rng(100 + k); nz.append(R.soft_bits(score(r.standard_normal(Ftr.shape), r.standard_normal(Fte.shape), alpha)[1] @ Yn.T))
        sh = [R.soft_bits(p @ Yn[np.random.default_rng(200 + k).permutation(200)].T) for k in range(n_noise)]
        return dict(yield_=y, noise=float(np.mean(nz)), shuffled=float(np.mean(sh)), floor=float(max(np.mean(nz), np.mean(sh))), alpha=alpha)
    return score, full, Yn


TAUS_G = None


def soft_bits_gpu(S, rows=False):
    """Batched run.soft_bits: S (B, n, n) cosine rows -> (B,) InfoNCE bits with the per-query LOO temperature over run._TAUS
    (rows=True: the (B, n) per-query values, whose mean is the yield)."""
    import run as R
    global TAUS_G
    if TAUS_G is None: TAUS_G = torch.tensor(R._TAUS, dtype=torch.float64, device=DEV)
    n = S.shape[-1]; Z = S[:, None] / TAUS_G[None, :, None, None]                      # (B, T, n, n)
    CE = -torch.diagonal(torch.log_softmax(Z, dim=-1), dim1=-2, dim2=-1) / np.log(2)    # (B, T, n)
    loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1); t_idx = loo.argmin(1, keepdim=True)
    q = np.log2(n) - torch.gather(CE, 1, t_idx)[:, 0]
    return q if rows else q.mean(-1)


def image_batch(Ftr, Fte, Ytr_g, Yn_g, R, n_noise, mem=1.5e9, per_query=False):
    """Batched image members of equal width, all on the GPU: Ftr (B, N, k), Fte (B, 200, k) tensors (any float dtype). Same rule as
    image_scorer.full per member: train-stat standardisation (ddof 0); alpha = first max dev-MRR over run.ALPHAS on the rng(0) 90/10
    split; ridge on all training rows; InfoNCE soft bound (soft_bits_gpu = run.soft_bits). Floors: noise input (torch generator
    100+j, the member's alpha) and shuffled pairs (numpy rng 200+j permutations). Returns a list of dicts."""
    Bn, N, k = Ftr.shape; step = max(1, int(mem // (N * k * 8))); out = []
    idx = np.random.default_rng(0).permutation(N); cut = int(0.9 * N); tr, va = torch.as_tensor(idx[:cut], device=DEV), torch.as_tensor(idx[cut:], device=DEV)
    I = torch.eye(k, dtype=torch.float64, device=DEV); T = Ytr_g[va]; T = T / T.norm(dim=1, keepdim=True)
    perms = [torch.as_tensor(np.random.default_rng(200 + j).permutation(200), device=DEV) for j in range(n_noise)]
    def fit(Z, Y):
        ym = Y.mean(0); return Z.transpose(1, 2) @ Z, Z.transpose(1, 2) @ (Y - ym), ym
    def std(Atr, Ate):
        mu, sd = Atr.mean(1, keepdim=True), Atr.std(1, unbiased=False, keepdim=True) + 1e-6; return (Atr - mu) / sd, (Ate - mu) / sd
    for i0 in range(0, Bn, step):
        A, B = std(torch.as_tensor(Ftr[i0:i0 + step], device=DEV).double(), torch.as_tensor(Fte[i0:i0 + step], device=DEV).double()); nb = A.shape[0]
        G, C, ym = fit(A[:, tr], Ytr_g[tr]); Zv = A[:, va]
        best = torch.full((nb,), -1.0, dtype=torch.float64, device=DEV); alpha = torch.zeros(nb, dtype=torch.float64, device=DEV)
        for a in R.ALPHAS:
            pv = Zv @ torch.linalg.solve(G + a * I, C) + ym; pv = pv / pv.norm(dim=2, keepdim=True); S = pv @ T.T
            d = torch.diagonal(S, dim1=1, dim2=2); m = (1.0 / ((S > d[..., None]).sum(2) + 1)).mean(1)
            upd = m > best; best = torch.where(upd, m, best); alpha = torch.where(upd, torch.full_like(alpha, float(a)), alpha)
        Ia = alpha[:, None, None] * I
        def predict(Zt, G, C, ym):
            q = Zt @ torch.linalg.solve(G + Ia, C) + ym; return q / q.norm(dim=2, keepdim=True)
        G, C, ym = fit(A, Ytr_g); P = predict(B, G, C, ym); yq = soft_bits_gpu(P @ Yn_g.T, rows=True); y = yq.mean(-1)
        nz = []
        for j in range(n_noise):
            g = torch.Generator(device=DEV); g.manual_seed(100 + j)
            Nt, Ne = std(torch.randn(A.shape, generator=g, device=DEV, dtype=torch.float64), torch.randn(B.shape, generator=g, device=DEV, dtype=torch.float64))
            nz.append(soft_bits_gpu(predict(Ne, *fit(Nt, Ytr_g)) @ Yn_g.T))
        sh = [soft_bits_gpu(P @ Yn_g[pm].T) for pm in perms]
        nzm, shm = torch.stack(nz).mean(0), torch.stack(sh).mean(0)
        for i in range(nb):
            out.append(dict(yield_=float(y[i]), noise=float(nzm[i]), shuffled=float(shm[i]), floor=float(max(nzm[i], shm[i])), alpha=float(alpha[i])))
            if per_query: out[-1]["pq"] = yq[i].cpu().numpy()
        del A, B, G, C, P
    return out


def image_broadband(Xtr4, Xte80, chs=None):
    """Broadband voltage, 20-ms bins (5 samples), 0-1000 ms, repetition mean: (N, len(chs) * 50)."""
    a = Xtr4.mean(1) if chs is None else Xtr4[:, :, chs].mean(1); b = Xte80.mean(1) if chs is None else Xte80[:, :, chs].mean(1)
    f = lambda X: X.reshape(len(X), X.shape[1], 50, 5).mean(-1).reshape(len(X), -1)
    return f(a), f(b)


def image_subject(s, out, smoke, data=None):
    import run as R
    t0 = time.time(); sub, Ytr, Yte, Xtr4, Xte80, ch = data if data is not None else image_load(s, R); sf, T = 250.0, 250
    # G2 reference: run.py raw path (PCA-1024 via run.pca_scores on the GPU), run.py's alpha sequence (rng seed 0) on the GPU
    Xtr, Xte = Xtr4.mean(1).reshape(len(Xtr4), -1), Xte80.mean(1).reshape(200, -1)
    Ztr, (Zte,), _ = R.pca_scores(Xtr, [Xte]); rng = np.random.default_rng(0); Yg0 = torch.tensor(Ytr, dtype=torch.float64, device=DEV)
    Zg = torch.as_tensor(Ztr, dtype=torch.float64, device=DEV)
    for L in [2, 5, 10, 25, 50, 100, 250, 1024]:
        a = fast_pick_alpha(Zg[:, :L], Yg0, rng, R)
    m = R.metrics(gpu_ridge(Zg[:, :1024], Yg0, Zte[:, :1024], [a])[0].cpu().numpy(), Yte); ref = dict(alpha=a, **m); del Zg, Yg0
    old = json.loads((ROOT / f"results/eegbits/raw_{sub}.json").read_text())
    o = next(r for r in old["rows"] if r["L"] == 1024 and r["k"] == 80 and r["condition"] == "eeg")
    g2 = dict(file=f"results/eegbits/raw_{sub}.json", row="L=1024,k=80,eeg", old={k: o[k] for k in ("alpha", "top1", "two_way", "mrr")},
              new={k: ref[k] for k in ("alpha", "top1", "two_way", "mrr")}, all_feature_soft_bits=ref["soft_bits"])
    g2["alpha_equal"] = bool(o["alpha"] == ref["alpha"])   # alphas on the run.ALPHAS grid can tie in dev MRR to float precision; metrics decide
    g2["pass"] = bool(abs(o["top1"] - ref["top1"]) < 1e-9 and abs(o["two_way"] - ref["two_way"]) < 1e-3 and abs(o["mrr"] - ref["mrr"]) < 1e-3)
    del Xtr, Xte, Ztr, Zte
    print(sub, "G2", g2, f"{time.time()-t0:.0f}s", flush=True)
    score, full, Yn = image_scorer(Ytr, Yte, R)
    bb = full(*image_broadband(Xtr4, Xte80))                 # broadband all-channel cell (= M3b tree depth 0)
    # features: sample 0 = stimulus onset (METHOD TIME-AXIS RULING); 10 windows of 25 samples, 0-1000 ms; reference = 0-60 ms voltage
    M, nfft, Wf = morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); G = bandpass_gains(sf, nfv)
    def feats(X, reps):   # X: (items, reps, C, T) numpy
        Ps, Vs, Rs = [], [], []
        for i in range(0, len(X), max(1, 800 // reps)):
            xb = torch.tensor(np.ascontiguousarray(X[i:i + max(1, 800 // reps)]), device=DEV)
            nI = xb.shape[0]; pw = band_power(xb.reshape(nI * reps, *xb.shape[2:]), sf, M, nfft, Wf)
            pw = pw.reshape(nI, reps, *pw.shape[1:]).mean(1)
            vo = band_voltage(xb.mean(1), sf, G, nfv)
            P, V = window_feats(pw, vo, 0, 25, 10); Ps.append(torch.log10(P + 1e-20).cpu().numpy()); Vs.append(V.cpu().numpy())
            Rs.append(vo[..., 0:15].reshape(*vo.shape[:3], 5, 3).mean(-1).cpu().numpy())
        return np.concatenate(Ps), np.concatenate(Vs), np.concatenate(Rs)
    Ptr, Vtr, Rtr = feats(Xtr4, 4); Pte, Vte, Rte = feats(Xte80, 80)
    print(sub, "features", Ptr.shape, Vtr.shape, f"{time.time()-t0:.0f}s", flush=True)
    sets = region_sets(ch); cells = []; Yg = torch.tensor(Ytr, dtype=torch.float64, device=DEV); Yng = torch.as_tensor(Yn, dtype=torch.float64, device=DEV)
    G_ = lambda X: torch.as_tensor(X, device=DEV)
    Ptr, Vtr, Rtr, Pte, Vte, Rte = map(G_, (Ptr, Vtr, Rtr, Pte, Vte, Rte))   # GPU-resident features (float32)
    grid = [(rname, chs, b, w) for rname, chs in sets.items() for b in range(5) for w in range(10)]
    CH = torch.as_tensor([g[1] for g in grid], device=DEV); BI = torch.as_tensor([g[2] for g in grid], device=DEV); WI = torch.as_tensor([g[3] for g in grid], device=DEV)
    def gather_sl(P, V, kind, sl):   # -> (B, N, k) for grid members sl: per member the cell's channels x [power | 5 voltage bins]
        ch_, b_, w_ = CH[sl], BI[sl][:, None], WI[sl][:, None]
        p = P[:, ch_, b_, w_].permute(1, 0, 2)                                            # (B, N, 6)
        v = V[:, ch_, b_, w_, :].permute(1, 0, 2, 3).flatten(2)                           # (B, N, 30)
        return p if kind == "power" else v if kind == "voltage" else torch.cat([p, v], 2)
    for kind in ("combined", "voltage", "power"):
        tk = time.time(); r = []
        for i0 in range(0, len(grid), 40):   # 40 cells per call: bounded GPU memory (the GPU is shared)
            sl = slice(i0, i0 + 40); sub_ = lambda P, V: gather_sl(P, V, kind, sl)
            r += image_batch(sub_(Ptr, Vtr), sub_(Pte, Vte), Yg, Yng, R, 3 if kind == "combined" else 1, mem=6e8)
        if kind == "combined":   # batched == per-cell scorer on the first cell
            rn, chs, b, w = grid[0]; chk = full(cell_feats(Ptr.cpu().numpy(), Vtr.cpu().numpy(), chs, b, w, kind), cell_feats(Pte.cpu().numpy(), Vte.cpu().numpy(), chs, b, w, kind), n_noise=3)
            assert abs(chk["yield_"] - r[0]["yield_"]) < 1e-4, (chk, r[0])      # noise draws differ (GPU generator); alpha ties allowed
            print(sub, "batch check", dict(per_cell=chk["yield_"], batched=r[0]["yield_"], alpha_equal=chk["alpha"] == r[0]["alpha"]), flush=True)
        for (rname, chs, b, w), x in zip(grid, r):
            cells.append(dict(region=rname, band=BNAMES[b], t_ms=100 * w, kind=kind, ref=False, flag=edge_flag(kind, b, 100 * w), **x))
        print(sub, kind, f"{len(grid)} cells in {time.time()-tk:.0f}s; max named cell {max(c['yield_'] for c in cells if c['kind']==kind and not c['region'].startswith('random')):.3f}", f"{time.time()-t0:.0f}s", flush=True)
    rgrid = [(rname, chs, b) for rname, chs in sets.items() for b in range(5)]
    RC = torch.as_tensor([g[1] for g in rgrid], device=DEV); RB = torch.as_tensor([g[2] for g in rgrid], device=DEV)
    rg = lambda Rx: Rx[:, RC, RB[:, None], :].permute(1, 0, 2, 3).flatten(2)
    r = image_batch(rg(Rtr), rg(Rte), Yg, Yng, R, 1, mem=6e8)
    for (rname, chs, b), x in zip(rgrid, r):
        cells.append(dict(region=rname, band=BNAMES[b], t_ms=0, kind="voltage_ref_0_60ms", ref=True, flag=False, **x))
    res = dict(target="image", participant=sub, unit="bits per query (k=80, 200-way InfoNCE)", time_axis="sample 0 = stimulus onset; windows 0-1000 ms",
               bands=BANDS, fwhm_ms=fwhm_ms(), regions={k: [ch[i] for i in v] for k, v in sets.items()}, g2=g2, all_feature_yield=ref["soft_bits"],
               broadband_cell=bb, cells=cells, runtime_s=round(time.time() - t0, 1))
    (out / f"image_{sub}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ PE
def pe_gain(F, colw, Ys, LAM):
    """Nested held-out density gain; a line-for-line restriction of pe_bottleneck_logscore.run_subject to (pca, 'full')."""
    from belief_bottleneck import folds
    from pe_bottleneck_logscore import gain_rows
    F = torch.tensor(F, dtype=torch.float64, device=DEV); cw = torch.tensor(colw, dtype=torch.float64, device=DEV)
    Y = torch.tensor(Ys, device=DEV); n = len(Ys); fo = folds(n); sc = np.full((n, Ys.shape[1]), np.nan)
    def pred(tr, te):
        A, B = F[tr], F[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * cw; B = (B - mu) / sd * cw
        Yc = Y[tr] - Y[tr].mean(0); k = A.shape[1]
        W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(k, dtype=torch.float64, device=DEV), A.T @ Yc)
        return (B @ W).cpu().numpy()
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]
        P = pred(torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)); inner = np.zeros((len(tr), Ys.shape[1])); fi = folds(len(tr), 9)
        for c in range(9):
            inner[fi == c] = pred(torch.tensor(tr[fi != c], device=DEV), torch.tensor(tr[fi == c], device=DEV))
        for j in range(Ys.shape[1]):
            sc[te, j] = gain_rows(Ys[tr, j], inner[:, j], Ys[te, j], P[:, j])
    assert np.isfinite(sc).all()
    return sc


def pe_gain_batch(Fb, Ys, LAM):
    """pe_gain for B cells of equal width at once (colw = 1): Fb (B, n, k) -> (B, n, J) per-trial gains. Same folds, in-fold z-scoring,
    ridge (LAM * n_train), inner 9-fold calibration and Gaussian density gain as pe_gain; batched torch solves + closed-form OLS."""
    from belief_bottleneck import folds
    F = torch.as_tensor(Fb, dtype=torch.float64, device=DEV); Y = torch.as_tensor(Ys, dtype=torch.float64, device=DEV)
    Bn, n, k = F.shape; J = Ys.shape[1]; fo = folds(n); I = torch.eye(k, dtype=torch.float64, device=DEV)
    def pred(tr, te):
        A, B = F[:, tr], F[:, te]; mu, sd = A.mean(1, keepdim=True), A.std(1, unbiased=False, keepdim=True) + 1e-6; A = (A - mu) / sd; B = (B - mu) / sd
        Yc = Y[tr] - Y[tr].mean(0); W = torch.linalg.solve(A.transpose(1, 2) @ A + LAM * len(tr) * I, A.transpose(1, 2) @ Yc.expand(Bn, -1, -1))
        return B @ W                                                  # (B, n_te, J)
    sc = torch.full((Bn, n, J), float("nan"), dtype=torch.float64, device=DEV)
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; P = pred(tr, te); inner = torch.zeros((Bn, len(tr), J), dtype=torch.float64, device=DEV); fi = folds(len(tr), 9)
        for c in range(9): inner[:, fi == c] = pred(tr[fi != c], tr[fi == c])
        yt, ye = Y[tr][None], Y[te][None]; xm, ym = inner.mean(1, keepdim=True), yt.mean(1, keepdim=True)
        slope = ((inner - xm) * (yt - ym)).sum(1, keepdim=True) / ((inner - xm) ** 2).sum(1, keepdim=True); icpt = ym - slope * xm
        rv = ((yt - icpt - slope * inner) ** 2).mean(1, keepdim=True); vv = ((yt - ym) ** 2).mean(1, keepdim=True)
        sc[:, te] = (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * P) ** 2 / rv) / (2 * np.log(2))
    out = sc.cpu().numpy(); assert np.isfinite(out).all(); return out


def pe_load(s, evaluate, fpath, dataset="ds003458"):
    sub = f"sub-{s:03d}"; fz = np.load(fpath("raw", s)); tri = fz["tri"]
    d = ROOT / ("results/fm_epochs" if dataset == "ds003458" else "results/fm_epochs_ds004595")
    ez = np.load(d / f"{sub}.npz"); ch = [str(c) for c in ez["ch"]]; pos = {t: i for i, t in enumerate(ez["trial"])}
    if dataset == "ds003458": assert all(t in pos for t in tri)
    else: tri = np.array([t for t in tri if t in pos])                                # ds004595: X7 trials that have a 200-Hz epoch
    n = len(tri); y = evaluate(s, tri).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * k / 4)) for k in (1, 2, 3)])
    X = ez["X"][[pos[t] for t in tri]].astype(np.float32).reshape(n, len(ch), 400)   # -1..1 s at 200 Hz, feedback at sample 200
    return sub, fz, tri, n, Ys, X, ch


def pe_broadband(X, chs=None):
    """Broadband voltage 0-600 ms after feedback, 20-ms bins (4 samples at 200 Hz)."""
    Z = X[:, :, 200:320] if chs is None else X[:, chs, 200:320]
    return Z.reshape(len(Z), Z.shape[1], 30, 4).mean(-1).reshape(len(Z), -1).astype(np.float64)


def pe_subject(s, out, smoke, evaluate, fpath):
    from belief_bottleneck import LAM
    t0 = time.time(); sub, fz, tri, n, Ys, X, ch = pe_load(s, evaluate, fpath)
    sc = pe_gain(fz["F"].astype(np.float64), fz["colw"].astype(np.float64), Ys, LAM)
    old = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text())
    o = next(r for r in old["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")
    g2 = dict(file=f"results/standard/pe_bottleneck/{sub}.json", old_gain=o["gain"], new_gain=float(sc[:, 0].mean()),
              old_shift=o["shifted"], new_shift=[float(v) for v in sc[:, 1:].mean(0)])
    g2["abs_diff"] = max(abs(g2["old_gain"] - g2["new_gain"]), *[abs(a - b) for a, b in zip(g2["old_shift"], g2["new_shift"])]); g2["pass"] = g2["abs_diff"] < 1e-9
    print(sub, "G2", g2, flush=True)
    sf, T = 200.0, 400; M, nfft, Wf = morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); G = bandpass_gains(sf, nfv)
    # windows: -200..600 ms (8 x 100 ms from sample 160); the two pre-feedback windows are the reference row (expectation, not leakage)
    Ps, Vs = [], []
    for i in range(0, n, 64):                                    # chunks of 64 trials (GPU memory is shared with other steps)
        xg = torch.tensor(X[i:i + 64], device=DEV)
        P, V = window_feats(band_power(xg, sf, M, nfft, Wf), band_voltage(xg, sf, G, nfv), 160, 20, 8)
        Ps.append(torch.log10(P + 1e-20).cpu().numpy()); Vs.append(V.cpu().numpy())
    P, V = np.concatenate(Ps), np.concatenate(Vs)
    s_bb = pe_gain(pe_broadband(X), np.ones(64 * 30), Ys, LAM)
    rng = np.random.default_rng(1000 + s); cells = []
    sets = region_sets(ch); grid = [(rname, chs, b, w) for rname, chs in sets.items() for b in range(5) for w in range(8)]
    for kind in ("combined", "voltage", "power"):
        Fb = np.stack([cell_feats(P, V, chs, b, w, kind) for _, chs, b, w in grid]).astype(np.float64)
        E = pe_gain_batch(Fb, Ys, LAM); Nz = pe_gain_batch(rng.standard_normal(Fb.shape), Ys[:, :1], LAM)
        if kind == "combined":    # batched == per-cell pe_gain (first cell), exact to float precision
            chk = pe_gain(Fb[0], np.ones(Fb.shape[2]), Ys, LAM); assert np.abs(chk - E[0]).max() < 1e-8, np.abs(chk - E[0]).max()
        for i, (rname, chs, b, w) in enumerate(grid):
            t_ms = 100 * w - 200; yv, sh, nv = float(E[i, :, 0].mean()), float(E[i, :, 1:].mean()), float(Nz[i, :, 0].mean())
            cells.append(dict(region=rname, band=BNAMES[b], t_ms=t_ms, kind=kind, ref=t_ms < 0, flag=edge_flag(kind, b, t_ms),
                              yield_=yv, shift=sh, noise=nv, floor=max(sh, nv)))
        print(sub, kind, f"max cell {max(c['yield_'] for c in cells if c['kind']==kind and not c['ref'] and not c['region'].startswith('random')):.4f}", f"{time.time()-t0:.0f}s", flush=True)
    res = dict(target="pe", participant=sub, unit="bits per feedback trial", n_trials=int(n), time_axis="ms after feedback; -200..0 = reference row",
               bands=BANDS, fwhm_ms=fwhm_ms(), regions={k: [ch[i] for i in v] for k, v in sets.items()}, g2=g2,
               all_feature_yield=g2["new_gain"], broadband_cell=dict(yield_=float(s_bb[:, 0].mean()), shift=float(s_bb[:, 1:].mean()), minus_x7=float(s_bb[:, 0].mean()) - g2["new_gain"]),
               cells=cells, runtime_s=round(time.time() - t0, 1))
    (out / f"pe_{sub}.json").write_text(json.dumps(res, indent=1))


def pe_subject_frozen(s, out, evaluate, fpath, win, bset, dataset):
    """Frozen PE grid (GRID ROUND 3 FREEZE, METHOD 10:50): win-ms windows -200..600 ms (pre-feedback windows = reference row), band set
    bset ('3' = delta-theta / alpha / beta-gamma; '5' = the M3 bands), combined + voltage + power kinds, 8 named regions (ds004595: the
    temporal regions swap the absent FT9/TP9/FT10/TP10 for F7/CP5 and F8/CP6, regions.py REGIONS_DS004595). Features via
    grid_iter.feature_sets (same Morlet / band-pass / window algebra; win 100 + 5 bands = the M3 grid exactly, the G2). Adds per participant:
    unflagged union, flagged-only union (dual pe_gain on the concatenated combined cells), map reliability (odd/even outer folds, SB)."""
    from belief_bottleneck import LAM, folds
    import grid_iter as GI
    t0 = time.time(); sub, fz, tri, n, Ys, X, ch = pe_load(s, evaluate, fpath, dataset)
    nw = 800 // win; F = GI.feature_sets(X[:, None], 1, 200.0, 160, 320, {win: nw})[win]; e = GI.edges(160, 320, nw)
    bnames = BNAMES if bset == "5" else ["delta_theta", "alpha", "beta_gamma"]; P, V = (F["P5"], F["V5"]) if bset == "5" else (F["P3"], F["V3"])
    regs = RG.REGIONS if dataset == "ds003458" else RG.REGIONS_DS004595
    sets = {k: RG.indices(ch, v) for k, v in regs.items()}
    for i, r in enumerate(RG.random_regions(ch)): sets[f"random{i}"] = r
    grid = [(rn, chs, b, w) for rn, chs in sets.items() for b in range(len(bnames)) for w in range(nw)]
    rng = np.random.default_rng(1000 + s); cells = []; odd = folds(n) % 2 == 1; comb = None
    for kind in ("combined", "voltage", "power"):
        Fb = torch.stack([(P[:, chs, b, w] if kind == "power" else V[:, chs, b, w, :].flatten(1) if kind == "voltage" else torch.cat([P[:, chs, b, w], V[:, chs, b, w, :].flatten(1)], 1))
                          for _, chs, b, w in grid]).double().cpu().numpy()
        E = pe_gain_batch(Fb, Ys, LAM); Nz = pe_gain_batch(rng.standard_normal(Fb.shape), Ys[:, :1], LAM)
        if kind == "combined": comb = Fb
        for i, (rn, chs, b, w) in enumerate(grid):
            t_ms = 5 * int(e[w] - 200); sh, nv = float(E[i, :, 1:].mean()), float(Nz[i, :, 0].mean())
            cells.append(dict(region=rn, band=bnames[b], t_ms=t_ms, kind=kind, ref=t_ms < 0, flag=GI.flag_of(bset, kind, b, t_ms), yield_=float(E[i, :, 0].mean()),
                              shift=sh, noise=nv, floor=max(sh, nv), y_odd=float(E[i, odd, 0].mean()), y_even=float(E[i, ~odd, 0].mean())))
    named = [i for i, (rn, _, _, w) in enumerate(grid) if not rn.startswith("random") and 5 * int(e[w] - 200) >= 0]
    fl = [i for i in named if cells[i]["flag"]]; uf = [i for i in named if not cells[i]["flag"]]
    un = GI.pe_union(comb[uf].transpose(1, 0, 2).reshape(n, -1), Ys, LAM); unf = GI.pe_union(comb[fl].transpose(1, 0, 2).reshape(n, -1), Ys, LAM) if fl else None
    rel = GI.spearman_sb([cells[i]["y_odd"] for i in named], [cells[i]["y_even"] for i in named])
    bb = GI.pe_union(pe_broadband(X), Ys, LAM); pw = GI.feature_sets(X[:, None], 1, 200.0, 200, 320, {600: 1})[600]
    pwr = (pw["P5"] if bset == "5" else pw["P3"]).flatten(1).double().cpu().numpy(); refc = GI.pe_union(np.concatenate([pe_broadband(X), pwr], 1), Ys, LAM)
    x7 = pe_gain(fz["F"].astype(np.float64)[[list(fz["tri"]).index(t) for t in tri]], fz["colw"].astype(np.float64), Ys, LAM)
    g2 = None
    if dataset == "ds003458" and win == 100 and bset == "5":
        m3 = {(c["region"], c["band"], c["t_ms"], c["kind"]): c["yield_"] for c in json.loads((ROOT / f"results/ext/m3/pe_{sub}.json").read_text())["cells"]}
        g2 = max(abs(c["yield_"] - m3[(c["region"], c["band"], c["t_ms"], c["kind"])]) for c in cells)
    res = dict(target="pe", dataset=dataset, participant=sub, frozen=dict(win_ms=win, bands=bset, band_names=bnames), n_trials=int(n), unit="bits per feedback trial",
               regions={k: [ch[i] for i in v] for k, v in sets.items()}, all_feature_yield=float(x7[:, 0].mean()), x7_shift=float(x7[:, 1:].mean()),
               broadband_cell=dict(yield_=bb["yield_"], shift=bb["shift"]), combined_ref=refc, union_unflagged=un, union_flagged=unf,
               map_reliability=dict(r_half=rel[0], sb=rel[1]), g2=dict(pass_=True if g2 is None else g2 < 1e-9, frozen_vs_m3=g2), cells=cells, fwhm_ms=fwhm_ms(),
               runtime_s=round(time.time() - t0, 1))
    res["g2"]["pass"] = res["g2"]["pass_"]
    (out / f"pe_{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, dataset, f"union {un['yield_']:.4f} flagged {unf['yield_'] if unf else None} ref {refc['yield_']:.4f} rel {rel[1]:.2f} x7 {res['all_feature_yield']:.4f} g2 {g2} {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ M3b: k-d spatial partition tree (METHOD CHANGE 11)
def kd_tree(xyz, idx, depth, maxd=4):
    """Median split on the axis of largest range; returns {path: [channel indices]} for depths 0..maxd. Data- and yield-free."""
    out = {"": list(idx)}
    if depth == maxd or len(idx) < 2: return out
    pts = xyz[idx]; ax = int(np.argmax(pts.max(0) - pts.min(0))); order = np.argsort(pts[:, ax], kind="stable"); h = len(idx) // 2
    for tag, sub in (("0", [idx[i] for i in order[:h]]), ("1", [idx[i] for i in order[h:]])):
        for k, v in kd_tree(xyz, sub, depth + 1, maxd).items(): out[tag + k] = v
    return out


def montage_xyz(ch):
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lut = {k.lower(): v for k, v in pos.items()}; keep = [i for i, c in enumerate(ch) if c.lower() in lut]
    xyz = np.array([lut[ch[i].lower()] for i in keep]); xyz = xyz - xyz.mean(0); xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return keep, xyz


N_ROT, N_SCAT = 50, {"image": 50, "pe": 20}


def tree_partitions(ch):
    from scipy.spatial.transform import Rotation
    keep, xyz = montage_xyz(ch)
    fixed = kd_tree(xyz, list(range(len(keep))), 0)
    rots = [kd_tree(xyz @ Rotation.random(random_state=1000 + i).as_matrix().T, list(range(len(keep))), 0) for i in range(N_ROT)]
    m = lambda t: {k: [keep[i] for i in v] for k, v in t.items()}
    return keep, m(fixed), [m(t) for t in rots]


def write_tree_file(path):
    import hashlib
    trees = {}
    for target, ch in (("image", json.loads((ROOT / "results/eegbits/raw_sub-01.json").read_text())["raw_meta"]["ch_names"]),
                       ("pe", [str(c) for c in np.load(ROOT / "results/fm_epochs/sub-001.npz")["ch"]])):
        keep, fixed, rots = tree_partitions(ch)
        trees[target] = dict(channels_used=[ch[i] for i in keep], excluded=[c for i, c in enumerate(ch) if i not in keep],
                             fixed={k: [ch[i] for i in v] for k, v in fixed.items()}, node_sizes={k: len(v) for k, v in fixed.items()},
                             rotation_seeds=[1000 + i for i in range(N_ROT)])
    txt = json.dumps(trees, indent=1)
    if path.exists(): assert path.read_text() == txt, "regions_kdtree.json differs from the deterministic tree"
    else: path.write_text(txt)
    print("regions_kdtree.json sha256", hashlib.sha256(txt.encode()).hexdigest()[:16], {t: v["excluded"] for t, v in trees.items()}, flush=True)


def scattered(keep, sizes_by_depth, seed):
    """Random (scattered) partition with the fixed tree's node sizes, per depth."""
    rng = np.random.default_rng(seed); out = {}
    for d, sizes in sizes_by_depth.items():
        perm = list(rng.permutation(keep)); i = 0
        for j, sz in enumerate(sizes): out[f"d{d}_{j}"] = perm[i:i + sz]; i += sz
    return out


def tree_subject(target, s, out, evaluate=None, fpath=None, data=None):
    """All node sets (fixed tree, 50 rotated trees, scattered partitions) are grouped by size and scored in GPU batches."""
    t0 = time.time(); write_tree_file(ROOT / "experiments/ext/regions_kdtree.json")
    if target == "image":
        import run as R
        sub, Ytr, Yte, Xtr4, Xte80, ch = data if data is not None else image_load(s, R); _, _, Yn = image_scorer(Ytr, Yte, R)
        Yg = torch.tensor(Ytr, dtype=torch.float64, device=DEV); Yng = torch.as_tensor(Yn, dtype=torch.float64, device=DEV)
        bin_ = lambda X: torch.as_tensor(X.reshape(len(X), X.shape[1], 50, 5).mean(-1), device=DEV)     # (N, 63, 50) float32 on the GPU
        btr, bte = bin_(Xtr4.mean(1)), bin_(Xte80.mean(1))
        def score(sets):
            out_ = []; step = max(1, int(6e8 // (btr.shape[0] * len(sets[0]) * 50 * 8)))
            for i in range(0, len(sets), step):
                Ix = torch.as_tensor([sorted(c) for c in sets[i:i + step]], device=DEV)               # (B, size)
                g = lambda Bx: Bx[:, Ix].permute(1, 0, 2, 3).flatten(2)                               # (B, N, size * 50)
                out_ += [dict(yield_=r["yield_"], floor=r["floor"]) for r in image_batch(g(btr), g(bte), Yg, Yng, R, 1, mem=6e8)]
            return out_
    else:
        from belief_bottleneck import LAM
        sub, fz, tri, n, Ys, X, ch = pe_load(s, evaluate, fpath)
        def score(sets):
            out_ = []
            for i in range(0, len(sets), 64):
                E = pe_gain_batch(np.stack([pe_broadband(X, sorted(c)) for c in sets[i:i + 64]]), Ys, LAM)
                out_ += [dict(yield_=float(e[:, 0].mean()), floor=float(e[:, 1:].mean())) for e in E]
            return out_
    keep, fixed, rots = tree_partitions(ch); depth = len
    sizes = {d: [len(v) for k, v in fixed.items() if depth(k) == d] for d in range(1, 5)}
    jobs = [("fixed", None, k, v) for k, v in fixed.items()]
    jobs += [("rotated", i, k, v) for i, t in enumerate(rots) for k, v in t.items() if depth(k) >= 1]
    jobs += [("scattered", i, k, v) for i in range(N_SCAT[target]) for k, v in scattered(keep, sizes, 5000 + i).items()]
    vals = [None] * len(jobs)
    for size in sorted({len(j[3]) for j in jobs}):
        ix = [i for i, j in enumerate(jobs) if len(j[3]) == size]
        for i, r in zip(ix, score([jobs[i][3] for i in ix])): vals[i] = r
        print(sub, f"size {size}: {len(ix)} sets", f"{time.time()-t0:.0f}s", flush=True)
    res = dict(target=target, participant=sub, fixed={}, rotated=[{} for _ in rots], scattered=[{} for _ in range(N_SCAT[target])])
    for (kind, i, k, v), r in zip(jobs, vals):
        if kind == "fixed": res["fixed"][k] = dict(n=len(v), channels=[ch[c] for c in v], **r)
        else: res[kind][i][k] = r["yield_"]
    if target == "image": res["broadband_all_channel_cell"] = res["fixed"][""]["yield_"]
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / f"tree_{target}_{sub}.json").write_text(json.dumps(res, indent=1)); print(sub, "tree done", res["runtime_s"], flush=True)


def tree_summarize(target, out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob(f"tree_{target}_sub-*.json"))]; n = len(per)
    cells = sorted(out.glob(f"{target}_sub-*.json")); g2 = []
    for p in per:   # G2: depth 0 = the broadband all-channel cell of the M3 map (same participant)
        f = out / f"{target}_{p['participant']}.json"
        if f.exists(): g2.append(abs(json.loads(f.read_text())["broadband_cell"]["yield_"] - p["fixed"][""]["yield_"]))
    rng = np.random.default_rng(20261121); res = dict(target=target, n=n, g2_depth0_max_abs_diff=max(g2) if g2 else None, depths={})
    if n <= 12: flips = np.array([[1 if (i >> j) & 1 == 0 else -1 for j in range(n)] for i in range(2 ** n)])
    else: flips = np.vstack([np.ones(n), rng.choice([-1, 1], size=(9999, n))])
    ci = lambda v: np.quantile(v[rng.integers(0, len(v), (10000, len(v)))].mean(1), [.025, .975]).tolist()
    passed = True
    for d in range(1, 5):
        nodes = [k for k in per[0]["fixed"] if len(k) == d]
        Yf = np.array([[p["fixed"][k]["yield_"] for k in nodes] for p in per])
        best = Yf.max(1); rot = np.array([np.mean([max(v for k, v in r.items() if len(k) == d) for r in p["rotated"]]) for p in per])
        sca = np.array([np.mean([max(v for k, v in r.items() if k.startswith(f"d{d}_")) for r in p["scattered"]]) for p in per])
        D = best - rot; tstat = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12)
        p_sf = float(np.mean([tstat(D * f) >= tstat(D) for f in flips]))
        lopo = []
        for i in range(n):
            k = int(np.delete(Yf, i, 0).mean(0).argmax()); lopo.append(Yf[i, k])
        boot = [nodes[int(Yf[rng.integers(0, n, n)].mean(0).argmax())] for _ in range(2000)]; win = nodes[int(Yf.mean(0).argmax())]
        passed = passed and p_sf < 0.05
        res["depths"][d] = dict(n_nodes=len(nodes), node_sizes=[per[0]["fixed"][k]["n"] for k in nodes], mean_node=float(Yf.mean()), best_node=win,
                                best_node_channels=per[0]["fixed"][win]["channels"], best_node_boot_freq=float(np.mean([b == win for b in boot])), lopo_best_node_yield=float(np.mean(lopo)), lopo_ci=ci(np.array(lopo)),
                                D_vs_rotated=float(D.mean()), D_ci=ci(D), p_signflip=p_sf, compact_minus_scattered=float((best - sca).mean()), compact_minus_scattered_ci=ci(best - sca),
                                fixed_sequence_pass=bool(passed))
    res["resolution_depth"] = max([0] + [d for d in range(1, 5) if res["depths"][d]["fixed_sequence_pass"]])
    res["depth0"] = dict(mean=float(np.mean([p["fixed"][""]["yield_"] for p in per])))
    (out / f"summary_tree_{target}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ group statistics (map)
def g2_ok(g):
    if "old" in g:   # image: metric equality (alpha grid ties are reported separately)
        o, n_ = g["old"], g["new"]; return abs(o["top1"] - n_["top1"]) < 1e-9 and abs(o["two_way"] - n_["two_way"]) < 1e-3 and abs(o["mrr"] - n_["mrr"]) < 1e-3
    return bool(g["pass"])


def summarize(target, out):
    files = sorted(out.glob(f"{target}_sub-*.json")); per = [json.loads(f.read_text()) for f in files]; n = len(per)
    key = lambda c: (c["region"], c["band"], c["t_ms"], c["kind"])
    lut = [{key(c): c for c in p["cells"]} for p in per]
    named = lambda c: not c["region"].startswith("random")
    fam = [key(c) for c in per[0]["cells"] if named(c) and c["kind"] == "combined" and not c["ref"]]
    elig = [key(c) for c in per[0]["cells"] if named(c) and not c["ref"] and (c["kind"] == "voltage" or (c["kind"] == "power" and not c["flag"]))]
    mat = lambda keys, field: np.array([[L[k][field] for k in keys] for L in lut])
    Yd, Fl = mat(fam, "yield_"), mat(fam, "floor"); D = Yd - Fl
    t = lambda X: X.mean(0) / (X.std(0, ddof=1) / np.sqrt(len(X)) + 1e-12)
    tobs = t(D)
    if n <= 12:
        flips = np.array([[1 if (i >> j) & 1 == 0 else -1 for j in range(n)] for i in range(2 ** n)])
    else:
        flips = np.vstack([np.ones(n), np.random.default_rng(20261121).choice([-1, 1], size=(9999, n))])
    tmax = np.array([t(D * f[:, None]).max() for f in flips]); p_fwer = np.array([(tmax >= x).mean() for x in tobs])
    p_t = stats.t.sf(tobs, n - 1); order = np.argsort(p_t); holm = np.ones_like(p_t); run_max = 0
    for r, i in enumerate(order):
        run_max = max(run_max, min(1, (len(p_t) - r) * p_t[i])); holm[i] = run_max
    # cluster mass over adjacent (band, window) combined cells within region
    thr = stats.t.ppf(0.95, n - 1); regs = sorted({k[0] for k in fam}); nb = len(BNAMES); tms = sorted({k[2] for k in fam}); nw = len(tms)
    idx = {k: i for i, k in enumerate(fam)}
    def clusters(tv):
        out_c = []
        for r in regs:
            grid = np.array([[tv[idx[(r, BNAMES[b], tms[w], "combined")]] for w in range(nw)] for b in range(nb)]); seen = np.zeros_like(grid, bool)
            for b in range(nb):
                for w in range(nw):
                    if grid[b, w] > thr and not seen[b, w]:
                        stack, members = [(b, w)], []; seen[b, w] = True
                        while stack:
                            bb, ww = stack.pop(); members.append((bb, ww))
                            for db, dw in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                b2, w2 = bb + db, ww + dw
                                if 0 <= b2 < nb and 0 <= w2 < nw and not seen[b2, w2] and grid[b2, w2] > thr:
                                    seen[b2, w2] = True; stack.append((b2, w2))
                        out_c.append((r, members, float(sum(grid[m] for m in members))))
        return out_c
    cobs = clusters(tobs); cmax = np.array([max([c[2] for c in clusters(t(D * f[:, None]))], default=0) for f in flips])
    cl = [dict(region=r, cells=[[BNAMES[b], tms[w]] for b, w in m], mass=ms, p=float((cmax >= ms).mean())) for r, m, ms in cobs]
    pr = dict(PRED[target])
    if "frozen" in per[0]:   # frozen grid: the same pre-stated prediction expressed in the frozen bands / window starts
        fz_ = per[0]["frozen"]; pr["t_ms"] = set(range(200, 400, fz_["win_ms"]))
        if fz_["bands"] == "3": pr["bands"] = {"delta_theta"}
    inset = lambda k: k[0] in pr["regions"] and k[1] in pr["bands"] and k[2] in pr["t_ms"]
    De = mat(elig, "yield_") - mat(elig, "floor"); rng = np.random.default_rng(20261121)
    peak = elig[int(De.mean(0).argmax())]; bpeaks = [elig[int(De[rng.integers(0, n, n)].mean(0).argmax())] for _ in range(2000)]
    freq = float(np.mean([inset(k) for k in bpeaks]))
    allf = np.array([p["all_feature_yield"] for p in per])
    def ci(v):
        ix = np.random.default_rng(1).integers(0, len(v), (10000, len(v))); return np.quantile(v[ix].mean(1), [.025, .975]).tolist()
    refk = [key(c) for c in per[0]["cells"] if named(c) and c["ref"]]; Dr = mat(refk, "yield_") - mat(refk, "floor")
    rand = {}
    for L in lut:
        for k, c in L.items():
            if k[0].startswith("random") and k[3] == "combined" and not c["ref"]: rand.setdefault((k[1], k[2]), []).append(c["yield_"] - c["floor"])
    cell_rows = [dict(region=k[0], band=k[1], t_ms=k[2], kind=k[3], flag=lut[0][k]["flag"], yield_mean=float(Yd[:, i].mean()), floor_mean=float(Fl[:, i].mean()),
                      d_mean=float(D[:, i].mean()), d_ci=ci(D[:, i]), t=float(tobs[i]), p_maxstat=float(p_fwer[i]), p_holm=float(holm[i]),
                      share_of_all_feature=float(Yd[:, i].mean() / allf.mean()) if allf.mean() > 0 else None) for i, k in enumerate(fam)]
    elig_rows = [dict(region=k[0], band=k[1], t_ms=k[2], kind=k[3], d_mean=float(De[:, i].mean()), d_ci=ci(De[:, i])) for i, k in enumerate(elig)]
    ipk = fam[int(D.mean(0).argmax())]
    res = dict(target=target, n=n, participants=[p["participant"] for p in per], n_cells_family=len(fam), flips=len(flips),
               n_sig_maxstat=int((p_fwer < 0.05).sum()), n_sig_holm=int((holm < 0.05).sum()),
               map_peak=dict(region=ipk[0], band=ipk[1], t_ms=ipk[2], d_mean=float(D[:, fam.index(ipk)].mean()), d_ci=ci(D[:, fam.index(ipk)]),
                             p_maxstat=float(p_fwer[fam.index(ipk)]), flag=lut[0][ipk]["flag"]),
               timing_peak=dict(region=peak[0], band=peak[1], t_ms=peak[2], kind=peak[3], d_mean=float(De[:, elig.index(peak)].mean()), d_ci=ci(De[:, elig.index(peak)])),
               prediction=dict(set=dict(regions=sorted(pr["regions"]), bands=sorted(pr["bands"]), t_ms=sorted(pr["t_ms"])), eligible="voltage cells + power cells not flagged",
                               peak_in_set=bool(inset(peak)), bootstrap_freq_in_set=freq, holds=bool(inset(peak) and freq >= 0.5)),
               all_feature_yield=dict(mean=float(allf.mean()), ci=ci(allf)), g2_all_pass=bool(all(g2_ok(p["g2"]) for p in per)),
               broadband_cell=dict(mean=float(np.mean([p["broadband_cell"]["yield_"] for p in per]))),
               reference_row=dict(kind="image: 0-60 ms voltage (before image-specific signal); PE: -200..0 ms pre-feedback (expectation)",
                                  max_d_mean=float(Dr.mean(0).max()), mean_d=float(Dr.mean()), cells=[dict(region=k[0], band=k[1], t_ms=k[2], kind=k[3], d_mean=float(Dr[:, i].mean()), d_ci=ci(Dr[:, i])) for i, k in enumerate(refk)]),
               random_region_reference={f"{b}_{tm}": float(np.mean(v)) for (b, tm), v in rand.items()}, clusters=cl, fwhm_ms=fwhm_ms(), cells=cell_rows, eligible_cells=elig_rows)
    if target == "pe":
        fb = np.array([p["broadband_cell"]["yield_"] for p in per]); res["fm_broadband_minus_x7"] = dict(mean=float((fb - allf).mean()), ci=ci(fb - allf), fm_mean=float(fb.mean()))
    if "frozen" in per[0]:
        u = np.array([p["union_unflagged"]["yield_"] for p in per]); uf_ = np.array([p["union_flagged"]["yield_"] for p in per]); rc = np.array([p["combined_ref"]["yield_"] for p in per])
        rl = np.array([p["map_reliability"]["sb"] for p in per]); xs = np.array([p["x7_shift"] for p in per])
        res["frozen"] = dict(spec=per[0]["frozen"], dataset=per[0]["dataset"], union_unflagged=dict(mean=float(u.mean()), ci=ci(u)), union_flagged=dict(mean=float(uf_.mean()), ci=ci(uf_)),
                             combined_ref=dict(mean=float(rc.mean()), ci=ci(rc)), union_over_ref=float(u.mean() / rc.mean()), map_reliability=dict(mean=float(rl.mean()), ci=ci(rl)),
                             group_gate=dict(x7_minus_shift_ci=ci(allf - xs), passes=bool(ci(allf - xs)[0] > 0)), g2_max=max((p["g2"]["frozen_vs_m3"] or 0) for p in per))
    (out / f"summary_{target}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("cells", "clusters", "random_region_reference", "eligible_cells", "reference_row")}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["image", "pe"]); ap.add_argument("--subjects", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tree", action="store_true", help="M3b k-d spatial partition tree (METHOD CHANGE 11)")
    ap.add_argument("--frozen", action="store_true", help="PE frozen grid (GRID ROUND 3): --win, --bands, --dataset")
    ap.add_argument("--win", type=int, default=50); ap.add_argument("--bands", default="3"); ap.add_argument("--dataset", default="ds003458", choices=["ds003458", "ds004595"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        return tree_summarize(a.target, out) if a.tree else summarize(a.target, out)
    if a.target == "image":
        subs = [int(x) for x in a.subjects.split(",")] if a.subjects else list(range(1, 11))
        for s in subs:
            need_map = not (out / f"image_sub-{s:02d}.json").exists(); need_tree = a.tree and not (out / f"tree_image_sub-{s:02d}.json").exists()
            if not (need_map or need_tree): continue
            import run as R
            data = image_load(s, R)                               # one 8 GB load per participant for map + tree
            if need_map: image_subject(s, out, a.smoke, data)
            if need_tree: tree_subject("image", s, out, data=data)
            del data
    else:
        sys.path.insert(0, str(ROOT / "experiments/standard"))
        if a.frozen and a.dataset == "ds004595":
            src = open(ROOT / "experiments/loop/s13b_gauss_bits.py").read(); ns = {}
            exec(compile(src[:src.index("DATASETS = {")], "s13b_head", "exec"), ns); evaluate, allsubs, fpath, _ = ns["targets_ds004595"]()
        else:
            from i2_complementarity import targets
            evaluate, allsubs, fpath, _ = targets()
        subs = [int(x) for x in a.subjects.split(",")] if a.subjects else allsubs
        if a.frozen:
            for s in subs:
                if (out / f"pe_sub-{s:03d}.json").exists(): continue
                if a.dataset == "ds004595" and not (ROOT / f"results/fm_epochs_ds004595/sub-{s:03d}.npz").exists(): print("no 200-Hz epochs yet", s, flush=True); continue
                pe_subject_frozen(s, out, evaluate, fpath, a.win, a.bands, a.dataset)
            return
        for s in subs:
            if a.tree:
                if not (out / f"tree_pe_sub-{s:03d}.json").exists(): tree_subject("pe", s, out, evaluate, fpath)
            elif not (out / f"pe_sub-{s:03d}.json").exists(): pe_subject(s, out, a.smoke, evaluate, fpath)


if __name__ == "__main__":
    main()
