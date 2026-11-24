"""M7 + A1 (rung R7; R4 on Z): how much of each target's yield does ONE frozen EEG foundation-model latent carry?
Spec: EXTENSIONS_LOOP_20261121 "## M7 REVIEW", "## T3 RULINGS" (M7 + A1), METHOD CHANGE 13, and the coordinator's 05:30 contrarian conditions
(random-init twin + five-band log-power baseline under the identical probe; layer picked on dev only; input-format check).

Latent    : REVE-base (external/reve/base, safetensors, local code as used by Q7), positions from external/reve/positions by channel name.
            Input format check (printed into every output): REVE patch = 200 samples at 200 Hz, overlap 20. THINGS-EEG2 epochs 0-1.0 s at
            250 Hz -> Fourier-resampled to 200 Hz = exactly 1 patch (Q7 convention). ds003458 epochs (results/fm_epochs, 200 Hz, -1..1 s, 400
            samples) -> 2 patches; the post-feedback patch (-0.1..0.9 s) is used (E11/revetok convention). ZuCo: no raw time series on disk
            (METHOD data gate, 04:45) -> not run. Per-channel z-scoring with training-set statistics (image) / session statistics (PE, as E11).
            Feature = all channel tokens of the patch at layer l, flattened (channels x 512; the Q7/E11 layer is l = 22 = final).
Layers    : candidates LAYERS = (6, 11, 16, 22); chosen on DEV only, separately for pretrained and twin:
            image: held-out tenth of the training images (Q7 split, seed 0); criterion = dev InfoNCE log-score at the MRR-chosen alpha.
            PE   : per outer fold, the inner-CV (9 contiguous inner folds of the training rows) correlation of the read-out with the target.
Twin      : the same REVE architecture with random initialisation (AutoModel.from_config, torch seed 0), identical pipeline and probe.
Band power: five-band log power (1-4, 4-8, 8-13, 13-30, 30-45 Hz; Hann window, rFFT on the GPU) per channel of the probe's own input epoch
            (image: 250 Hz 0-1 s, of the repetition-averaged epoch, as every other image arm; PE: the 200 Hz post-feedback patch window).
Raw       : raw voltage (image: 63 x 250 standardized; PE: results/features/raw, the X7 1472 features).
Read-outs (locked per target, METHOD CHANGE 13):
  image: ridge (dual, GPU) from standardized features to CLIP ViT-H/14, alpha by MRR on the dev tenth, refit on all training rows (= Q7);
         score = InfoNCE log-score (run.soft_bits, ported to torch) over the 200 official test images, k = 80 and k = 40 repetition means.
  PE   : within-participant ridge (LAM * n, in-fold z-scoring * colw, dual form) with the STANDARD_TEST nested calibration
         (experiments/standard/pe_bottleneck_logscore.py, 'full' width): held-out Gaussian log-density gain over intercept-only.
         Plus, for G2 only, the E11 LOPO PCA-64 ridge on layer-22 tokens (lopo_reve_readout.py).
Ceilings / A1 : image latent split-half identification = per test image, mean standardized Z of repetitions 1-40 vs 41-80, cosine,
         200-way InfoNCE log-score (same for raw voltage and the twin). RSA (appendix): Spearman(RDM(Z_k80), RDM(CLIP)) on the 200 test images
         (RDM = 1 - Pearson), null = 10,000 joint row/column permutations (GPU index tensors), reliability = Spearman of split-half RDMs,
         Spearman-Brown corrected.
Dedicated : image = NICE-GA (ours) frozen, InfoNCE k = 80 (results/method_pilots/r1_input_bottleneck_sub-XX_basis-sub-XX.json, L = full);
            PE = EEGNet LOPO read-out (results/features/eegnet), density gain as common_log_yield (0.110).
Floors    : image: shuffled pairs; noise input (N(0,1) after standardization) through the same model, layer and probe.
            PE: three circular shifts of the target (n/4, n/2, 3n/4); noise epochs through REVE / twin; Gaussian-noise raw features (rng 1000+s).
G2        : image REVE layer 22 k80 reproduces Q7 (outputs/AGENT_LOOP_20261120/results/q7/fm_probe_reve_sub-XX.json soft_bits);
            PE layer-22 tokens reproduce results/features/revetok (corr), the LOPO read-out reproduces results/features/reve_lopo (corr) and
            E11's gain; raw within-ridge reproduces results/standard/pe_bottleneck (eeg, pca, full).
Numerics  : all fits, scores, bootstraps and permutations in torch on the GPU (CPU = file I/O only).
Output    : results/ext/m7/<target>_<sub>.json; summary via --summarize (participant bootstrap 10,000, Holm within target).
Usage     : python m7_latent.py --target image --subs 1 [--smoke] | --target pe [--smoke] | --summarize image|pe
"""
import sys, json, time, math, argparse, hashlib
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
DEV = torch.device('cuda'); OUT = ROOT / 'results/ext/m7'; OUT.mkdir(parents=True, exist_ok=True)
TRANSDUCTIVE = 'REVE pretrained on corpora that include THINGS-EEG2 (transductive)'   # METHOD 17:20 binding label
LAYERS = (6, 11, 16, 22); BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)); TAUS = torch.logspace(-2.5, 0, 12, dtype=torch.float64)
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='image'); ap.add_argument('--subs', default='1')
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--summarize', default=''); ap.add_argument('--tag', default=''); ap.add_argument('--lopo_only', action='store_true'); ap.add_argument('--inlp_only', action='store_true'); ap.add_argument('--min_free_gb', type=float, default=12.0)
ap.add_argument('--out', default=str(ROOT / 'results/ext/m7'))   # FIXER: new output folders (m7_v2) so old files never mix
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)


# ---------------- shared GPU helpers ----------------
def soft_bits_t(S):
    """torch port of run.soft_bits (InfoNCE log-score, leave-one-query-out temperature on the 12-point grid)."""
    S = S.double(); n = S.shape[0]; r = torch.arange(n, device=S.device)
    lp = torch.log_softmax(S[None] / TAUS.to(S.device)[:, None, None], dim=2); CE = -lp[:, r, r] / math.log(2)
    loo = (CE.sum(1, keepdim=True) - CE) / (n - 1); t = loo.argmin(0)
    return float(math.log2(n) - CE[t, r].mean())


def metrics_t(P, T):
    p = P / P.norm(dim=1, keepdim=True); t = T / T.norm(dim=1, keepdim=True); S = p @ t.T; d = S.diag()
    rank = (S > d[:, None]).sum(1) + 1; n = S.shape[0]
    return dict(top1=float((rank == 1).float().mean()), two_way=float(((S < d[:, None]).sum(1).float() / (n - 1)).mean()),
                mrr=float((1.0 / rank.float()).mean()), soft_bits=soft_bits_t(S))


def resample200(X):
    """scipy.signal.resample(X, 200, axis=-1) for real input of length 250 (Fourier method, Nyquist bin doubled), on the GPU."""
    N = X.shape[-1]; Y = torch.fft.rfft(X.float(), dim=-1)[..., :101].clone(); Y[..., 100] *= 2
    return torch.fft.irfft(Y, n=200, dim=-1) * (200.0 / N)


def bandpower(X, sf):
    """five-band log power per channel; X (n, ch, t) -> (n, ch*5)."""
    t = X.shape[-1]; w = torch.hann_window(t, periodic=False, device=X.device, dtype=torch.float32)
    P = torch.fft.rfft(X.float() * w, dim=-1).abs() ** 2; f = torch.fft.rfftfreq(t, 1 / sf).to(X.device)
    B = torch.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1)
    return torch.log(B + 1e-12).reshape(len(X), -1)


def boot_ci(v, B=10000, seed=20261121):
    v = torch.as_tensor(np.asarray(v, float), device=DEV); g = torch.Generator(device=DEV).manual_seed(seed)
    ix = torch.randint(0, len(v), (B, len(v)), device=DEV, generator=g); m = v[ix].mean(1)
    return [float(torch.quantile(m, .025)), float(torch.quantile(m, .975))]


def ratio_ci(num, den, B=10000, seed=20261121):
    nu = torch.as_tensor(np.asarray(num, float), device=DEV); de = torch.as_tensor(np.asarray(den, float), device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed); ix = torch.randint(0, len(nu), (B, len(nu)), device=DEV, generator=g)
    r = nu[ix].mean(1) / de[ix].mean(1); return float(nu.mean() / de.mean()), [float(torch.quantile(r, .025)), float(torch.quantile(r, .975))]


def signflip_p(d, B=100000, seed=20261121):
    """one-sided sign-flip p for mean(d) > 0 (exact enumeration for n <= 16)."""
    d = torch.as_tensor(np.asarray(d, float), device=DEV); n = len(d)
    if n <= 16:
        S = ((torch.arange(2 ** n, device=DEV)[:, None] >> torch.arange(n, device=DEV)) & 1).double() * 2 - 1
    else:
        g = torch.Generator(device=DEV).manual_seed(seed); S = torch.randint(0, 2, (B, n), device=DEV, generator=g).double() * 2 - 1
    return float(((S * d).mean(1) >= d.mean() - 1e-12).double().mean())


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj.tolist()


def load_reve(pretrained, chans):
    from transformers import AutoModel, AutoConfig
    positions = AutoModel.from_pretrained(str(ROOT / 'external/reve/positions'), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(DEV).eval()
    if pretrained:
        m = AutoModel.from_pretrained(str(ROOT / 'external/reve/base'), trust_remote_code=True, local_files_only=True, use_safetensors=True)
    else:
        torch.manual_seed(0); cfg = AutoConfig.from_pretrained(str(ROOT / 'external/reve/base'), trust_remote_code=True, local_files_only=True)
        m = AutoModel.from_config(cfg, trust_remote_code=True)
    m = m.to(DEV).eval(); assert all(c in positions.mapping for c in chans), 'REVE position missing'
    pos = positions(list(chans)).to(DEV); assert pos.shape == (len(chans), 3)
    return m, pos


@torch.no_grad()
def reve_layers(m, pos, X, patch=-1, bs=200, cpu=False):
    """X (n, ch, t) standardized (GPU or CPU) -> {layer: (n, ch*512) float16} for the chosen time patch; cpu=True keeps outputs in host RAM."""
    out = {l: [] for l in LAYERS}
    for b in range(0, len(X), bs):
        xb = X[b:b + bs].to(DEV).float(); hs = m(xb, pos.expand(len(xb), -1, -1), return_output=True)
        nch = xb.shape[1]
        for l in LAYERS:
            h = hs[l]; h = h.reshape(len(xb), nch, -1, h.shape[-1])[:, :, patch]; out[l].append(h.reshape(len(xb), -1).half().cpu() if cpu else h.reshape(len(xb), -1).half())
    return {l: torch.cat(v) for l, v in out.items()}


# ---------------- image arm ----------------
def ridge_image(Ftr, Ytr, Ftes, seed=0):
    """Q7's dual ridge (alpha by MRR on a held-out tenth, seed 0), in torch; returns alpha, dev log-score, predictions."""
    A = Ftr.to(DEV, dtype=torch.float32, copy=True); del Ftr; fm = A.mean(0); fs = A.std(0, unbiased=False) + 1e-6
    A.sub_(fm).div_(fs); Ym = Ytr.mean(0); Y = Ytr - Ym; n, D = A.shape
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(DEV); cut = int(0.9 * n); tr, va = idx[:cut], idx[cut:]
    K = A @ A.T; Ktr = K[tr][:, tr]; Kva = K[va][:, tr]; best, ba, bdev = -1, None, None
    for al in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
        coef = torch.linalg.solve(Ktr + al * D * torch.eye(len(tr), device=DEV), Y[tr]); P = Kva @ coef + Ym
        s = metrics_t(P, Ytr[va])['mrr']
        if s > best: best, ba, bdev = s, al, P
    dev_bits = metrics_t(bdev, Ytr[va])['soft_bits']; del Ktr, Kva
    coef = torch.linalg.solve(K + ba * D * torch.eye(n, device=DEV), Y); del K
    preds = [((((F.to(DEV).float() - fm) / fs) @ A.T) @ coef) + Ym for F in Ftes]; del A; torch.cuda.empty_cache()
    return ba, dev_bits, preds


def split_half(H1, H2, fm, fs, g):
    z1 = (H1.float() - fm) / fs; z2 = (H2.float() - fm) / fs
    S = torch.nn.functional.normalize(z1, dim=1) @ torch.nn.functional.normalize(z2, dim=1).T
    return dict(split_half=soft_bits_t(S), split_half_shuffled=soft_bits_t(S[:, torch.randperm(len(S), generator=g).to(DEV)]))


def rdm(Z):
    Z = Z.double(); Z = Z - Z.mean(1, keepdim=True); Z = Z / Z.norm(dim=1, keepdim=True); return 1 - Z @ Z.T


def rsa(Zk80, Zh1, Zh2, Yte, nperm=10000, seed=20261121):
    n = len(Yte); iu = torch.triu_indices(n, n, 1, device=DEV)
    def ranks_full(R):
        v = R[iu[0], iu[1]]; rk = torch.empty_like(v); rk[v.argsort()] = torch.arange(len(v), device=DEV, dtype=v.dtype)
        M = torch.zeros(n, n, device=DEV, dtype=v.dtype); M[iu[0], iu[1]] = rk; return M + M.T, rk
    def pc(x, y):
        x = x - x.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True); return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1))
    Mz, rz = ranks_full(rdm(Zk80)); _, rc = ranks_full(rdm(Yte)); rho = float(pc(rz, rc))
    g = torch.Generator(device=DEV).manual_seed(seed); null = []
    for b in range(0, nperm, 500):
        P = torch.stack([torch.randperm(n, device=DEV, generator=g) for _ in range(min(500, nperm - b))])
        Rp = Mz[P[:, :, None], P[:, None, :]][:, iu[0], iu[1]]; null.append(pc(Rp, rc[None]))
    null = torch.cat(null); _, r1 = ranks_full(rdm(Zh1)); _, r2 = ranks_full(rdm(Zh2)); rel = float(pc(r1, r2))
    return dict(rsa_rho=rho, rsa_p=float(((null >= rho).sum() + 1) / (len(null) + 1)), rsa_null_p95=float(torch.quantile(null, .95)),
                rsa_split_half_rho=rel, rsa_reliability_sb=2 * rel / (1 + rel))


def inlp_image(Ftr, Fte, Ytr, Yte, g, rounds=10, r=25):
    """INLP null-space probe (METHOD pre-ruling #1): each round fits the locked ridge (alpha by MRR on the dev tenth, refit on all training
    rows), scores it on test, then takes the rank-r reduced-rank head (top-r right singular vectors of the training fit) and projects its
    r input directions (training rows only) out of train and test features; k = 0 is the A1 probe."""
    Z = Ftr.to(DEV, dtype=torch.float32, copy=True); fm = Z.mean(0); fs = Z.std(0, unbiased=False) + 1e-6; Z.sub_(fm).div_(fs); T = (Fte.to(DEV).float() - fm) / fs
    Ym = Ytr.mean(0); Y = Ytr - Ym; n, D = Z.shape; idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(DEV); cut = int(0.9 * n); tr, va = idx[:cut], idx[cut:]
    out = []; perm = torch.randperm(len(Yte), generator=g).to(DEV)
    for k in range(rounds + 1):
        K = Z @ Z.T; best, ba = -1, None
        for al in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            c = torch.linalg.solve(K[tr][:, tr] + al * D * torch.eye(len(tr), device=DEV), Y[tr]); sc = metrics_t(K[va][:, tr] @ c + Ym, Ytr[va])['mrr']
            if sc > best: best, ba = sc, al
        coef = torch.linalg.solve(K + ba * D * torch.eye(n, device=DEV), Y); P = (T @ Z.T) @ coef + Ym
        out.append(dict(k=k, alpha=ba, soft_bits=metrics_t(P, Yte)['soft_bits'], shuffled_pair=metrics_t(P, Yte[perm])['soft_bits']))
        Vr = torch.linalg.svd(K @ coef, full_matrices=False)[2][:r].T; Q = torch.linalg.qr(Z.T @ (coef @ Vr))[0]; del K
        Z = Z - (Z @ Q) @ Q.T; T = T - (T @ Q) @ Q.T
    del Z, T; torch.cuda.empty_cache(); return out


def cka(X, Y, nperm=10000, seed=20261121):
    """linear CKA between test-image features and CLIP (appendix only), null = 10,000 image permutations (GPU index tensors)."""
    X = X.double() - X.double().mean(0); Y = Y.double() - Y.double().mean(0); Kx = X @ X.T; Ky = Y @ Y.T; n = len(Kx)
    c = float((Kx * Ky).sum() / (Kx.norm() * Ky.norm())); g = torch.Generator(device=DEV).manual_seed(seed); null = []
    for b in range(0, nperm, 1000):
        P = torch.stack([torch.randperm(n, device=DEV, generator=g) for _ in range(min(1000, nperm - b))]); Kp = Ky[P[:, :, None], P[:, None, :]]
        null.append((Kx[None] * Kp).sum((1, 2)) / (Kx.norm() * Ky.norm()))
    null = torch.cat(null); return dict(cka=c, cka_p=float(((null >= c).sum() + 1) / (len(null) + 1)), cka_null_mean=float(null.mean()))


def wait_gpu(gb):
    """memory guard (PLAN rule 7): start only when the shared A100 has >= gb free."""
    import subprocess, os
    def free():
        q = subprocess.run(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        return float(q.strip().splitlines()[0]) / 1024 if q.strip() else 0.0
    while free() < gb:
        print(f'waiting: GPU free {free():.1f} GB < {gb}', flush=True); time.sleep(60)


def run_image(sub):
    from run import CLIP, load_torch, load_raw
    wait_gpu(a.min_free_gb)
    t0 = time.time(); sid = f'sub-{sub:02d}'; g = torch.Generator().manual_seed(sub)
    Xtr4, Xte80, meta = load_raw(sid); chans = meta['ch_names']; assert Xtr4.shape[2:] == (63, 250)
    Xtr = torch.from_numpy(Xtr4.mean(1)).to(DEV); del Xtr4
    Xte = torch.from_numpy(Xte80); del Xte80                                # (200, 80, 63, 250), host RAM; repetitions moved to the GPU one at a time
    Ytr = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_train.pt')).to(DEV); Yte = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_test.pt')).to(DEV)
    h1 = sum(Xte[:, r_].to(DEV) for r_ in range(40)) / 40; h2 = sum(Xte[:, r_].to(DEV) for r_ in range(40, 80)) / 40; k80, k40 = (h1 + h2) / 2, h1
    res = dict(target='image', dataset='THINGS-EEG2', transductive_label=TRANSDUCTIVE, subject=sid, n_train=int(len(Xtr)), smoke=a.smoke, layers=LAYERS,
               input_check=dict(reve_patch=200, reve_sf=200, overlap=20, epoch='0-1.0 s at 250 Hz -> 200 samples at 200 Hz = 1 patch', channels=len(chans)), arms={})
    def score(name, Ftr, Fk80, Fk40, Fnz_tr=None, Fnz_te=None, extra=None):
        al, dev_bits, (p80, p40) = ridge_image(Ftr, Ytr, [Fk80, Fk40])
        r = dict(alpha=al, dev_soft_bits=dev_bits, k80=metrics_t(p80, Yte), k40=metrics_t(p40, Yte),
                 shuffled_pair_k80=metrics_t(p80, Yte[torch.randperm(200, generator=g).to(DEV)]))
        if Fnz_tr is not None:
            _, _, (pn,) = ridge_image(Fnz_tr, Ytr, [Fnz_te]); r['noise_k80'] = metrics_t(pn, Yte)
        if extra: r.update(extra)
        if name.startswith('reve'): r['transductive_label'] = TRANSDUCTIVE
        res['arms'][name] = r; print(f"{sid} {name}: k80 {r['k80']['soft_bits']:.4f} k40 {r['k40']['soft_bits']:.4f} dev {dev_bits:.4f} "
                                     f"pair {r['shuffled_pair_k80']['soft_bits']:.4f} noise {r.get('noise_k80', {}).get('soft_bits', float('nan')):.4f} ({time.time()-t0:.0f}s)", flush=True)
        return r
    # raw voltage + band power (no model)
    Rtr = Xtr.reshape(len(Xtr), -1); fm, fs = Rtr.mean(0), Rtr.std(0, unbiased=False) + 1e-6
    noise_tr = torch.randn(Xtr.shape, generator=torch.Generator().manual_seed(1000 + sub)); noise_te = torch.randn((200,) + Xtr.shape[1:], generator=torch.Generator().manual_seed(2000 + sub)).to(DEV)
    sd_ch = Xtr.std((0, 2), unbiased=False, keepdim=True); mu_ch = Xtr.mean((0, 2), keepdim=True)
    sh = split_half(h1.reshape(200, -1), h2.reshape(200, -1), fm, fs, g)
    rs = rsa(((k80.reshape(200, -1) - fm) / fs), (h1.reshape(200, -1) - fm) / fs, (h2.reshape(200, -1) - fm) / fs, Yte)
    ck = cka((k80.reshape(200, -1) - fm) / fs, Yte); res['cka_selfcheck'] = cka((k80.reshape(200, -1) - fm) / fs, k80.reshape(200, -1), nperm=10)['cka']
    nz = lambda: noise_tr.to(DEV) * sd_ch + mu_ch                            # noise at the data's scale, created on the GPU only when used
    score('raw', Rtr, k80.reshape(200, -1), k40.reshape(200, -1), nz().reshape(len(Xtr), -1), (noise_te * sd_ch + mu_ch).reshape(200, -1), {**sh, **rs, **ck})
    res['arms']['raw']['inlp'] = inlp_image(Rtr, k80.reshape(200, -1), Ytr, Yte, g); torch.cuda.empty_cache()
    res['arms']['raw']['inlp_noise'] = inlp_image(nz().reshape(len(Xtr), -1), (noise_te * sd_ch + mu_ch).reshape(200, -1), Ytr, Yte, g); torch.cuda.empty_cache()
    print(sid, 'raw INLP', [round(x['soft_bits'], 3) for x in res['arms']['raw']['inlp']], f'({time.time()-t0:.0f}s)', flush=True)
    score('bandpower', torch.cat([bandpower(Xtr[b:b + 2000], 250) for b in range(0, len(Xtr), 2000)]), bandpower(k80, 250), bandpower(k40, 250),
          torch.cat([bandpower(noise_tr[b:b + 2000].to(DEV) * sd_ch + mu_ch, 250) for b in range(0, len(Xtr), 2000)]), bandpower(noise_te * sd_ch + mu_ch, 250))
    del Rtr; torch.cuda.empty_cache()
    # REVE pretrained and random-init twin
    Ztr = resample200(Xtr); zmu, zsd = Ztr.mean((0, 2), keepdim=True), Ztr.std((0, 2), unbiased=False, keepdim=True) + 1e-6
    std = lambda Z: (Z - zmu) / zsd
    del Xtr; torch.cuda.empty_cache()
    for name, pre in (('reve', True), ('reve_twin', False)):
        m, pos = load_reve(pre, chans)
        Ftr = reve_layers(m, pos, std(Ztr), cpu=True); F80 = reve_layers(m, pos, std(resample200(k80))); F40 = reve_layers(m, pos, std(resample200(k40)))
        H1 = {l: 0 for l in LAYERS}; H2 = {l: 0 for l in LAYERS}
        for r_ in range(80):                                               # single repetitions -> half means of Z
            Fr = reve_layers(m, pos, std(resample200(Xte[:, r_].to(DEV))))
            for l in LAYERS:
                if r_ < 40: H1[l] = H1[l] + Fr[l].float() / 40
                else: H2[l] = H2[l] + Fr[l].float() / 40
        per = {}
        for l in LAYERS:                                                   # layer choice on dev only
            al, dev_bits, _ = ridge_image(Ftr[l], Ytr, []); per[l] = dict(alpha=al, dev_soft_bits=dev_bits)
        lc = max(LAYERS, key=lambda l: per[l]['dev_soft_bits'])
        print(f'{sid} {name} dev log-score by layer', {l: round(v["dev_soft_bits"], 4) for l, v in per.items()}, '-> layer', lc, flush=True)
        # FIXER (METHOD layer rule, pre-stated): test k80 log-score at EVERY candidate layer, both models (appendix table only; never used to choose)
        for l in LAYERS:
            _, _, (p80,) = ridge_image(Ftr[l], Ytr, [F80[l]]); per[l]['test_k80'] = metrics_t(p80, Yte)
        print(f'{sid} {name} test k80 by layer', {l: round(v["test_k80"]["soft_bits"], 4) for l, v in per.items()}, flush=True)
        for l in LAYERS:
            if l not in (lc, 22): del Ftr[l], F80[l], F40[l], H1[l], H2[l]
        torch.cuda.empty_cache()
        Fn_tr = torch.cat([reve_layers(m, pos, std(resample200(noise_tr[b:b + 2000].to(DEV) * sd_ch + mu_ch)), cpu=True)[lc] for b in range(0, len(noise_tr), 2000)]); Fn_te = reve_layers(m, pos, std(resample200(noise_te * sd_ch + mu_ch)))[lc]
        f = Ftr[lc].to(DEV).float(); fm_, fs_ = f.mean(0), f.std(0, unbiased=False) + 1e-6; del f
        # FIXER (R2 ruling 3, C4): ceiling = latent of each split half's AVERAGED EEG (same construction as the yield; F40 = latent of h1);
        # the old mean-of-single-repetition-latents value is kept as split_half_mean_of_latents
        Fh2 = reve_layers(m, pos, std(resample200(h2)))
        sh = split_half(F40[lc], Fh2[lc], fm_, fs_, g); sh_old = split_half(H1[lc], H2[lc], fm_, fs_, g)
        sh.update(split_half_mean_of_latents=sh_old['split_half'], split_half_mean_of_latents_shuffled=sh_old['split_half_shuffled'], ceiling_construction='latent of half-mean EEG (k40 halves)')
        f22 = Ftr[22].to(DEV).float(); fm22, fs22 = f22.mean(0), f22.std(0, unbiased=False) + 1e-6; del f22
        sh22 = split_half(F40[22], Fh2[22], fm22, fs22, g); sh22o = split_half(H1[22], H2[22], fm22, fs22, g)
        sh22.update(split_half_mean_of_latents=sh22o['split_half'], split_half_mean_of_latents_shuffled=sh22o['split_half_shuffled'], ceiling_construction='latent of half-mean EEG (k40 halves)')
        del Fh2
        rs = rsa((F80[lc].float() - fm_) / fs_, (H1[lc] - fm_) / fs_, (H2[lc] - fm_) / fs_, Yte)
        score(name, Ftr[lc], F80[lc], F40[lc], Fn_tr, Fn_te, dict(layer=lc, dev_by_layer=per, **sh, **rs, **cka((F80[lc].float() - fm_) / fs_, Yte)))
        res['arms'][name]['inlp'] = inlp_image(Ftr[lc], F80[lc], Ytr, Yte, g); torch.cuda.empty_cache(); res['arms'][name]['inlp_noise'] = inlp_image(Fn_tr, Fn_te, Ytr, Yte, g)
        print(sid, name, 'INLP', [round(x['soft_bits'], 3) for x in res['arms'][name]['inlp']], f'({time.time()-t0:.0f}s)', flush=True)
        # final layer (22) for BOTH models (METHOD layer rule (ii)); reve_L22 is also the Q7 G2 arm; duplicate of the primary when lc == 22
        score(f'{name}_L22', Ftr[22], F80[22], F40[22], extra=sh22)
        del Ftr, F80, F40, H1, H2, Fn_tr, Fn_te, m; torch.cuda.empty_cache()
    q7 = ROOT / f'outputs/AGENT_LOOP_20261120/results/q7/fm_probe_reve_{sid}.json'
    if q7.exists() and not a.smoke:
        q = json.loads(q7.read_text())['rows'][0]['soft_bits']; mine = res['arms'].get('reve_L22', res['arms']['reve'])['k80']['soft_bits']
        res['g2_q7'] = dict(q7_k80_soft_bits=q, ours_layer22_k80=mine, abs_diff=abs(q - mine))
    r1 = ROOT / f'results/method_pilots/r1_input_bottleneck_{sid}_basis-{sid}.json'
    res['dedicated_nicega_k80'] = next(r['soft_bits'] for r in json.loads(r1.read_text())['rows'] if r['L'] == 'full' and r['k'] == 80 and r['condition'] == 'eeg')
    res['runtime_s'] = round(time.time() - t0, 1); res['gpu_peak_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (OUT / f"image_{sid}{'_smoke' if a.smoke else ''}{a.tag}.json").write_text(json.dumps(res, indent=1)); print('wrote', sid, res['runtime_s'], 's', flush=True)


# ---------------- PE arm ----------------
LN2 = math.log(2)


def gain_rows_t(y_tr, x_tr, y_te, x_te):
    """torch port of pe_bottleneck_logscore.gain_rows (calibration on training rows only); y_* (n, J), x_* (n, J)."""
    out = []
    for j in range(y_tr.shape[1]):
        D = torch.stack([torch.ones_like(x_tr[:, j]), x_tr[:, j]], 1); c = torch.linalg.lstsq(D, y_tr[:, j:j + 1]).solution[:, 0]
        pt = D @ c; pe = c[0] + c[1] * x_te[:, j]; rv = ((y_tr[:, j] - pt) ** 2).mean(); m = y_tr[:, j].mean(); vv = ((y_tr[:, j] - m) ** 2).mean()
        out.append((torch.log(vv / rv) + (y_te[:, j] - m) ** 2 / vv - (y_te[:, j] - pe) ** 2 / rv) / (2 * LN2))
    return torch.stack(out, 1)


def ridge_dual_pe(F, colw, Y, tr, te, lam):
    A = F[tr]; mu = A.mean(0); sd = A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * colw; B = (F[te] - mu) / sd * colw
    Yc = Y[tr] - Y[tr].mean(0); K = A @ A.T
    return (B @ A.T) @ torch.linalg.solve(K + lam * len(A) * torch.eye(len(A), dtype=K.dtype, device=DEV), Yc)


def nested_pe(F, colw, Ys, fo, lam):
    """STANDARD_TEST nested calibration, full width. Returns outer test predictions (n, 4), inner-CV preds per outer fold, gains (n, 4)."""
    from belief_bottleneck import folds
    n = len(Ys); G = torch.full((n, 4), float('nan'), dtype=torch.float64, device=DEV); P = torch.zeros(n, 4, dtype=torch.float64, device=DEV); inner_r = {}
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; trt, tet = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
        Po = ridge_dual_pe(F, colw, Ys, trt, tet, lam); fi = folds(len(tr), 9); Pin = torch.zeros(len(tr), 4, dtype=torch.float64, device=DEV)
        for c in range(9):
            itr, ite = torch.tensor(tr[fi != c], device=DEV), torch.tensor(tr[fi == c], device=DEV); Pin[torch.tensor(fi == c, device=DEV)] = ridge_dual_pe(F, colw, Ys, itr, ite, lam)
        G[tet] = gain_rows_t(Ys[trt], Pin, Ys[tet], Po); P[tet] = Po
        inner_r[b] = float(torch.corrcoef(torch.stack([Pin[:, 0], Ys[trt, 0]]))[0, 1])
    return P, inner_r, G


def inlp_pe(Fb, Ys, fo, lam, rounds=10):
    """INLP null-space probe for PE (rank-1 head = the ridge direction), v2 (no leakage): every fit (the outer fit and each of the 9 inner
    calibration fits of every outer fold) runs its OWN projection chain estimated on its OWN training rows only; round k scores the
    held-out rows with the k-times-projected model of the chain that never saw them (features z-scored once per outer fold on its
    training rows; dual ridge LAM * n; STANDARD_TEST nested calibration). Fb(b) -> (n, D) features for fold b."""
    from belief_bottleneck import folds
    n = len(Ys); G = torch.full((rounds + 1, n, 4), float('nan'), dtype=torch.float64, device=DEV)
    def chain(A, r_tr, r_te):                                              # -> [rounds+1] predictions for r_te
        A = A.clone(); out = []
        for k in range(rounds + 1):
            At = A[r_tr]; Yc = Ys[r_tr] - Ys[r_tr].mean(0); c = torch.linalg.solve(At @ At.T + lam * len(r_tr) * torch.eye(len(r_tr), dtype=A.dtype, device=DEV), Yc)
            out.append((A[r_te] @ At.T) @ c); w = At.T @ c[:, 0]; q = w / w.norm(); A = A - (A @ q)[:, None] * q[None]
        return out
    for b in range(10):
        F = Fb(b).double(); tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; trt, tet = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
        mu = F[trt].mean(0); sd = F[trt].std(0, unbiased=False) + 1e-6; A = (F - mu) / sd; fi = folds(len(tr), 9)
        Po = chain(A, trt, tet); Pin = [torch.zeros(len(tr), 4, dtype=torch.float64, device=DEV) for _ in range(rounds + 1)]
        for c_ in range(9):
            pk = chain(A, torch.tensor(tr[fi != c_], device=DEV), torch.tensor(tr[fi == c_], device=DEV)); m_ = torch.tensor(fi == c_, device=DEV)
            for k in range(rounds + 1): Pin[k][m_] = pk[k]
        for k in range(rounds + 1): G[k, tet] = gain_rows_t(Ys[trt], Pin[k], Ys[tet], Po[k])
    return [dict(k=k, gain=float(G[k, :, 0].mean()), shifted=[float(v) for v in G[k, :, 1:].mean(0)]) for k in range(rounds + 1)]


def density_gain_t(y, x, fo):
    y = torch.as_tensor(y, dtype=torch.float64, device=DEV); x = torch.as_tensor(x, dtype=torch.float64, device=DEV); s = torch.zeros_like(y)
    for b in np.unique(fo):
        trm = torch.tensor(fo != b, device=DEV); s[~trm] = gain_rows_t(y[trm][:, None], x[trm][:, None], y[~trm][:, None], x[~trm][:, None])[:, 0]
    return float(s.mean())


def run_pe():
    from belief_bottleneck import LAM, folds
    sys.path.insert(0, str(ROOT / 'experiments/loop'))
    src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); src = src[:src.index('DATASETS = {')]; ns = {}; exec(compile(src, 's13b_head', 'exec'), ns)
    qtil, subs, fpath, _ = ns['targets_ds003458'](); wait_gpu(8.0); t0 = time.time()
    run_subs = subs[:2] if a.smoke else subs
    from transformers import AutoModel
    positions = AutoModel.from_pretrained(str(ROOT / 'external/reve/positions'), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(DEV).eval()
    known = set(positions.mapping)
    mods = {}
    for name, pre in (('reve', True), ('reve_twin', False)):
        from transformers import AutoConfig
        if pre: m = AutoModel.from_pretrained(str(ROOT / 'external/reve/base'), trust_remote_code=True, local_files_only=True, use_safetensors=True)
        else:
            torch.manual_seed(0); m = AutoModel.from_config(AutoConfig.from_pretrained(str(ROOT / 'external/reve/base'), trust_remote_code=True, local_files_only=True), trust_remote_code=True)
        mods[name] = m.to(DEV).eval()
    tok22, ys_all, per = {}, {}, {}
    if a.lopo_only:                                                        # resume: per-participant files exist; layer-22 tokens = cached revetok (G2 corr 1.0000)
        for s in run_subs:
            per[s] = json.loads((OUT / f'pe_sub-{s:03d}.json').read_text()); tri = np.load(fpath('revetok', s))['tri']
            tok22[s] = torch.tensor(np.load(fpath('revetok', s))['F'].astype(np.float32), device=DEV); ys_all[s] = torch.tensor(qtil(s, tri), dtype=torch.float64, device=DEV)
        run_subs = []
    if a.inlp_only: run_subs = [x for x in run_subs if 'inlp_v2' not in json.loads((OUT / f'pe_sub-{x:03d}.json').read_text())]
    for s in run_subs:
        tri = np.load(fpath('fm', s))['tri']; E = np.load(ROOT / f'results/fm_epochs/sub-{s:03d}.npz'); pidx = {t: k for k, t in enumerate(E['trial'])}
        names = [str(c) for c in E['ch']]; keep = [k for k, c in enumerate(names) if c in known]
        X = E['X'][[pidx[t] for t in tri]][:, keep].reshape(len(tri), len(keep), -1).astype(np.float32)
        X = torch.from_numpy(X).to(DEV); X = (X - X.mean((0, 2), keepdim=True)) / (X.std((0, 2), unbiased=False, keepdim=True) + 1e-6)
        pos = positions([names[k] for k in keep]).to(DEV)
        Xn = torch.randn(X.shape, generator=torch.Generator().manual_seed(3000 + s)).to(DEV)
        y = torch.tensor(qtil(s, tri), dtype=torch.float64, device=DEV); n = len(y); ys_all[s] = y
        Ys = torch.stack([y] + [torch.roll(y, round(n * k / 4)) for k in (1, 2, 3)], 1); fo = folds(n)
        raw = np.load(fpath('raw', s)); assert (raw['tri'] == tri).all()
        Fraw = torch.tensor(raw['F'], dtype=torch.float64, device=DEV); cw = torch.tensor(raw['colw'], dtype=torch.float64, device=DEV)
        Fraw_n = torch.tensor(np.random.default_rng(1000 + s).standard_normal(tuple(Fraw.shape)), device=DEV)
        r = dict(subject=f'sub-{s:03d}', n_trials=n, channels_with_positions=len(keep), channels=len(names),
                 input_check=dict(reve_patch=200, reve_sf=200, overlap=20, epoch='-1..1 s at 200 Hz = 400 samples -> 2 patches; post-feedback patch (-0.1..0.9 s) used'), arms={})
        if a.inlp_only:                                                    # INLP v2 only, layers as chosen (inner-CV) in the full run; merged into the file
            old = json.loads((OUT / f'pe_sub-{s:03d}.json').read_text()); old['inlp_v1_leaky'] = old.pop('inlp', None); iv = {}
            rw = X[:, :, 180:380].reshape(n, X.shape[1], 50, 4).mean(-1).reshape(n, -1).double(); rwn = Xn[:, :, 180:380].reshape(n, X.shape[1], 50, 4).mean(-1).reshape(n, -1).double()
            iv['raw_win'] = inlp_pe(lambda b: rw, Ys, fo, LAM); iv['raw_win_noise'] = inlp_pe(lambda b: rwn, Ys, fo, LAM)
            for name, m in mods.items():
                for inp, XX in (('', X), ('_noise', Xn)):
                    Fl = reve_layers(m, pos, XX, patch=-1, bs=64); ch = {int(k): v for k, v in old['arms'][name + inp]['layers'].items()}
                    iv[name + inp] = inlp_pe(lambda b: Fl[ch[b]], Ys, fo, LAM); del Fl; torch.cuda.empty_cache()
            old['inlp_v2'] = iv; (OUT / f'pe_sub-{s:03d}.json').write_text(json.dumps(old, indent=1))
            print(f'sub-{s:03d} INLP v2', {k: [round(x['gain'], 3) for x in v][:4] for k, v in iv.items()}, f'({time.time()-t0:.0f}s)', flush=True); continue
        def summ(name, G, extra=None):
            d = dict(gain=float(G[:, 0].mean()), shifted=[float(v) for v in G[:, 1:].mean(0)], **(extra or {})); r['arms'][name] = d
            print(f"sub-{s:03d} {name}: gain {d['gain']:.4f} shift {np.mean(d['shifted']):.4f}", {k: v for k, v in (extra or {}).items() if k == 'layers'}, f'({time.time()-t0:.0f}s)', flush=True)
        _, _, G = nested_pe(Fraw, cw, Ys, fo, LAM); summ('raw', G)
        _, _, G = nested_pe(Fraw_n, cw, Ys, fo, LAM); summ('raw_noise', G)
        win = slice(180, 380)                                              # samples of the post-feedback REVE patch
        rw = X[:, :, win].reshape(n, X.shape[1], 50, 4).mean(-1).reshape(n, -1).double(); rwn = Xn[:, :, win].reshape(n, X.shape[1], 50, 4).mean(-1).reshape(n, -1).double()
        onew = torch.ones(rw.shape[1], dtype=torch.float64, device=DEV)
        _, _, G = nested_pe(rw, onew, Ys, fo, LAM); summ('raw_win', G, dict(window='-0.1..0.9 s post-feedback, 20-ms bins (REVE patch window; METHOD G1 condition 1)'))
        _, _, G = nested_pe(rwn, onew, Ys, fo, LAM); summ('raw_win_noise', G)
        r['inlp'] = {'raw_win': inlp_pe(lambda b: rw, Ys, fo, LAM), 'raw_win_noise': inlp_pe(lambda b: rwn, Ys, fo, LAM)}
        bp = bandpower(X[:, :, win], 200).double(); one = lambda F: torch.ones(F.shape[1], dtype=torch.float64, device=DEV)
        _, _, G = nested_pe(bp, one(bp), Ys, fo, LAM); summ('bandpower', G)
        bpn = bandpower(Xn[:, :, win], 200).double(); _, _, G = nested_pe(bpn, one(bpn), Ys, fo, LAM); summ('bandpower_noise', G)
        for name, m in mods.items():
            for inp, XX in (('', X), ('_noise', Xn)):
                Fl = reve_layers(m, pos, XX, patch=-1, bs=64); outs = {l: nested_pe(Fl[l].double(), one(Fl[l]), Ys, fo, LAM) for l in LAYERS}
                Gs = torch.full((n, 4), float('nan'), dtype=torch.float64, device=DEV); chosen = {}
                for b in range(10):                                        # layer chosen per outer fold on inner-CV only
                    lb = max(LAYERS, key=lambda l: outs[l][1][b]); chosen[b] = lb; m_ = torch.tensor(fo == b, device=DEV); Gs[m_] = outs[lb][2][m_]
                summ(name + inp, Gs, dict(layers=chosen, by_layer={l: float(outs[l][2][:, 0].mean()) for l in LAYERS}))
                r['inlp'][name + inp] = inlp_pe(lambda b: Fl[chosen[b]], Ys, fo, LAM)
                print(f'sub-{s:03d} {name+inp} INLP', [round(x["gain"], 4) for x in r['inlp'][name + inp]], flush=True)
                if name == 'reve' and inp == '': tok22[s] = Fl[22].float(); r['g2_revetok_corr'] = float(torch.corrcoef(torch.stack([Fl[22].float().flatten(), torch.tensor(np.load(fpath('revetok', s))['F'].astype(np.float32), device=DEV).flatten()]))[0, 1])
                del Fl, outs; torch.cuda.empty_cache()
        ez = np.load(fpath('eegnet', s)); r['dedicated_eegnet'] = density_gain_t(qtil(s, ez['tri']), ez['F'][:, 0], folds(len(ez['tri'])))
        std_f = ROOT / f'results/standard/pe_bottleneck/sub-{s:03d}.json'
        if std_f.exists():
            sr = next(x for x in json.loads(std_f.read_text())['rows'] if x['input'] == 'eeg' and x['mode'] == 'pca' and x['L'] == 'full')
            r['g2_raw_standard'] = dict(standard=sr['gain'], ours=r['arms']['raw']['gain'], abs_diff=abs(sr['gain'] - r['arms']['raw']['gain']))
        per[s] = r; r['runtime_s'] = round(time.time() - t0, 1); (OUT / f"pe_sub-{s:03d}{'_smoke' if a.smoke else ''}{a.tag}.json").write_text(json.dumps(r, indent=1))
    # E11 LOPO PCA-64 ridge on layer-22 tokens (G2), as lopo_reve_readout.py (float32, per-participant z-scoring)
    if len(tok22) == len(subs) and not a.inlp_only:
        feats = {s: (T - T.mean(0)) / (T.std(0, unbiased=False) + 1e-6) for s, T in tok22.items()}
        for h in subs:
            A = torch.cat([feats[k] for k in subs if k != h]); yv = torch.cat([ys_all[k].float() for k in subs if k != h]); mu = A.mean(0)
            Ac = A - mu; del A; w, U = torch.linalg.eigh((Ac @ Ac.T).double()); Vk = ((U[:, -64:].T.float() @ Ac) / w[-64:].clamp_min(1e-12).sqrt().float()[:, None]).T
            Atr = Ac @ Vk; W = torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(64, device=DEV), Atr.T @ (yv - yv.mean()))
            ph = ((feats[h] - mu) @ Vk @ W).double(); cached = np.load(fpath('reve_lopo', h))['F'][:, 0]
            fo = folds(len(ph)); per[h]['reve_lopo'] = dict(gain=density_gain_t(ys_all[h], ph, fo), cached_gain=density_gain_t(ys_all[h], cached, fo),
                                                             corr_with_cached=float(np.corrcoef(ph.cpu().numpy(), cached)[0, 1]))
            del Ac, U, w; torch.cuda.empty_cache()
    for s, r in (per.items() if not a.inlp_only else []):
        r['runtime_s'] = round(time.time() - t0, 1); (OUT / f"pe_sub-{s:03d}{'_smoke' if a.smoke else ''}{a.tag}.json").write_text(json.dumps(r, indent=1))
    print('wrote', len(per), 'participants', round(time.time() - t0), 's', flush=True)


# ---------------- summary ----------------
def summarize(target):
    files = sorted(OUT.glob(f'{target}_sub-*.json')); files = [f for f in files if 'smoke' not in f.name]; R = [json.loads(f.read_text()) for f in files]
    out = dict(target=target, n=len(R), participants=[r['subject'] for r in R], arms={}, note=__doc__)
    if target == 'image': out['transductive_label'] = TRANSDUCTIVE
    if target == 'image':
        ded = [r['dedicated_nicega_k80'] for r in R]; out['dedicated_nicega_k80'] = dict(mean=float(np.mean(ded)), ci=boot_ci(ded))
        fam = []
        for arm in [x for x in R[0]['arms'] if all(x in r['arms'] for r in R)]:
            v = [r['arms'][arm]['k80']['soft_bits'] for r in R]; fl = [r['arms'][arm]['shuffled_pair_k80']['soft_bits'] for r in R]
            d = dict(k80=dict(mean=float(np.mean(v)), ci=boot_ci(v)), k40=dict(mean=float(np.mean([r['arms'][arm]['k40']['soft_bits'] for r in R]))),
                     shuffled_pair=float(np.mean(fl)), n_above_floor=int(sum(x > y for x, y in zip(v, fl))))
            if 'noise_k80' in R[0]['arms'][arm]: d['noise_k80'] = float(np.mean([r['arms'][arm]['noise_k80']['soft_bits'] for r in R]))
            fr, frci = ratio_ci(v, ded); d['fraction_of_dedicated'] = dict(ratio_of_means=fr, ci=frci, per_participant_mean=float(np.mean(np.array(v) / np.array(ded))))
            if 'inlp' in R[0]['arms'][arm]:
                for nm in ('inlp', 'inlp_noise'):
                    d[nm] = [dict(k=k, soft_bits=float(np.mean([r['arms'][arm][nm][k]['soft_bits'] for r in R])), ci=boot_ci([r['arms'][arm][nm][k]['soft_bits'] for r in R]),
                                  shuffled_pair=float(np.mean([r['arms'][arm][nm][k]['shuffled_pair'] for r in R]))) for k in range(11)]
            for k in ('split_half', 'split_half_shuffled', 'split_half_mean_of_latents', 'rsa_rho', 'rsa_p', 'rsa_null_p95', 'rsa_reliability_sb', 'cka', 'cka_p', 'cka_null_mean'):
                if k in R[0]['arms'][arm]:
                    vv = [r['arms'][arm][k] for r in R]; d[k] = dict(mean=float(np.mean(vv)), ci=boot_ci(vv))
            if 'layer' in R[0]['arms'][arm]:
                d['layers_chosen'] = [r['arms'][arm]['layer'] for r in R]
                if all('test_k80' in r['arms'][arm]['dev_by_layer'][str(LAYERS[0])] for r in R):   # FIXER: per-layer appendix table
                    d['per_layer'] = {str(l): dict(dev=float(np.mean([r['arms'][arm]['dev_by_layer'][str(l)]['dev_soft_bits'] for r in R])),
                                                   test_k80=float(np.mean([r['arms'][arm]['dev_by_layer'][str(l)]['test_k80']['soft_bits'] for r in R]))) for l in LAYERS}
            d['p_yield_gt_floor'] = signflip_p(np.array(v) - np.array(fl)); fam.append(('yield>floor ' + arm, d['p_yield_gt_floor']))
            if arm.startswith('reve'): d['transductive_label'] = TRANSDUCTIVE
            out['arms'][arm] = d
        for x, y in (('reve', 'reve_twin'), ('reve_L22', 'reve_twin_L22'), ('reve', 'bandpower'), ('reve', 'raw')):
            if not all(x in r['arms'] and y in r['arms'] for r in R): continue      # FIXER: reve_twin_L22 exists only in v2 files
            dv = [r['arms'][x]['k80']['soft_bits'] - r['arms'][y]['k80']['soft_bits'] for r in R]
            out[f'{x}_minus_{y}_k80'] = dict(mean=float(np.mean(dv)), ci=boot_ci(dv), p=signflip_p(dv), n_pos=int(sum(np.array(dv) > 0))); fam.append((f'{x}>{y}', out[f'{x}_minus_{y}_k80']['p']))
            ds = [r['arms'][x]['split_half'] - r['arms'][y]['split_half'] for r in R if 'split_half' in r['arms'][y]]
            if ds: out[f'{x}_minus_{y}_split_half'] = dict(mean=float(np.mean(ds)), ci=boot_ci(ds), p=signflip_p(ds)); fam.append((f'{x}>{y} split-half', out[f'{x}_minus_{y}_split_half']['p']))
        g2 = [r['g2_q7']['abs_diff'] for r in R if 'g2_q7' in r]; out['g2_q7_max_abs_diff'] = max(g2) if g2 else None
    else:
        ded = [r['dedicated_eegnet'] for r in R]; out['dedicated_eegnet'] = dict(mean=float(np.mean(ded)), ci=boot_ci(ded)); fam = []
        for arm in R[0]['arms']:
            v = [r['arms'][arm]['gain'] for r in R]; fl = [np.mean(r['arms'][arm]['shifted']) for r in R]; fr, frci = ratio_ci(v, ded)
            out['arms'][arm] = d = dict(gain=dict(mean=float(np.mean(v)), ci=boot_ci(v)), shift_floor=float(np.mean(fl)), n_above_floor=int(sum(x > y for x, y in zip(v, fl))),
                                        fraction_of_dedicated=dict(ratio_of_means=fr, ci=frci))
            if not arm.endswith('noise'): d['p_yield_gt_floor'] = signflip_p(np.array(v) - np.array(fl)); fam.append(('yield>floor ' + arm, d['p_yield_gt_floor']))
        v1 = [r['arms']['reve']['gain'] for r in R]; v2 = [r['arms']['raw_win']['gain'] for r in R]; fr, frci = ratio_ci(v1, v2)
        out['fraction_within_regime'] = dict(label='REVE within-participant ridge / raw voltage on the same -0.1..0.9 s window (within-participant ridge)', ratio_of_means=fr, ci=frci)
        ik = 'inlp_v2' if all('inlp_v2' in r for r in R) else 'inlp'; out['inlp_version'] = ik
        out['inlp'] = {nm: [dict(k=k, gain=float(np.mean([r[ik][nm][k]['gain'] for r in R])), gain_ci=boot_ci([r[ik][nm][k]['gain'] for r in R]),
                                shift=float(np.mean([np.mean(r[ik][nm][k]['shifted']) for r in R]))) for k in range(11)] for nm in R[0][ik]}
        for x, y in (('reve', 'reve_twin'), ('reve', 'bandpower'), ('reve', 'raw_win'), ('reve', 'raw')):
            dv = [r['arms'][x]['gain'] - r['arms'][y]['gain'] for r in R]
            out[f'{x}_minus_{y}'] = dict(mean=float(np.mean(dv)), ci=boot_ci(dv), p=signflip_p(dv), n_pos=int(sum(np.array(dv) > 0))); fam.append((f'{x}>{y}', out[f'{x}_minus_{y}']['p']))
        L = [r['reve_lopo'] for r in R if 'reve_lopo' in r]
        if L:
            v = [x['gain'] for x in L]; fr, frci = ratio_ci(v, ded[:len(v)])
            out['reve_lopo_E11'] = dict(gain=dict(mean=float(np.mean(v)), ci=boot_ci(v)), cached_gain=float(np.mean([x['cached_gain'] for x in L])),
                                        min_corr_with_cached=float(min(x['corr_with_cached'] for x in L)),
                                        fraction_cross_regime=dict(label='REVE LOPO (E11) / EEGNet LOPO', ratio_of_means=fr, ci=frci))
        out['g2_raw_standard_max_abs_diff'] = max((r['g2_raw_standard']['abs_diff'] for r in R if 'g2_raw_standard' in r), default=None)
        out['g2_revetok_min_corr'] = min((r.get('g2_revetok_corr', 1) for r in R), default=None)
    adj = holm([p for _, p in fam]); out['holm'] = {k: dict(p=p, p_holm=q) for (k, p), q in zip(fam, adj)}
    (OUT / f'summary_{target}.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1)[:6000])


if __name__ == '__main__':
    if a.summarize: summarize(a.summarize)
    elif a.target == 'image':
        for s in [int(x) for x in a.subs.split(',')]:
            if (OUT / f"image_sub-{s:02d}{'_smoke' if a.smoke else ''}{a.tag}.json").exists() and not a.smoke: continue
            run_image(s)
    else: run_pe()
