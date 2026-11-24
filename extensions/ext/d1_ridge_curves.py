"""D1 RIDGE YIELD CURVES (B7: D1.1-D1.4 + D5.2 simulated montages; rungs R4/R5/R9). EXECUTOR-3, 21 Nov 2026.
Design = EXPERIMENT_TABLES_FINAL.md D1.1-D1.4, D5.2 + "## D1 REVIEW v1" / "## D5 REVIEW v1" binding changes (EXTENSIONS_LOOP_20261121.md).

Read-out (identical to the base raw ridge, experiments/eegbits/run.py --input raw): training input = mean of the 4 repetitions,
  restricted to the mask (channels x samples), flattened; standardise (train stats) -> PCA on training rows only (run.pca_scores) ->
  ridge -> CLIP ViT-H/14. Alpha per L by run.pick_alpha's rule (90/10 split of TRAINING rows, max MRR over run.ALPHAS, first maximum),
  one rng = default_rng(0) carried through L = 2,5,10,25,50,100,250,Lmax (run.py order), (batched GPU: masks with equal D share one batched PCA (float64 Gram eigh) and one batched eigh-ridge per L).
  Lmax = min(1024, D). Test set never used for any choice. Candidates N = 200 at every point (queries split for D1.3, never candidates).
Masks (fixed by NAME before any outcome; missing names are dropped and reported, never substituted; THINGS-EEG2 has no Fz/Fpz/PO9/PO10):
  all            63 ch x 0-996 ms (250 samples; data start at stimulus onset, METHOD TIME-AXIS RULING)
  groups (8 ch, 10-10 names, size-matched): occipital, parietal, temporal, central, frontal; frontpole = Fp1 Fp2 AF7 AF3 AFz AF4 AF8 Fpz
             ("ocular-sensitive": no EOG in THINGS-EEG2, so its yield is an upper bound on any ocular contribution; Fpz absent -> 7 ch)
  random8 x 20 size-matched random channel draws (seed 20261121), full window: the reference for the 8-ch groups; random7 x 20
                 (seed 20261122) = the reference for the 7-ch frontpole group (condition 1). Groups/devices live in regions.py (condition 2).
  win100s50      19 sliding 100-ms windows, 50-ms step, 0-1000 ms, all 63 ch (windows > 200 ms contain the next RSVP image: noise, not leakage)
  cum            10 cumulative windows 0-t, t = 100..1000 ms, all 63 ch
  devices (D5.2) tentwenty32 (Alljoined card does not list the montage -> the 10-20 32-ch standard, rule D5 REVIEW 2), emotiv14, cyton8
Score (METHOD G1 B7 condition 3): PRIMARY = 'soft12' = run.soft_bits (InfoNCE log-score, 12-point LOO temperature; the StdImg*
  variant, G2 0.2315); SENSITIVITY = 'ls' (same, 40-point grid); top-1, 2-way, MRR beside them.
  Raw bits at k = 80 (labelled). Fraction of own ceiling at matched k = 40 (group level = ratio of participant means; per-participant ratios are unstable near floor): yield on the 40-repetition halves (reps 0-39, 40-79; mean
  of the two) / same-mask split-half ceiling (half A vs half B of the test recording, z-scored over the 200 items, cosine; the
  smoke_splithalf.py raw-voltage ceiling restricted to the mask).
Floors (same mask, L = Lmax): noise input (N(0,1) through PCA + ridge at the selected alpha; seeds 0-2 for named masks, seed 0 for
  windows, none for random draws) and shuffled pairs (3 seeds); floor = max(mean noise, mean shuffled). Ceiling floor: shuffled pairs.
L90 per mask: run.py rule (smallest L with (2-way(L) - 0.5) >= 0.9 (2-way(Lmax) - 0.5)) and the same rule on 'ls'.
D1.3 query halves (scored among all 200 candidates; per-query bits with the temperature chosen leave-one-query-out over ALL 200
  queries, then averaged within the half; condition 4): random half (seed 7, fixed) A/B = PRIMARY; animate
  (38 animal concepts by name, ANIMATE below) vs inanimate.
--summarize: D1.1 group - mean(random8) yield, exact sign-flip p (one-sided); D1.2 window/cum yield - floor, sign-flip p; Holm over the
  35 points (6 groups + 19 windows + 10 cum); peak window in 100-300 ms (P1c, >= 8/10); D1.3 per-participant Spearman between half
  curves (35 points), median + bootstrap CI; D1.4 LOO-mean vs flat curve MAE on curves normalised to the full-mask yield, paired
  bootstrap + sign test; D5.2 device - all (yield, fraction) with Holm over 3 + monotonicity. Participant bootstrap 10,000.
G2: mask 'all' at L = 1024, k = 80 reproduces results/eegbits/raw_sub-XX.json (top-1, 2-way, MRR). G3: all floors within 0.02 bits.
Usage: python d1_ridge_curves.py --subjects 1 --out DIR [--families all,groups,...] | --summarize --out DIR
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT / "experiments/ext", ROOT / "experiments/eegbits"):
    sys.path.insert(0, str(p))
import run as R
import regions as RG

DEV = "cuda"
TAUS40 = np.logspace(-2.5, 0, 40)
LS_BASE = [2, 5, 10, 25, 50, 100, 250, 1024]
GROUPS, DEVICES, DEVICE_LABELS = RG.D1_GROUPS8, RG.D5_DEVICES, RG.D5_DEVICE_LABELS   # single source: experiments/ext/regions.py
ANIMATE = {"antelope", "beaver", "bug", "cat", "caterpillar", "cheetah", "cobra", "crab", "crow", "dalmatian", "dragonfly", "eagle", "eel",
           "elephant", "flamingo", "goose", "gopher", "gorilla", "grasshopper", "hummingbird", "lamb", "lightning_bug", "manatee", "mosquito",
           "mussel", "ostrich", "panther", "pheasant", "pigeon", "piglet", "possum", "pug", "rhinoceros", "rooster", "scallop", "seagull",
           "tick", "turkey"}
TEST_DIR = ROOT / "data_raw/things_eeg2_images/images_set/test_images"


def row_bits(S, taus):
    """Per-query InfoNCE bits log2 N - CE_i(tau_i), tau_i chosen by leave-query-i-out over ALL rows (run.soft_bits rule); mean = soft_bits."""
    n = S.shape[0]; rows = np.arange(n); CE = []
    for t in taus:
        Z = S / t; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); CE.append(-lp[rows, rows] / np.log(2))
    CE = np.stack(CE); loo = (CE.sum(1, keepdims=True) - CE) / (n - 1)
    return np.log2(S.shape[1]) - CE[loo.argmin(0), rows]


def ls40(S):
    return float(row_bits(S, TAUS40).mean())


def cos(P, Y):
    return (P / np.linalg.norm(P, axis=1, keepdims=True)) @ (Y / np.linalg.norm(Y, axis=1, keepdims=True)).T


def score(P, Y, halves=None):
    m = R.metrics(P, Y); S = cos(P, Y)
    out = dict(soft12=m["soft_bits"], ls=ls40(S), top1=m["top1"], two_way=m["two_way"], mrr=m["mrr"])
    if halves:
        b12, b40 = row_bits(S, R._TAUS), row_bits(S, TAUS40)
        assert abs(b12.mean() - m["soft_bits"]) < 1e-9
        out["halves"] = {k: dict(soft12=float(b12[v].mean()), ls=float(b40[v].mean())) for k, v in halves.items()}
    return out


def masks(ch, families):
    """name -> (family, channel indices, t0, t1, dropped names)."""
    out, allc = {}, list(range(len(ch)))
    def idx(names):
        keep = [n for n in names if n in ch]; return [ch.index(n) for n in keep], [n for n in names if n not in ch]
    if "all" in families:
        out["all"] = ("all", allc, 0, 250, [])
    if "groups" in families:
        for k, v in GROUPS.items():
            i, d = idx(v); out[f"group_{k}"] = ("group", i, 0, 250, d)
    if "random8" in families:
        rng = np.random.default_rng(20261121)
        for j in range(20):
            out[f"random8_{j:02d}"] = ("random8", sorted(rng.choice(len(ch), 8, replace=False).tolist()), 0, 250, [])
        rng = np.random.default_rng(20261122)                       # METHOD G1 B7 (1): frontpole has 7 channels -> random-7 reference
        for j in range(20):
            out[f"random7_{j:02d}"] = ("random7", sorted(rng.choice(len(ch), 7, replace=False).tolist()), 0, 250, [])
    if "win100s50" in families:
        for w in range(19):
            a = int(np.floor(w * 12.5 + 0.5)); out[f"win_{50 * w:03d}"] = ("window", allc, a, a + 25, [])
    if "cum" in families:
        for j in range(1, 11):
            out[f"cum_{100 * j:04d}"] = ("cum", allc, 0, 25 * j, [])
    if "devices" in families:
        for k, v in DEVICES.items():
            i, d = idx(v); out[f"device_{k}"] = ("device", i, 0, 250, d)
    return out


TAUS12_T = None


def gscore(S, halves=None):
    """GPU batched scorer. S (..., 200, 200) float64 cosine (query x candidate, truth on the diagonal). Returns dict of tensors (...):
    soft12 (= run.soft_bits: log2 N - LOO-temperature CE, 12-point grid), ls (40-point grid), top1, two_way, mrr; halves -> per-half means
    of the per-query bits (temperature chosen leave-one-query-out over ALL 200 queries)."""
    n = S.shape[-1]; d = torch.diagonal(S, dim1=-2, dim2=-1); rank = (S > d[..., None]).sum(-1) + 1
    out = dict(top1=(rank == 1).double().mean(-1), two_way=((S < d[..., None]).sum(-1).double() / (n - 1)).mean(-1), mrr=(1.0 / rank.double()).mean(-1))
    for key, taus in (("soft12", R._TAUS), ("ls", TAUS40)):
        t = torch.as_tensor(taus, dtype=torch.float64, device=S.device)
        rb = []
        for c in range(0, len(t), 8):                                   # tau chunks keep memory bounded
            lp = torch.log_softmax(S[..., None, :, :] / t[c:c + 8, None, None], -1)
            rb.append(-torch.diagonal(lp, dim1=-2, dim2=-1) / np.log(2))          # (..., T, n) per-query CE in bits
        CE = torch.cat(rb, -2); loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1)
        bits = np.log2(n) - CE.gather(-2, loo.argmin(-2, keepdim=True)).squeeze(-2)   # (..., n)
        out[key] = bits.mean(-1)
        if halves:
            for h, idx in halves.items():
                out[f"{h}|{key}"] = bits[..., torch.as_tensor(idx, device=S.device)].mean(-1)
    return out


def gcos(P, Y):
    """P (..., m, d), Y (m, d) GPU float64 -> cosine (..., m, m)."""
    return (P / P.norm(dim=-1, keepdim=True)) @ (Y / Y.norm(dim=-1, keepdim=True)).T


def _mrr(P, T):
    """P, T: (..., m, d) GPU; true candidate of row i = row i of T. Returns MRR per leading index."""
    P = P / P.norm(dim=-1, keepdim=True); S = P @ (T / T.norm(dim=-1, keepdim=True)).transpose(-1, -2)
    d = torch.diagonal(S, dim1=-2, dim2=-1); return (1.0 / ((S > d[..., None]).sum(-1) + 1).double()).mean(-1)


def pca_batch(Ftr, Ftes, Lmax):
    """Batched run.pca_scores: Ftr (B, n, D) float32 GPU; standardise with training stats (ddof 0, + 1e-6); PCA basis = top eigenvectors of
    the float64 D x D Gram (= right singular vectors of the standardised training matrix, as run.pca_scores); returns scores."""
    mu = Ftr.mean(1, keepdim=True); sd = Ftr.std(1, keepdim=True, unbiased=False) + 1e-6; Z = (Ftr - mu) / sd
    C = torch.stack([(z.double().T @ z.double()) for z in Z])
    _, V = torch.linalg.eigh(C); P = V.flip(-1)[..., :Lmax].float(); del C, V
    return Z @ P, [((X - mu) / sd) @ P for X in Ftes]


def ridge_eig(A, Yc, alphas):
    """A (B, m, L) float64, Yc (m, 1024) centred targets: W_a = (A'A + a I)^-1 A'Yc for every alpha via one batched eigh."""
    e, V = torch.linalg.eigh(A.transpose(1, 2) @ A); VC = V.transpose(1, 2) @ (A.transpose(1, 2) @ Yc)
    return [V @ (VC / (e + a)[..., None]) for a in alphas]


def fit_batch(Ztr, Zte_list, Yg, Ls, perms):
    """Ztr (B, n, Lmax) float32 GPU. For each L (run.py order) pick alpha per mask by run.pick_alpha's rule (90/10 split = perms[L],
    max MRR, first maximum), then fit on all training rows; returns {L: (alpha[B], [pred (B, m, 1024) numpy per test set])}."""
    out = {}; ALPHAS = R.ALPHAS
    for L in Ls:
        perm = perms[L]; cut = int(0.9 * len(perm)); tr = torch.as_tensor(perm[:cut], device=DEV); va = torch.as_tensor(perm[cut:], device=DEV)
        A = Ztr[:, tr, :L].double(); ym = Yg[tr].mean(0)
        Ws = ridge_eig(A, Yg[tr] - ym, ALPHAS); Av = Ztr[:, va, :L].double()
        M = torch.stack([_mrr(Av @ W + ym, Yg[va]) for W in Ws], 1)          # (B, n_alpha)
        best = [int(np.argmax(np.round(m, 12))) for m in M.cpu().numpy()]      # first maximum, as max(ALPHAS, key=...)
        A = Ztr[:, :, :L].double(); ym = Yg.mean(0); e, V = torch.linalg.eigh(A.transpose(1, 2) @ A); VC = V.transpose(1, 2) @ (A.transpose(1, 2) @ (Yg - ym))
        al = torch.tensor([ALPHAS[i] for i in best], dtype=torch.float64, device=DEV)
        W = V @ (VC / (e + al[:, None])[..., None])
        out[L] = ([ALPHAS[i] for i in best], [(Zt[:, :, :L].double() @ W + ym) for Zt in Zte_list])
    return out


def load_gpu(sub):
    """THINGS-EEG2 preprocessed arrays (run.load_raw layout) moved to the GPU in chunks; repetition means computed on the GPU in float32
    (same values as run.load_raw's float32 cast + numpy mean up to float32 rounding)."""
    d = R.RAW / sub
    def ld(p):
        o = np.load(p, allow_pickle=True); return o.item() if isinstance(o, np.ndarray) and o.dtype == object else o
    tr = ld(d / "preprocessed_eeg_training.npy"); ch = list(tr.get("ch_names", [])); X = tr["preprocessed_eeg_data"]; assert X.shape[-1] == 250 and len(ch) == 63
    Xtr = torch.cat([torch.as_tensor(np.ascontiguousarray(X[i:i + 2000])).to(DEV).float().mean(1) for i in range(0, len(X), 2000)]); del tr, X
    X = ld(d / "preprocessed_eeg_test.npy")["preprocessed_eeg_data"]; Xt = torch.as_tensor(np.ascontiguousarray(X)).to(DEV).float(); del X
    out = (Xtr, Xt.mean(1), Xt[:, :40].mean(1), Xt[:, 40:].mean(1), ch); del Xt; torch.cuda.empty_cache(); return out


def subject(s, out, families, smoke):
    t0 = time.time(); sub = f"sub-{s:02d}"
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
    Yg = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV); Yte_g = torch.as_tensor(Yte, dtype=torch.float64, device=DEV)
    Xtr, Xk80, XA, XB, ch = load_gpu(sub)
    names = [p.name.split("_", 1)[1] for p in sorted(TEST_DIR.iterdir())]; assert len(names) == 200
    anim = np.array([n in ANIMATE for n in names]); assert anim.sum() == len(ANIMATE)
    perm7 = np.random.default_rng(7).permutation(200)
    halves = dict(randA=np.sort(perm7[:100]), randB=np.sort(perm7[100:]), animate=np.where(anim)[0], inanimate=np.where(~anim)[0])
    M = masks(ch, families); n = len(Ytr)
    res = dict(target="image", participant=sub, unit="bits per query (200-way InfoNCE log-score)",
               split="THINGS-EEG2 official (16,540 train x 4 reps / 200 test x 80 reps); no test data used for any choice",
               seed=0, selection="alpha: run.pick_alpha rule on training rows (rng 0 through the L sequence)",
               bottleneck="PCA-L of the masked input", N=200, masks={}, rows={})
    print(sub, "loaded", f"{time.time() - t0:.0f}s", flush=True)
    # batches = masks with the same input dimension D (one batched PCA / ridge per batch); memory cap ~4e9 floats of training input
    byD = {}
    for name, (fam, chs, a, b, dropped) in M.items():
        byD.setdefault(len(chs) * (b - a), []).append(name)
    zs = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-6)
    for D, group in sorted(byD.items()):
        Lmax = min(1024, D); Ls = sorted(set([L for L in LS_BASE if L <= Lmax] + [Lmax]))
        rng = np.random.default_rng(0); perms = {L: rng.permutation(n) for L in Ls}         # run.py: one rng(0) through the L sequence
        cap = 1 if D > 8000 else max(1, int(3e8 // (n * D)))         # memory cap ~8 GB peak (cluster is shared); big-D masks one at a time
        for i0 in range(0, len(group), cap):
            nb = group[i0:i0 + cap]
            sel = lambda X: torch.stack([X[:, M[k][1], M[k][2]:M[k][3]].reshape(len(X), -1) for k in nb])
            Ftr = sel(Xtr); Fte = [sel(Xk80), sel(XA), sel(XB)]
            Ztr, Zte = pca_batch(Ftr, Fte, Lmax)
            fits = fit_batch(Ztr, Zte, Yg, Ls, perms); del Ztr, Zte
            al_top = fits[Lmax][0]
            fams = [M[k][0] for k in nb]; nseeds = 0 if all(f.startswith("random") for f in fams) else (1 if all(f in ("window", "cum") for f in fams) else 3)
            # ---- all scoring batched on the GPU (PLAN 10 rule 10): masks x widths x conditions in few tensor ops
            E = ("soft12", "ls", "top1", "two_way", "mrr"); tc = lambda d: {k: v.cpu().numpy() for k, v in d.items()}
            Sc80 = gcos(torch.stack([fits[L][1][0] for L in Ls], 1), Yte_g)                       # (B, nL, 200, 200)
            cur = tc(gscore(Sc80[:, :-1])) if len(Ls) > 1 else None; top = tc(gscore(Sc80[:, -1], halves))
            kAB = tc(gscore(gcos(torch.stack([fits[Lmax][1][1], fits[Lmax][1][2]], 1), Yte_g)))  # (B, 2)
            shp = torch.stack([torch.as_tensor(np.random.default_rng(200 + sd).permutation(200), device=DEV) for sd in range(3)])
            sh = tc(gscore(torch.stack([Sc80[:, -1][:, :, q] for q in shp], 1)))            # (B, 3) shuffled pairs: candidates permuted
            noise_t = []
            for sd in range(nseeds):
                g = torch.Generator(device=DEV); g.manual_seed(100 + sd)
                Ntr = torch.randn(Ftr.shape, generator=g, device=DEV); Nte = torch.randn((len(nb), 200, D), generator=g, device=DEV)
                Zn, (Znt,) = pca_batch(Ntr, [Nte], Lmax); del Ntr, Nte
                A = Zn.double(); ym = Yg.mean(0); e, V = torch.linalg.eigh(A.transpose(1, 2) @ A); VC = V.transpose(1, 2) @ (A.transpose(1, 2) @ (Yg - ym))
                al = torch.tensor(al_top, dtype=torch.float64, device=DEV); noise_t.append(gcos(Znt.double() @ (V @ (VC / (e + al[:, None])[..., None])) + ym, Yte_g))
                del Zn, Znt, A, V, VC
            nz = tc(gscore(torch.stack(noise_t, 1))) if noise_t else None                         # (B, nseeds)
            fa, fb = Fte[1].double(), Fte[2].double()
            fa = (fa - fa.mean(1, keepdim=True)) / (fa.std(1, keepdim=True, unbiased=False) + 1e-6); fb = (fb - fb.mean(1, keepdim=True)) / (fb.std(1, keepdim=True, unbiased=False) + 1e-6)
            Scl = (fa / fa.norm(dim=-1, keepdim=True)) @ (fb / fb.norm(dim=-1, keepdim=True)).transpose(1, 2)    # (B, 200, 200)
            cl = tc(gscore(Scl)); cperm = [np.random.default_rng(300 + sd).permutation(200) for sd in range(3)]
            cls = tc(gscore(torch.stack([Scl[:, :, torch.as_tensor(q, device=DEV)] for q in cperm], 1)))       # (B, 3)
            for j, name in enumerate(nb):
                fam, chs, a, b, dropped = M[name]
                curve = {L: {k: float(cur[k][j, i]) for k in E} for i, L in enumerate(Ls[:-1])} if cur else {}
                curve[Lmax] = {k: float(top[k][j]) for k in E}
                curve[Lmax]["halves"] = {h: {k: float(top[f"{h}|{k}"][j]) for k in ("soft12", "ls")} for h in halves}
                kA, kB = ({k: float(kAB[k][j, i]) for k in E} for i in (0, 1))
                shuf = [{k: float(sh[k][j, i]) for k in ("soft12", "ls")} for i in range(3)]
                noise = [] if fam.startswith("random") or nz is None else [{k: float(nz[k][j, i]) for k in ("soft12", "ls")} for i in range(nseeds)]
                ceil = dict(soft12=float(cl["soft12"][j]), ls=float(cl["ls"][j]), top1=float(cl["top1"][j]), two_way=float(cl["two_way"][j]),
                            shuffled={k: float(cls[k][j].mean()) for k in ("soft12", "ls")})
                def l90(key):
                    v = {L: (curve[L][key] - (0.5 if key == "two_way" else 0)) for L in Ls}
                    return next((L for L in Ls if v[Lmax] > 0 and v[L] >= 0.9 * v[Lmax]), None)
                fl = {e: max(float(np.mean([x[e] for x in noise])) if noise else -np.inf, float(np.mean([x[e] for x in shuf]))) for e in ("soft12", "ls")}
                k40 = {e: (kA[e] + kB[e]) / 2 for e in ("soft12", "ls")}
                res["masks"][name] = dict(family=fam, channels=[ch[i] for i in chs], dropped=dropped, ms=[4 * a, 4 * b], D=D, Lmax=Lmax,
                                          label=DEVICE_LABELS.get(name.replace("device_", ""), name))
                res["rows"][name] = dict(alpha=fits[Lmax][0][j], k80=curve[Lmax], k40=dict(A=kA, B=kB, **k40), ceiling_k40=ceil,
                                         fraction_k40={e: (k40[e] / ceil[e] if ceil[e] > 0 else None) for e in ("soft12", "ls")},
                                         noise=noise, shuffled=shuf, floor=fl, L90_two_way=l90("two_way"), L90_soft12=l90("soft12"), L90_ls=l90("ls"),
                                         curve={str(L): dict(alpha=fits[L][0][j], **{k: curve[L][k] for k in E}) for L in Ls})
                print(sub, name, f"D={D} soft12={curve[Lmax]['soft12']:.3f} ls={curve[Lmax]['ls']:.3f} top1={curve[Lmax]['top1']:.3f} k40={k40['soft12']:.3f} ceil={ceil['soft12']:.3f} floor={fl['soft12']:.3f}",
                      f"{time.time() - t0:.0f}s", flush=True)
            del Ftr, Fte, fits; torch.cuda.empty_cache()
    if "all" in res["rows"]:
        old = json.loads((ROOT / f"results/eegbits/raw_{sub}.json").read_text())
        o = next(r for r in old["rows"] if r["L"] == 1024 and r["k"] == 80 and r["condition"] == "eeg"); n_ = res["rows"]["all"]
        new = dict(alpha=n_["curve"]["1024"]["alpha"], top1=n_["k80"]["top1"], two_way=n_["k80"]["two_way"], mrr=n_["k80"]["mrr"], soft12=n_["k80"]["soft12"])
        res["g2"] = dict(file=f"results/eegbits/raw_{sub}.json", old={k: o.get(k) for k in ("alpha", "top1", "two_way", "mrr", "soft_bits")}, new=new,
                         passed=bool(abs(o["top1"] - new["top1"]) < 1e-9 and abs(o["two_way"] - new["two_way"]) < 1e-3 and abs(o["mrr"] - new["mrr"]) < 1e-3))
        print(sub, "G2", res["g2"], flush=True)
    fl = [max([x[e] for x in r["noise"] + r["shuffled"] for e in ("soft12", "ls")]) for r in res["rows"].values()]
    res["g3"] = dict(max_floor=float(max(fl)), passed=bool(max(fl) <= 0.02))
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / f"d1_{sub}.json").write_text(json.dumps(res, indent=1)); print(sub, "G3", res["g3"], "done", res["runtime_s"], flush=True)


# ------------------------------------------------------------------ group statistics
def boot_ci(v, B=10000, seed=1):
    v = np.asarray(v, float); ix = np.random.default_rng(seed).integers(0, len(v), (B, len(v))); return np.quantile(v[ix].mean(1), [.025, .975]).tolist()


def signflip_p(d):
    """Exact one-sided sign-flip p for mean(d) > 0 (2^n flips)."""
    d = np.asarray(d, float); n = len(d); F = np.array([[1 if (i >> j) & 1 == 0 else -1 for j in range(n)] for i in range(2 ** n)])
    return float(((F * d).mean(1) >= d.mean() - 1e-12).mean())


def holm(p):
    p = np.asarray(p); o = np.argsort(p); h = np.empty_like(p); m = 0.0
    for r, i in enumerate(o):
        m = max(m, min(1.0, (len(p) - r) * p[i])); h[i] = m
    return h.tolist()


def summarize(out):
    from scipy import stats
    per = [json.loads(f.read_text()) for f in sorted(out.glob("d1_sub-*.json"))]; n = len(per); subs = [p["participant"] for p in per]
    names = list(per[0]["rows"]); groups = [k for k in names if k.startswith("group_")]; wins = [k for k in names if k.startswith("win_")]
    cums = [k for k in names if k.startswith("cum_")]; r8 = [k for k in names if k.startswith("random8_")]; r7 = [k for k in names if k.startswith("random7_")]
    devs = [k for k in names if k.startswith("device_")]; pts = groups + wins + cums
    res = dict(experiment="D1 ridge curves (B7)", n=n, participants=subs, primary_estimator="soft12 (run.soft_bits, 12-point LOO temperature; StdImg* variant)",
               sensitivity_estimator="ls (40-point LOO temperature)", fraction_note="group-level fraction = ratio of participant means (k40 yield / same-mask k40 split-half ceiling)",
               masks={k: per[0]["masks"][k] for k in names if not k.startswith("random")},
               g2_all_pass=bool(all(p.get("g2", {}).get("passed", False) for p in per)), g3_max_floor=float(max(p["g3"]["max_floor"] for p in per)))
    for e in ("soft12", "ls"):
        Y = lambda name, f=lambda r: r["k80"][e]: np.array([f(p["rows"][name]) for p in per])
        R_ = {}
        table = {k: dict(yield_k80=float(Y(k).mean()), ci=boot_ci(Y(k)), floor=float(Y(k, lambda r: r["floor"][e]).mean()),
                         yield_k40=float(Y(k, lambda r: r["k40"][e]).mean()), ceiling_k40=float(Y(k, lambda r: r["ceiling_k40"][e]).mean()),
                         fraction_k40_ratio_of_means=float(Y(k, lambda r: r["k40"][e]).mean() / Y(k, lambda r: r["ceiling_k40"][e]).mean()),
                         L90_two_way_median=float(np.median([p["rows"][k]["L90_two_way"] or np.nan for p in per])),
                         top1=float(Y(k, lambda r: r["k80"]["top1"]).mean()), two_way=float(Y(k, lambda r: r["k80"]["two_way"]).mean()))
                 for k in names if not k.startswith("random")}
        ref8 = np.mean([Y(k) for k in r8], 0) if r8 else None; ref7 = np.mean([Y(k) for k in r7], 0) if r7 else None
        table["random8_mean"] = dict(yield_k80=float(ref8.mean()), ci=boot_ci(ref8)) if r8 else None
        table["random7_mean"] = dict(yield_k80=float(ref7.mean()), ci=boot_ci(ref7)) if r7 else None
        tests = []
        for k in groups:
            ref, lab = (ref7, "random7 x20") if per[0]["masks"][k]["channels"].__len__() == 7 else (ref8, "random8 x20")
            if ref is None: continue
            d = Y(k) - ref; tests.append(dict(point=k, contrast=f"group - mean({lab})", d_mean=float(d.mean()), ci=boot_ci(d), p=signflip_p(d), n_pos=int((d > 0).sum())))
        for k in wins + cums:
            d = Y(k) - Y(k, lambda r: r["floor"][e]); tests.append(dict(point=k, contrast="yield - floor", d_mean=float(d.mean()), ci=boot_ci(d), p=signflip_p(d), n_pos=int((d > 0).sum())))
        for t, h in zip(tests, holm([t["p"] for t in tests]) if tests else []):
            t["p_holm"] = h
        R_.update(table=table, tests=tests, n_tests_holm=len(tests))
        if wins:
            W = np.array([[p["rows"][k]["k80"][e] for k in wins] for p in per]); starts = np.array([int(k[4:]) for k in wins])
            pk = starts[W.argmax(1)]; inside = (pk >= 100) & (pk + 100 <= 300)
            R_["P1c_peak"] = dict(peak_start_ms=pk.tolist(), n_inside_100_300=int(inside.sum()), passed=bool(inside.sum() >= 8))
        if pts and "halves" in per[0]["rows"][pts[0]]["k80"]:
            H = lambda h: np.array([[p["rows"][k]["k80"]["halves"][h][e] for k in pts] for p in per])
            for a_, b_, role in (("randA", "randB", "primary"), ("animate", "inanimate", "secondary")):
                rho = np.array([stats.spearmanr(x, y)[0] for x, y in zip(H(a_), H(b_))])
                R_[f"D1.3_{a_}_vs_{b_}"] = dict(role=role, rho=rho.tolist(), median=float(np.median(rho)), mean=float(rho.mean()), ci_mean=boot_ci(rho),
                                                passed=bool(np.median(rho) >= 0.8 and boot_ci(rho)[0] > 0.5))
        if pts and "all" in names and n >= 3:
            C = np.array([[p["rows"][k]["k80"][e] for k in pts] for p in per]) / Y("all")[:, None]
            mae_loo, mae_flat = [], []
            for i in range(n):
                o = np.delete(C, i, 0); loo = o.mean(0); flat = np.full(len(pts), o.mean())
                mae_loo.append(float(np.abs(C[i] - loo).mean())); mae_flat.append(float(np.abs(C[i] - flat).mean()))
            d = np.array(mae_flat) - np.array(mae_loo); kk = int((d > 0).sum())
            R_["D1.4_loo_vs_flat"] = dict(mae_loo=mae_loo, mae_flat=mae_flat, d_mean=float(d.mean()), ci=boot_ci(d), n_better=kk,
                                          sign_p=float(stats.binomtest(kk, n, 0.5, alternative="greater").pvalue), passed=bool(kk >= 8 and boot_ci(d)[0] > 0))
        if devs and "all" in names:
            dt = []
            fr = lambda k: Y(k, lambda r: r["k40"][e]).mean() / Y(k, lambda r: r["ceiling_k40"][e]).mean()
            for k in devs:
                dy = Y(k) - Y("all")
                dt.append(dict(device=k, label=per[0]["masks"][k]["label"], d_yield=float(dy.mean()), ci=boot_ci(dy), p_two_sided=min(1.0, 2 * min(signflip_p(dy), signflip_p(-dy))),
                               fraction_ratio_of_means=float(fr(k)), fraction_all=float(fr("all")), d_fraction=float(fr(k) - fr("all")),
                               within_20pct=bool(abs(fr(k) - fr("all")) <= 0.2 * abs(fr("all")))))
            for t, h in zip(dt, holm([t["p_two_sided"] for t in dt])):
                t["p_holm"] = h
            m = {t["device"]: table[t["device"]]["yield_k80"] for t in dt}
            R_["D5.2_devices"] = dict(tests=dt, monotone=bool(table["all"]["yield_k80"] >= m.get("device_tentwenty32", -1) >= m.get("device_emotiv14", -1) >= m.get("device_cyton8", -1)))
        res[e] = R_
    (out / "summary_d1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res["soft12"].items() if k != "table"}, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="all,groups,random8,win100s50,cum,devices"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        return summarize(out)
    fams = set(a.families.split(","))
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
        if not (out / f"d1_sub-{s:02d}.json").exists():
            subject(s, out, fams, a.smoke)


if __name__ == "__main__":
    main()
