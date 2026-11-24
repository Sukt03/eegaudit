"""RUNNER FIX 1 (round_20261123_auto, ideation Idea 1): is the UBP 'untrained-encoder share' an encoder effect or linear readability of the EEG? THINGS-EEG2, UBP release data (17 posterior channels, whitened, 250 samples), RN50 fovea-blur 'medium' features.
Arms, ALL scored by the same scorer (soft12 LOO-temperature InfoNCE, 200-way, ubp_card.soft_bits) at k80 and k40 (mean of two session halves):
  native            released UBP checkpoint, its own output space (no ridge)                     GATE: == saved released k80_bits (1e-4)
  trained_ridge     released checkpoint, frozen, 1024-d output -> the SAME refitted ridge to the RN50 train features
  untrained_ridge   torch-default-init EEGProjectLayer (seed s), frozen -> same ridge            GATE (seed 0): == saved random_init_ridge k80_bits (1e-4)
  raw_ridge         no encoder: flattened 17x250 EEG (4250-d) -> same ridge
Ridge = ubp_card.py random-init twin recipe verbatim (SVD ridge, 25-point penalty grid on a held-out tenth of training images by mean cosine, refit on all).
Nulls per arm: noise input N(0,1) through the same frozen map + ridge (generator seed 7, as ubp_card); shuffled-label ridge (training image features permuted
across training images before the fit, seed 11); shuffled candidates (ubp_card floor). Positive control (planted): training/test input = A y + noise with a fixed random
A (4250 x 1024), SNR 1 -> raw_ridge must be far above every EEG arm (the ridge + scorer can read a linear code).
All numerics in torch on the GPU. Usage: --out DIR [--subs 1,2] [--seeds 0,1,2]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load
ap = argparse.ArgumentParser(); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--out', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = torch.device('cuda'); t0 = time.time()
ROOT = '/path/to/eegaudit'; D = f'{ROOT}/data_raw/third_party/ubp'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
TAUS = torch.logspace(-2.5, 0, 12, device=dev); LOG2 = math.log(2)
P = lambda *s: print(f'[{time.time()-t0:7.1f}s]', *s, flush=True)

def soft_bits(S):                                           # = ubp_card.soft_bits (12-point grid)
    n = S.shape[0]; r = torch.arange(n, device=dev); Z = S[None] / TAUS[:, None, None]; lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, r, r] / LOG2; loo = (CE.sum(1, keepdim=True) - CE) / (n - 1); return math.log2(n) - CE[loo.argmin(0), r]
def bits(zq, zc, shuffle=False):
    S = F.normalize(zq.float(), dim=1) @ F.normalize(zc.float(), dim=1).T
    if shuffle: g = torch.Generator(device=dev); g.manual_seed(0); S = S[:, torch.randperm(len(S), device=dev, generator=g)]
    return float(soft_bits(S).mean())

class Res(torch.nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x): return x + self.f(x)
class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024), Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))), torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.model(x.reshape(x.shape[0], -1))

FT = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt')['img_features']
FTR = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_train.pt')['img_features']

def load_test(s):                                          # = ubp_card.load_test
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt'); E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL]))
    ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0]); assert imgs == sorted(imgs)
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA; assert all((np.isin(ses[i], [0, 1]) == hA.numpy()).all() for i in range(len(ses)))
    Y = torch.stack([FT['medium'][p] for p in imgs]).float(); av = lambda Z: Z.mean(1).float()[..., :250].to(dev)
    return av(E), av(E[:, hA]), av(E[:, hB]), Y.to(dev), imgs
def load_train(s):                                         # = ubp_card.load_train
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/train.pt'); X = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])).mean(1)[..., :250].half()
    return X, list(np.asarray(t['img'])[:, 0])

def ridge(H, Ytr):
    """ubp_card.py random-init twin ridge, verbatim (incl. its use of the train-part mean mu for centring the refit). Returns map h -> y, (dev cos, lambda)."""
    perm = torch.randperm(len(H), generator=torch.Generator().manual_seed(0)).to(dev); nv = len(H) // 10; va, tr = perm[:nv], perm[nv:]
    mu = H[tr].mean(0); Hc = H - mu; ym = Ytr[tr].mean(0); U, Sv, Vh = torch.linalg.svd(Hc[tr].double(), full_matrices=False); UtY = U.T @ (Ytr[tr] - ym).double(); best = (-1e9, None)
    for lam in torch.logspace(-2, 6, 25, dtype=torch.float64):
        W = Vh.T @ ((Sv / (Sv ** 2 + lam))[:, None] * UtY); sc = float(F.cosine_similarity(Hc[va].double() @ W + ym.double(), Ytr[va].double(), dim=1).mean())
        if sc > best[0]: best = (sc, float(lam))
    U, Sv, Vh = torch.linalg.svd(Hc.double(), full_matrices=False); W = (Vh.T @ ((Sv / (Sv ** 2 + best[1]))[:, None] * (U.T @ (Ytr - Ytr.mean(0)).double()))).float(); b0 = Ytr.mean(0)
    del U, Vh, Hc; return (lambda h: (h - mu) @ W + b0), best

def ridge_ident(H, Ytr):
    """SENSITIVITY read-out (same for every arm): targets L2-normalised, penalty chosen by held-out IDENTIFICATION bits (soft12, first 500 held-out
    training images as queries and candidates) instead of the mean cosine to uncentred features; centring by the mean of the rows fitted."""
    Yn = F.normalize(Ytr, dim=1); perm = torch.randperm(len(H), generator=torch.Generator().manual_seed(0)).to(dev); nv = len(H) // 10; va, tr = perm[:nv][:500], perm[nv:]
    mu = H[tr].mean(0); ym = Yn[tr].mean(0); U, Sv, Vh = torch.linalg.svd((H[tr] - mu).double(), full_matrices=False); UtY = U.T @ (Yn[tr] - ym).double(); best = (-1e9, None)
    for lam in torch.logspace(-2, 6, 25, dtype=torch.float64):
        W = Vh.T @ ((Sv / (Sv ** 2 + lam))[:, None] * UtY); sc = bits(((H[va] - mu).double() @ W).float() + ym, Yn[va])
        if sc > best[0]: best = (sc, float(lam))
    mu = H.mean(0); b0 = Yn.mean(0); U, Sv, Vh = torch.linalg.svd((H - mu).double(), full_matrices=False); W = (Vh.T @ ((Sv / (Sv ** 2 + best[1]))[:, None] * (U.T @ (Yn - b0).double()))).float()
    del U, Vh; return (lambda h: (h - mu) @ W + b0), best

def card(fn, T):
    X80, XA, XB, Y = T[:4]
    with torch.no_grad():
        z80, zA, zB = fn(X80), fn(XA), fn(XB); gn = torch.Generator(device=dev); gn.manual_seed(7); zn = fn(torch.randn(X80.shape, device=dev, generator=gn))
    return dict(k80=bits(z80, Y), k40=0.5 * (bits(zA, Y) + bits(zB, Y)), noise_k80=bits(zn, Y), shuffled_cand_k80=bits(z80, Y, shuffle=True))

def arm(feat, Xtr, Ytr, T, shuffle_null=True):
    """feat: EEG (n,17,250) -> features. Fits the ridge on feat(train), returns the card + shuffled-label-ridge null."""
    with torch.no_grad(): H = torch.cat([feat(Xtr[i:i + 4096].to(dev).float()) for i in range(0, len(Xtr), 4096)])
    pr, best = ridge(H, Ytr); r = card(lambda x: pr(feat(x)), T); r.update(ridge_lambda=best[1], ridge_dev_cos=best[0], dim=int(H.shape[1]))
    if shuffle_null:
        g = torch.Generator().manual_seed(11); ps, _ = ridge(H, Ytr[torch.randperm(len(Ytr), generator=g).to(dev)]); r['shuffled_label_ridge_k80'] = card(lambda x: ps(feat(x)), T)['k80']
    pi, bi = ridge_ident(H, Ytr); ci = card(lambda x: pi(feat(x)), T); r.update(ident_k80=ci['k80'], ident_k40=ci['k40'], ident_noise_k80=ci['noise_k80'], ident_lambda=bi[1], ident_dev_bits=bi[0])
    del H; torch.cuda.empty_cache(); return r

seeds = [int(x) for x in a.seeds.split(',')]
for s in [int(x) for x in a.subs.split(',')]:
    dest = os.path.join(a.out, f'sub-{s:02d}.json')
    if os.path.exists(dest): continue
    T = load_test(s); Xtr, imgs = load_train(s); assert not (set(imgs) & set(T[4])), 'train/test image overlap'
    Ytr = torch.stack([FTR['medium'][p] for p in imgs]).float().to(dev)
    saved = json.load(open(f'{ROOT}/results/baselines/ubp/ubp_card_soft12_sub{s}.json'))['subjects'][str(s)]
    ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt'); enc = Enc().to(dev); enc.load_state_dict({k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
    res = dict(participant=s, code_sha256=CODE_SHA256, arms={})
    nat = card(enc, T); res['arms']['native'] = nat; g1 = abs(nat['k80'] - saved['released']['k80_bits']); g1b = abs(nat['k40'] - saved['released']['k40_bits'])
    res['gate_native'] = dict(this=nat['k80'], saved=saved['released']['k80_bits'], abs_diff=g1, k40_abs_diff=g1b); assert g1 < 1e-4 and g1b < 1e-4, f'GATE native FAIL sub {s}: {res["gate_native"]}'
    res['arms']['trained_ridge'] = arm(enc, Xtr, Ytr, T)
    rows = []
    for sd in seeds:
        torch.manual_seed(sd); rnd = Enc().to(dev).eval(); r = arm(rnd, Xtr, Ytr, T, shuffle_null=(sd == 0)); r['seed'] = sd; rows.append(r)
        if sd == 0:
            g2 = abs(r['k80'] - saved['random_init_ridge']['k80_bits']); res['gate_untrained_seed0'] = dict(this=r['k80'], saved=saved['random_init_ridge']['k80_bits'], abs_diff=g2, lam_this=r['ridge_lambda'], lam_saved=saved['random_init_ridge']['ridge_lambda'])
            assert g2 < 1e-4, f'GATE untrained FAIL sub {s}: {res["gate_untrained_seed0"]}'
    res['arms']['untrained_ridge'] = dict(rows=rows, **{k: float(np.mean([r[k] for r in rows])) for k in ('k80', 'k40', 'noise_k80', 'shuffled_cand_k80', 'ident_k80', 'ident_k40', 'ident_noise_k80')}, shuffled_label_ridge_k80=rows[0]['shuffled_label_ridge_k80'], seed0_k80=rows[0]['k80'], seed0_k40=rows[0]['k40'])
    flat = lambda x: x.reshape(x.shape[0], -1)
    res['arms']['raw_ridge'] = arm(flat, Xtr, Ytr, T)
    # planted positive control: input = A y + unit noise (SNR 1 per column on average), train and test
    g = torch.Generator(device=dev); g.manual_seed(100 + s); A = torch.randn(1024, 4250, device=dev, generator=g) / math.sqrt(1024)
    def plant(Y): Yz = (Y - Ytr.mean(0)) / (Ytr.std(0) + 1e-6); return (Yz @ A + torch.randn(len(Y), 4250, device=dev, generator=g)).reshape(len(Y), 17, 250)
    Xp = plant(Ytr).half().cpu(); Yte = T[3]; Tp = (plant(Yte), plant(Yte), plant(Yte), Yte)
    res['arms']['planted_raw_ridge'] = arm(flat, Xp, Ytr, Tp, shuffle_null=False); del Xp
    res['runtime_s'] = round(time.time() - t0, 1); json.dump(res, open(dest, 'w'), indent=1)
    P(f'sub-{s:02d}', {k: (round(v['k80'], 3), round(v['k40'], 3)) for k, v in res['arms'].items()}, 'nulls', {k: [round(v.get(q, float('nan')), 4) for q in ('noise_k80', 'shuffled_label_ridge_k80', 'shuffled_cand_k80')] for k, v in res['arms'].items()},
      'gates', f'{g1:.1e}', f"{res['gate_untrained_seed0']['abs_diff']:.1e}")
    del enc, Xtr, Ytr, T; torch.cuda.empty_cache()

# ---------------- summary (participant bootstrap, percentile + BCa + t), all on the GPU
R = [json.load(open(os.path.join(a.out, f))) for f in sorted(os.listdir(a.out)) if f.startswith('sub-')]; n = len(R)
def stat(v, B=10000):
    v = torch.tensor(np.asarray(v, np.float64), device=dev); g = torch.Generator(device=dev); g.manual_seed(0); m = v[torch.randint(len(v), (B, len(v)), device=dev, generator=g)].mean(1)
    o = dict(mean=float(v.mean()), ci95=[float(m.quantile(.025)), float(m.quantile(.975))], n=len(v))
    if len(v) > 2:
        from scipy.stats import t as tdist                 # scalar quantile lookup only
        h = float(tdist.ppf(0.975, len(v) - 1)) * float(v.std(unbiased=True)) / math.sqrt(len(v)); o['t95'] = [o['mean'] - h, o['mean'] + h]
        nd = torch.distributions.Normal(0., 1.); z0 = nd.icdf(((m < v.mean()).double().mean()).clamp(1e-6, 1 - 1e-6)); jk = (v.sum() - v) / (len(v) - 1); d = jk.mean() - jk
        acc = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5 + 1e-300); zq = torch.tensor([-1.959964, 1.959964], dtype=torch.float64, device=dev)
        o['bca95'] = [float(x) for x in m.quantile(nd.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq))))]
    return o
S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, n_participants=n, seeds=seeds, unit='bits per query (soft12 LOO InfoNCE, 200-way)', bootstrap='participant, 10000 resamples, GPU torch, seed 0',
         gate_native_max_abs_diff=max(r['gate_native']['abs_diff'] for r in R), gate_untrained_max_abs_diff=max(r['gate_untrained_seed0']['abs_diff'] for r in R), arms={}, contrasts={})
for k in ('k80', 'k40', 'ident_k80', 'ident_k40'):
    for nm in R[0]['arms']:
        if k in R[0]['arms'][nm]: S['arms'].setdefault(nm, {})[k] = stat([r['arms'][nm][k] for r in R])
    g = lambda nm: np.array([r['arms'][nm][k if nm != 'native' else k.replace('ident_', '')] for r in R])
    for x, y in (('untrained_ridge', 'raw_ridge'), ('trained_ridge', 'raw_ridge'), ('trained_ridge', 'untrained_ridge'), ('native', 'trained_ridge')):
        S['contrasts'][f'{x}_minus_{y}|{k}'] = dict(stat(g(x) - g(y)), n_pos=int((g(x) > g(y)).sum()))
    for x, y in (('untrained_ridge', 'native'), ('raw_ridge', 'native'), ('untrained_ridge', 'trained_ridge'), ('raw_ridge', 'trained_ridge'), ('untrained_ridge', 'raw_ridge')):
        S['contrasts'][f'share_{x}_of_{y}|{k}'] = dict(stat(g(x) / g(y)), ratio_of_means=float(g(x).mean() / g(y).mean()))
for nm in R[0]['arms']:
    for q in ('noise_k80', 'shuffled_label_ridge_k80', 'shuffled_cand_k80', 'ident_noise_k80'):
        if q in R[0]['arms'][nm]: S['arms'][nm][q] = stat([r['arms'][nm][q] for r in R])
S['nulls_ok_abs_le_0.02'] = bool(all(abs(S['arms'][nm][q]['mean']) <= 0.02 for nm in S['arms'] if nm != 'planted_raw_ridge' for q in ('noise_k80', 'shuffled_label_ridge_k80', 'shuffled_cand_k80') if q in S['arms'][nm]))
json.dump(S, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1); P(json.dumps(S, indent=1)[:6000])
