"""L90 for UBP (, 22 Nov 2026). No training: released UBP checkpoints frozen, soft12 scorer
(copied verbatim from experiments/baselines/ubp_phase_control.py). Input bottleneck: the test input (17 ch x 250 samples = 4250 dims) is projected onto the
top-L principal components of the participant's TRAINING input (16540 images x 4-rep means, label-free, covariance eigh in float64 on GPU), centred on the
training mean, then fed to the frozen encoder. L90 = smallest L in the grid whose k80 yield >= 0.9 x full-width yield (L=4250, identity).
Reference: random L-dim orthonormal subspace (seed 0) of the same grid. G1: L=4250 reproduces saved ubp_card_soft12 k80 bits to 1e-4."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load
ap = argparse.ArgumentParser()
ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--out', default='/path/to/eegaudit/results/ext/l90_ubp')
ap.add_argument('--summary_only', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda'); t0 = time.time()
D = '/path/to/eegaudit/data_raw/third_party/ubp'
SAVED = '/path/to/eegaudit/results/baselines/ubp/ubp_card_soft12_sub{}.json'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
T0, T1 = 0, 250; DIM = 17 * 250
TAUS = torch.logspace(-2.5, 0, 12, device=dev); LOG2 = math.log(2)
GRID = [1, 2, 5, 10, 25, 50, 100, 150, 250, 300, 350, 400, 500, 750, 1000, 2000, 3000, DIM]
subs = [int(s) for s in a.subs.split(',')]

def soft_bits(S, taus=None):
    taus = TAUS if taus is None else taus
    n = S.shape[0]; r = torch.arange(n, device=dev)
    Z = S[None] / taus[:, None, None]; lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, r, r] / LOG2
    loo = (CE.sum(1, keepdim=True) - CE) / (n - 1)
    return math.log2(n) - CE[loo.argmin(0), r]

def bits_top1(zq, zc):
    S = F.normalize(zq.float(), dim=1) @ F.normalize(zc.float(), dim=1).T
    return float(soft_bits(S).mean()), float((S.argmax(1) == torch.arange(len(S), device=dev)).float().mean())

FT = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt')['img_features']
class Res(torch.nn.Module):
    def __init__(self, f):
        super().__init__(); self.f = f
    def forward(self, x):
        return x + self.f(x)
class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024), Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))), torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))

def load_test(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt')
    E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL]))
    ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0]); assert imgs == sorted(imgs)
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA
    Y = torch.stack([FT['medium'][p] for p in imgs]).float().to(dev)
    return E, hA, hB, Y

def train_pca(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/train.pt')
    X = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])).to(dev).float().mean(1)[..., T0:T1].reshape(-1, DIM).double()
    mu = X.mean(0); Xc = X - mu; C = Xc.T @ Xc / (len(X) - 1); del Xc, X
    w, V = torch.linalg.eigh(C); idx = w.argsort(descending=True)
    return mu, V[:, idx].float(), w[idx]

def proj(X, mu, B):  # X (n,17,250) float; B (DIM, L) orthonormal columns
    x = X.reshape(len(X), -1) - mu.float()
    return ((x @ B) @ B.T + mu.float()).reshape(X.shape)

def readout(enc, X80, XA, XB, Y):
    with torch.no_grad():
        z80, zA, zB = enc(X80), enc(XA), enc(XB)
    return {'k80_bits': bits_top1(z80, Y)[0], 'k40_bits': (bits_top1(zA, Y)[0] + bits_top1(zB, Y)[0]) / 2}

def l90(Ls, ys):
    full = ys[-1]; thr = 0.9 * full
    for L, y in zip(Ls, ys):
        if y >= thr: return L
    return Ls[-1]

if not a.summary_only:
    for s in subs:
        ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt')
        enc = Enc().to(dev); enc.load_state_dict({k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
        E, hA, hB, Y = load_test(s)
        Ef = E.float()[..., T0:T1]
        X80, XA, XB = Ef.mean(1).to(dev), Ef[:, hA].mean(1).to(dev), Ef[:, hB].mean(1).to(dev)
        mu, V, w = train_pca(s)
        g = torch.Generator(device=dev); g.manual_seed(0)
        R = torch.linalg.qr(torch.randn(DIM, max(GRID), device=dev, generator=g))[0]
        res = {'sub': s, 'code_sha256': CODE_SHA256, 'NOT_CITABLE': 'until independent review', 'grid': GRID, 'pca': {}, 'random': {}}
        ev = (w.cumsum(0) / w.sum()).cpu()
        res['pca_var_explained'] = {str(L): float(ev[L - 1]) for L in GRID}
        for name, B_ in (('pca', V), ('random', R)):
            for L in GRID:
                B = B_[:, :L]
                res[name][str(L)] = readout(enc, *[proj(Z, mu, B) for Z in (X80, XA, XB)], Y)
        full = readout(enc, X80, XA, XB, Y); res['real'] = full
        sv = json.load(open(SAVED.format(s)))['subjects'][str(s)]['released']
        d80 = abs(full['k80_bits'] - sv['k80_bits']); d40 = abs(full['k40_bits'] - sv['k40_bits'])
        res['G1'] = {'saved_k80': sv['k80_bits'], 'saved_k40': sv['k40_bits'], 'abs_diff_k80': d80, 'abs_diff_k40': d40, 'pass': bool(max(d80, d40) < 1e-4),
                     'full_width_projection_k80_diff': abs(res['pca'][str(DIM)]['k80_bits'] - full['k80_bits'])}
        for name in ('pca', 'random'):
            for k in ('k80', 'k40'):
                res[f'L90_{name}_{k}'] = l90(GRID, [res[name][str(L)][f'{k}_bits'] for L in GRID])
        json.dump(res, open(f'{a.out}/sub{s:02d}.json', 'w'), indent=1)
        print(f'sub-{s:02d} G1 {res["G1"]["pass"]} (d80 {d80:.2e} d40 {d40:.2e}) real k80 {full["k80_bits"]:.4f} | L90 pca k80 {res["L90_pca_k80"]} k40 {res["L90_pca_k40"]} '
              f'random k80 {res["L90_random_k80"]} | pca k80 by L ' + ' '.join(f'{L}:{res["pca"][str(L)]["k80_bits"]:.2f}' for L in GRID) + f' ({time.time()-t0:.0f}s)', flush=True)
        del ck, enc, E, Ef, V, R; torch.cuda.empty_cache()

Rs = [json.load(open(f'{a.out}/sub{s:02d}.json')) for s in subs]
def boot(v, fn='mean', B=10000):
    v = torch.tensor(v, device=dev, dtype=torch.float64); g = torch.Generator(device=dev); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=dev, generator=g)]
    m = m.mean(1) if fn == 'mean' else m.median(1).values
    return {'est': float(v.mean() if fn == 'mean' else v.median()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}
S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'n_participants': len(Rs), 'participants': subs, 'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0',
     'G1_all_pass': all(r['G1']['pass'] for r in Rs), 'G1_max_abs_diff': max(max(r['G1']['abs_diff_k80'], r['G1']['abs_diff_k40']) for r in Rs),
     'definition': 'L90 = smallest grid L with k yield >= 0.9 x full-width (L=4250) yield; input projected on top-L training PCs (17 ch x 250 samples = 4250 dims)', 'grid': GRID}
for name in ('pca', 'random'):
    for k in ('k80', 'k40'):
        S[f'L90_{name}_{k}_mean'] = boot([r[f'L90_{name}_{k}'] for r in Rs]); S[f'L90_{name}_{k}_median'] = boot([r[f'L90_{name}_{k}'] for r in Rs], 'median')
        S[f'L90_{name}_{k}_per_participant'] = [r[f'L90_{name}_{k}'] for r in Rs]
    S[f'yield_by_L_{name}_k80'] = {str(L): boot([r[name][str(L)]['k80_bits'] for r in Rs]) for L in GRID}
S['real_k80'] = boot([r['real']['k80_bits'] for r in Rs])
S['pca_var_explained_mean'] = {str(L): float(np.mean([r['pca_var_explained'][str(L)] for r in Rs])) for L in GRID}
S['per_participant_file_sha256'] = {f'sub{s:02d}': _hl.sha256(open(f'{a.out}/sub{s:02d}.json', 'rb').read()).hexdigest() for s in subs}
json.dump(S, open(f'{a.out}/summary.json', 'w'), indent=1)
print({k: S[k] for k in S if k.startswith('L90') and 'per_part' not in k}, 'G1', S['G1_all_pass'], S['G1_max_abs_diff'], 'n', S['n_participants'])
