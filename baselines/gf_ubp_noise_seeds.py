"""GPU-FILL UBP noise-input seeds (, 22 Nov 2026).
Released UBP checkpoints (frozen, no training), soft12 scorer; loaders/Enc/soft_bits copied verbatim from ubp_phase_control.py (that script runs on import).
Noise-input floor with MORE SEEDS and a SEPARATE k=40 pass (removes the 'single seed reused for k=40' caveat):
  g1   seed 7, N(0,1) of the k80 average shape through the model, soft12 -> must reproduce saved noise_input_floor_bits (ubp_card_soft12_sub*.json) to 1e-4
  n01  N(0,1) drawn directly in the shape of each read-out input (k80: X80 shape; k40: XA and XB shapes, separate draws), seeds 11-15 (same construction as the saved floor)
  avg  per-trial N(0,1) averaged over the repetitions actually averaged (80 for k80, 40 for each k40 half): std 1/sqrt(reps), as real averaging would give
"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load

ap = argparse.ArgumentParser()
ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--seeds', default='11,12,13,14,15')
ap.add_argument('--out', default='/path/to/eegaudit/results/gpu_fill_20261122/ubp_noise_seeds')
ap.add_argument('--summary_only', action='store_true')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda'); t0 = time.time()
D = '/path/to/eegaudit/data_raw/third_party/ubp'
SAVED = '/path/to/eegaudit/results/baselines/ubp/ubp_card_soft12_sub{}.json'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
T0, T1 = 0, 250
TAUS = torch.logspace(-2.5, 0, 12, device=dev)
LOG2 = math.log(2)
subs = [int(s) for s in a.subs.split(',')]; seeds = [int(s) for s in a.seeds.split(',')]


def soft_bits(S, taus=None):  # verbatim from ubp_card.py
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


class Enc(torch.nn.Module):  # verbatim from ubp_card.py
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024),
                                         Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))),
                                         torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))


def load_trials(s):  # as ubp_card.load_test but keeps repetitions: E (200, reps, 17, T) in file dtype
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt')
    E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL]))
    ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0])
    assert imgs == sorted(imgs)
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA
    assert all((np.isin(ses[i], [0, 1]) == hA.numpy()).all() for i in range(len(ses)))
    Y = torch.stack([FT['medium'][p] for p in imgs]).float().to(dev)
    return E, hA, hB, Y


def averages(E, hA, hB):  # E on cpu file dtype or gpu float
    av = lambda Z: Z.mean(1).float()[..., T0:T1].to(dev)
    return av(E), av(E[:, hA]), av(E[:, hB])



def noise_readout(enc, shape80, shapeA, shapeB, Y, g, avg):
    def draw(shape, reps):
        if not avg: return torch.randn(shape, device=dev, generator=g)
        return torch.randn((reps,) + tuple(shape), device=dev, generator=g).mean(0)
    with torch.no_grad():
        z80 = enc(draw(shape80, 80)); zA = enc(draw(shapeA, 40)); zB = enc(draw(shapeB, 40))
    b80, t80 = bits_top1(z80, Y); bA, tA = bits_top1(zA, Y); bB, tB = bits_top1(zB, Y)
    return {'k80_bits': b80, 'k80_top1': t80, 'k40_bits': (bA + bB) / 2, 'k40_top1': (tA + tB) / 2}


if not a.summary_only:
    for s in subs:
        ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt')
        enc = Enc().to(dev); enc.load_state_dict({k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
        E, hA, hB, Y = load_trials(s)
        X80, XA, XB = averages(E, hA, hB)
        res = {'sub': s, 'code_sha256': CODE_SHA256, 'NOT_CITABLE': 'until independent review'}
        sv = json.load(open(SAVED.format(s)))['subjects'][str(s)]['released']
        g = torch.Generator(device=dev); g.manual_seed(7)
        with torch.no_grad(): zn = enc(torch.randn(X80.shape, device=dev, generator=g))
        b, t = bits_top1(zn, Y)
        res['G1'] = {'saved_noise_input_bits': sv['noise_input_floor_bits'], 'rerun_bits': b, 'abs_diff': abs(b - sv['noise_input_floor_bits'])}
        res['G1']['pass'] = bool(res['G1']['abs_diff'] < 1e-4)
        res['n01'] = {}; res['avg'] = {}
        for sd in seeds:
            for nm, avg in (('n01', False), ('avg', True)):
                g = torch.Generator(device=dev); g.manual_seed(5000 + sd + (100 if avg else 0))
                res[nm][str(sd)] = noise_readout(enc, X80.shape, XA.shape, XB.shape, Y, g, avg)
        for c in ('n01', 'avg'):
            for k in ('k80_bits', 'k40_bits'):
                res[f'{c}_{k}_mean'] = float(np.mean([res[c][str(sd)][k] for sd in seeds]))
        json.dump(res, open(f'{a.out}/sub{s:02d}.json', 'w'), indent=1)
        print(f'sub-{s:02d} G1 {res["G1"]["pass"]} (d {res["G1"]["abs_diff"]:.2e}) n01 k80 {res["n01_k80_bits_mean"]:.4f} k40 {res["n01_k40_bits_mean"]:.4f} | avg k80 {res["avg_k80_bits_mean"]:.4f} k40 {res["avg_k40_bits_mean"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
        del ck, enc, E; torch.cuda.empty_cache()

R = [json.load(open(f'{a.out}/sub{s:02d}.json')) for s in subs]
def boot(v, B=10000):
    v = torch.tensor(v, device=dev, dtype=torch.float64); g = torch.Generator(device=dev); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=dev, generator=g)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}
S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'n_participants': len(R), 'participants': subs, 'seeds': seeds,
     'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0', 'G1_all_pass': all(r['G1']['pass'] for r in R), 'G1_max_abs_diff': max(r['G1']['abs_diff'] for r in R), 'bits': {}, 'floor_max_abs_per_participant': {}}
for c in ('n01', 'avg'):
    for k in ('k80', 'k40'):
        S['bits'][f'{c}_{k}'] = boot([r[f'{c}_{k}_bits_mean'] for r in R])
        S['floor_max_abs_per_participant'][f'{c}_{k}'] = max(abs(r[f'{c}_{k}_bits_mean']) for r in R)
        S['bits'][f'{c}_{k}_per_seed'] = {sd: boot([r[c][sd][f'{k}_bits'] for r in R]) for sd in R[0][c]}
S['floors_ok_le_0.02'] = all(v <= 0.02 for v in S['floor_max_abs_per_participant'].values())
S['per_participant_file_sha256'] = {f'sub{s:02d}': _hl.sha256(open(f'{a.out}/sub{s:02d}.json', 'rb').read()).hexdigest() for s in subs}
json.dump(S, open(f'{a.out}/summary.json', 'w'), indent=1)
print(json.dumps({k: S[k] for k in ('G1_all_pass', 'G1_max_abs_diff', 'floors_ok_le_0.02', 'floor_max_abs_per_participant')}, indent=1)); print({k: v for k, v in S['bits'].items() if 'per_seed' not in k})
