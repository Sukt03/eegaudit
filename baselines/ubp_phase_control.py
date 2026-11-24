"""UBP matched-statistics control (, 22 Nov 2026).
Released UBP checkpoints (frozen, no training), soft12 scorer (12-point LOO temperature grid), k=80 and k=40 (split-half sessions) read-out,
identical to experiments/baselines/ubp_card.py (loaders, Enc, soft_bits, card are copied verbatim; ubp_card.py runs on import so it cannot be imported).
Inputs: (G1) real test EEG, unmodified -> must reproduce saved ubp_card_soft12_sub*.json k80/k40 bits to 1e-4;
 (phase) per-trial (each image x repetition), per-channel INDEPENDENT phase randomisation of the 250-sample window (rfft, amplitude kept, uniform phase,
   DC and Nyquist real), then average over repetitions as in the read-out;
 (shuffle) real trials, trial-to-image assignment permuted: for each repetition an independent permutation of the 200 images.
Seeds 0,1,2 for both controls. Expected ~0 bits if yield is stimulus-locked neural content."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load

ap = argparse.ArgumentParser()
ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/ubp_phase_control')
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


def phase_randomise(Ew, g):  # Ew (n, reps, 17, T) float32 on gpu; independent uniform phase per (image, rep, channel)
    Fq = torch.fft.rfft(Ew, dim=-1)
    ph = torch.rand(Fq.shape, device=dev, generator=g) * 2 * math.pi
    ph[..., 0] = 0
    if Ew.shape[-1] % 2 == 0:
        ph[..., -1] = 0
    return torch.fft.irfft(Fq.abs() * torch.exp(1j * ph), n=Ew.shape[-1], dim=-1)


def shuffle_trials(Ew, g):  # for each repetition an independent permutation of images
    out = torch.empty_like(Ew)
    for r in range(Ew.shape[1]):
        out[:, r] = Ew[torch.randperm(Ew.shape[0], device=dev, generator=g), r]
    return out


def readout(enc, X80, XA, XB, Y):
    with torch.no_grad():
        z80, zA, zB = enc(X80), enc(XA), enc(XB)
    b80, t80 = bits_top1(z80, Y); bA, tA = bits_top1(zA, Y); bB, tB = bits_top1(zB, Y)
    return {'k80_bits': b80, 'k80_top1': t80, 'k40_bits': (bA + bB) / 2, 'k40_top1': (tA + tB) / 2}


if not a.summary_only:
    for s in subs:
        ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt')
        enc = Enc().to(dev); enc.load_state_dict({k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
        E, hA, hB, Y = load_trials(s)
        X80, XA, XB = averages(E, hA, hB)
        res = {'sub': s, 'code_sha256': CODE_SHA256, 'NOT_CITABLE': 'until independent review'}
        real = readout(enc, X80, XA, XB, Y); res['real'] = real
        sv = json.load(open(SAVED.format(s)))['subjects'][str(s)]['released']
        res['G1'] = {'saved_k80_bits': sv['k80_bits'], 'saved_k40_bits': sv['k40_bits'], 'abs_diff_k80': abs(real['k80_bits'] - sv['k80_bits']),
                     'abs_diff_k40': abs(real['k40_bits'] - sv['k40_bits'])}
        res['G1']['pass'] = bool(max(res['G1']['abs_diff_k80'], res['G1']['abs_diff_k40']) < 1e-4)
        Ew = E.float()[..., T0:T1].to(dev)
        res['stats'] = {'noise_scale_used_earlier_N01_std': 1.0, 'real_avg80_std': float(X80.std()), 'real_single_trial_std': float(Ew.std()),
                        'real_avg_mean_abs_over_images_std': float(X80.mean(0).abs().mean())}
        res['phase'] = {}; res['shuffle'] = {}
        for sd in seeds:
            g = torch.Generator(device=dev); g.manual_seed(1000 + sd)
            Pw = phase_randomise(Ew, g)
            ratio = float((Pw.abs().pow(2).mean() / Ew.pow(2).mean()))
            r = readout(enc, *[Z.mean(1) for Z in (Pw, Pw[:, hA.to(dev)], Pw[:, hB.to(dev)])], Y); r['avg80_std'] = float(Pw.mean(1).std()); r['power_ratio'] = ratio
            res['phase'][str(sd)] = r; del Pw
            g.manual_seed(2000 + sd)
            Sw = shuffle_trials(Ew, g)
            r2 = readout(enc, *[Z.mean(1) for Z in (Sw, Sw[:, hA.to(dev)], Sw[:, hB.to(dev)])], Y); r2['avg80_std'] = float(Sw.mean(1).std())
            res['shuffle'][str(sd)] = r2; del Sw
        for c in ('phase', 'shuffle'):
            for k in ('k80_bits', 'k40_bits'):
                res[f'{c}_{k}_mean'] = float(np.mean([res[c][str(sd)][k] for sd in seeds]))
        json.dump(res, open(f'{a.out}/sub{s:02d}.json', 'w'), indent=1)
        print(f'sub-{s:02d} G1 {res["G1"]["pass"]} (d80 {res["G1"]["abs_diff_k80"]:.2e}, d40 {res["G1"]["abs_diff_k40"]:.2e}) real k80 {real["k80_bits"]:.4f} k40 {real["k40_bits"]:.4f} | '
              f'phase k80 {res["phase_k80_bits_mean"]:.4f} k40 {res["phase_k40_bits_mean"]:.4f} | shuffle k80 {res["shuffle_k80_bits_mean"]:.4f} k40 {res["shuffle_k40_bits_mean"]:.4f} | '
              f'std avg80 real {res["stats"]["real_avg80_std"]:.3f} single {res["stats"]["real_single_trial_std"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
        del ck, enc, E, Ew; torch.cuda.empty_cache()

# ---- summary: participant bootstrap (10000 resamples, GPU), over the participants present
R = [json.load(open(f'{a.out}/sub{s:02d}.json')) for s in subs]
def boot(v, B=10000):
    v = torch.tensor(v, device=dev, dtype=torch.float64); g = torch.Generator(device=dev); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=dev, generator=g)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}
S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'n_participants': len(R), 'participants': subs, 'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0',
     'G1_all_pass': all(r['G1']['pass'] for r in R), 'G1_max_abs_diff': max(max(r['G1']['abs_diff_k80'], r['G1']['abs_diff_k40']) for r in R), 'bits': {}, 'abs_bits_per_participant': {}}
for k in ('k80', 'k40'):
    S['bits'][f'real_{k}'] = boot([r['real'][f'{k}_bits'] for r in R])
    for c in ('phase', 'shuffle'):
        S['bits'][f'{c}_{k}'] = boot([r[f'{c}_{k}_bits_mean'] for r in R])
        S['abs_bits_per_participant'][f'{c}_{k}'] = [abs(r[f'{c}_{k}_bits_mean']) for r in R]
        S['bits'][f'{c}_{k}_seed_values'] = {sd: [r[c][sd][f'{k}_bits'] for r in R] for sd in R[0][c]}
S['stats'] = {'noise_scale_used_earlier': 'N(0,1) test input (ubp_card.py full_card), std 1.0 on the k80-shaped input',
              'real_avg80_std_mean': float(np.mean([r['stats']['real_avg80_std'] for r in R])), 'real_avg80_std_per_participant': [r['stats']['real_avg80_std'] for r in R],
              'real_single_trial_std_mean': float(np.mean([r['stats']['real_single_trial_std'] for r in R])),
              'phase_avg80_std_mean': float(np.mean([np.mean([r['phase'][sd]['avg80_std'] for sd in r['phase']]) for r in R])),
              'phase_power_ratio_mean': float(np.mean([np.mean([r['phase'][sd]['power_ratio'] for sd in r['phase']]) for r in R]))}
S['per_participant_file_sha256'] = {f'sub{s:02d}': _hl.sha256(open(f'{a.out}/sub{s:02d}.json', 'rb').read()).hexdigest() for s in subs}
json.dump(S, open(f'{a.out}/summary.json', 'w'), indent=1)
print(json.dumps({k: S[k] for k in ('G1_all_pass', 'G1_max_abs_diff', 'bits', 'stats') if k != 'bits'}, indent=1)); print({k: v for k, v in S['bits'].items() if 'seed' not in k})
