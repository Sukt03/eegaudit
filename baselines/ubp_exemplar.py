"""N-C DE-SATURATED OWN-CEILING FRACTION (METHOD IDEATION ROUND 3, N-C; rung R4). FIXER, 21 Nov 2026.
UBP's 200-way ceiling is near log2 200, so its 200-way fraction is compressed; the within-category (exemplar) ceiling is far from saturation.
UBP (released intra-subject checkpoint, its own RN50 'medium' space, whitened 17 posterior channels, as ubp_card.py; k80 = 80 repetitions,
  k40 = session halves 1-2 / 3-4 as the card):
  exemplar yield at k40 and k80 = s1_averaging.nway with the R2 temperature rule (one leave-one-query-out N-way temperature per call, fit
  on that construction's own scores; matched queries = exact-profile pools with >= N + 1 images); null = the S1 v3 null (20 replicates);
  within-category ceiling at k40 = half A vs half B and B vs A (k-mean vs rest-mean at k = 40 = the S1 construction), same rule.
NICE-GA: the same quantities are the S1 v3 rows (results/ext/s1_v3, nice_ga_k40 / k80; same nway, pools and rule) and are read from there.
Fraction (k40) = (exemplar - floor) / (ceiling_exemplar - floor), ratio of participant means, participant-bootstrap CI (GPU, 10,000).
Falsifier (pre-stated): UBP exemplar fraction CI lower bound > 0.50 -> "the best open decoder reaches most of the recording's exemplar
  reproducibility"; otherwise "it reaches X % of 200-way but Y % of exemplar" (X = the card's floor-corrected 200-way k40 fraction).
G2: the 200-way k40 InfoNCE (card estimator, 40-point grid) reproduces results/baselines/ubp/ubp_card_full_{a,b}.json within 1e-3.
Usage: python -I -u ubp_exemplar.py --subs 1,...,10 --out results/baselines/ubp_exemplar ; --summarize --out results/baselines/ubp_exemplar"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
ROOT = '/path/to/eegaudit'
sys.path.insert(0, f'{ROOT}/experiments/baselines'); sys.path.insert(0, f'{ROOT}/experiments/ext')
from safe_load import safe_load
import s1_averaging as S1

ap = argparse.ArgumentParser(); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--out', default=f'{ROOT}/results/baselines/ubp_exemplar')
ap.add_argument('--summarize', action='store_true'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = 'cuda'; D = f'{ROOT}/data_raw/third_party/ubp'; LOG2 = math.log(2); TAUS = torch.logspace(-2.5, 0, 40, device=dev)
# channel list and selection copied verbatim from ubp_card.py
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
T0, T1 = 0, 250


class Res(torch.nn.Module):
    def __init__(self, f):
        super().__init__(); self.f = f
    def forward(self, x):
        return x + self.f(x)


class Enc(torch.nn.Module):  # = ubp_card.Enc (UBP EEGProjectLayer)
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024), Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))), torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))


def soft_bits(S):  # = ubp_card.soft_bits (40-point grid, LOO temperature), per-row bits
    n = S.shape[0]; r = torch.arange(n, device=dev); Z = S[None] / TAUS[:, None, None]; lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, r, r] / LOG2; loo = (CE.sum(1, keepdim=True) - CE) / (n - 1); return math.log2(n) - CE[loo.argmin(0), r]


def load_test(s):  # = ubp_card.load_test, plus the image list
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt')
    E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])); ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0]); assert imgs == sorted(imgs)
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA
    assert all((np.isin(ses[i], [0, 1]) == hA.numpy()).all() for i in range(len(ses)))
    Y = torch.stack([FT['medium'][p] for p in imgs]).float(); av = lambda Z: Z.mean(1).float()[..., T0:T1].to(dev)
    return av(E), av(E[:, hA]), av(E[:, hB]), Y.to(dev), imgs, E


def subject(s):
    t0 = time.time(); X80, XA, XB, Y, imgs, E = load_test(s)
    meta = np.load(f'{ROOT}/data_raw/things_eeg2_images/images_set/image_metadata.npy', allow_pickle=True).item()
    assert [p.replace('\\', '/').split('/')[-2] for p in imgs] == list(meta['test_img_concepts']), 'UBP test order != THINGS-EEG2 test order'
    ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt'); enc = Enc().to(dev); enc.load_state_dict({k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
    with torch.no_grad():
        z80, zA, zB = [F.normalize(enc(x).float(), dim=1).double() for x in (X80, XA, XB)]
    Yn = F.normalize(Y, dim=1).double()
    g2 = float((soft_bits((zA @ Yn.T).float()).mean() + soft_bits((zB @ Yn.T).float()).mean()) / 2)
    S80 = (z80 @ Yn.T)[None]; S40s = torch.stack([zA @ Yn.T, zB @ Yn.T]); Scs = torch.stack([zA @ zB.T, zB @ zA.T])
    # FIXER (METHOD 18:40 G1 FIX NC): PRIMARY k40 = the S1 construction: the 20 random 40-subsets of the s1_v3 rng stream for the yield and
    # the 20 k-vs-rest splits (40 vs 40, seeds 4000 + 97k + j) for the ceiling; the session split is kept as a labelled sensitivity.
    rng = np.random.default_rng(20261121); subs_ = {k: ([np.arange(80)] if k == 80 else [rng.choice(80, k, replace=False) for _ in range(S1.NSUB)]) for k in S1.KS}
    rest = [(lambda p_: (p_[:40], p_[40:]))(np.random.default_rng(4000 + 97 * 40 + j).permutation(80)) for j in range(S1.NSUB)]
    enc_ = lambda ids: F.normalize(enc(E[:, ids].mean(1).float()[..., T0:T1].to(dev)).float(), dim=1).double()
    with torch.no_grad():
        S40 = torch.stack([enc_(ids) @ Yn.T for ids in subs_[40]]); Sc = torch.stack([enc_(a_) @ enc_(b_).T for a_, b_ in rest])
    gN = torch.Generator(device=dev); gN.manual_seed(1000 + s)
    with torch.no_grad():
        zn = F.normalize(enc(torch.randn((200, 17, T1 - T0), generator=gN, device=dev)).float(), dim=1).double()
    Sn = (zn @ Yn.T)[None]; perms = [torch.as_tensor(np.random.default_rng(300 + j).permutation(200), device=dev) for j in range(20)]
    w200 = dict(yield200=float(S1.info200(S40)[0].mean()), ceiling200=float(S1.info200(Sc)[0].mean()), floor200_shuffled=float(torch.stack([S1.info200(S40[:, :, p_])[0].mean() for p_ in perms]).mean()),
                floor200_noise=float(S1.info200(Sn)[0].mean()), exemplar_noise=float(S1.nway(Sn, None, POOL, N, 50, 7)['matched'].mean()))
    row = dict(participant=f'sub-{s:02d}', coverage=COV, k40_200way_bits=g2, k40_random_split_200way=w200, k40_construction='20 random 40-subsets (s1_v3 stream); ceiling = 20 random 40-vs-rest splits')
    for k, S, tag in ((80, S80, ''), (40, S40, ''), (40, S40s, '_session')):
        ex = S1.nway(S, None, POOL, N, 50, 7 + k); exf = torch.stack([S1.nway(S, None, POOL, N, 5, 900 + 10 * k + r, shuffled=True)['matched'].mean() for r in range(20)])
        row[f'k{k}{tag}'] = dict(exemplar=float(ex['matched'].mean()), exemplar_allq=float(ex['all'].mean()), exemplar_tau=ex['tau'], exemplar_floor=float(exf.mean()), exemplar_floor_p95=float(torch.quantile(exf, .95)))
    ce = S1.nway(Sc, None, POOL, N, 50, 70 + 40); row['k40'].update(ceiling_exemplar=float(ce['matched'].mean()), ceiling_exemplar_allq=float(ce['all'].mean()), ceiling_tau=ce['tau'], ceiling_construction='20 random 40-vs-rest splits (S1 construction)')
    ces = S1.nway(Scs, None, POOL, N, 50, 70 + 40); row['k40_session'].update(ceiling_exemplar=float(ces['matched'].mean()), ceiling_construction='session halves A vs B, both directions (sensitivity)')
    reg = REG.get(str(s)); row['g2'] = dict(card_k40_bits=reg and reg['k40_bits'], ours=g2, abs_diff=None if reg is None else abs(reg['k40_bits'] - g2))
    row['g2']['pass'] = bool(reg is not None and row['g2']['abs_diff'] < 1e-3)
    row['card_200way'] = None if reg is None else {k: reg[k] for k in ('k40_bits', 'k40_floor_bits', 'ceiling_k40_bits', 'fraction_k40')}
    row['runtime_s'] = round(time.time() - t0, 1); json.dump(row, open(f'{a.out}/ubp_sub-{s:02d}.json', 'w'), indent=1)
    print(row['participant'], 'G2', row['g2'], 'k40', {q: round(v, 4) for q, v in row['k40'].items() if isinstance(v, float)}, 'k80 ex', round(row['k80']['exemplar'], 4), f"{row['runtime_s']}s", flush=True)


def summarize():
    g = torch.Generator(device=dev); g.manual_seed(20261121)
    def frac(ex, fl, ce):
        ex, fl, ce = [torch.as_tensor(np.asarray(v, float), device=dev) for v in (ex, fl, ce)]; n = len(ex); ix = torch.randint(0, n, (10000, n), generator=g, device=dev)
        num, den = ex - fl, ce - fl; b = num[ix].mean(1) / den[ix].mean(1); ci = torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=dev)).tolist()
        return dict(ratio_of_means=float(num.mean() / den.mean()), ci=ci, per_participant=(num / den).tolist(), n=n)
    U = [json.load(open(f'{a.out}/{f}')) for f in sorted(os.listdir(a.out)) if f.startswith('ubp_sub-')]
    res = dict(note=__doc__, n_ubp=len(U), g2_all_pass=all(u['g2']['pass'] for u in U))
    k = lambda R, kk, q: [r[kk][q] for r in R]
    res['ubp'] = dict(exemplar_fraction_k40=frac(k(U, 'k40', 'exemplar'), k(U, 'k40', 'exemplar_floor'), k(U, 'k40', 'ceiling_exemplar')),
                      fraction_200way_k40_floor_corrected=frac([u['k40_random_split_200way']['yield200'] for u in U], [max(u['k40_random_split_200way']['floor200_shuffled'], u['k40_random_split_200way']['floor200_noise']) for u in U], [u['k40_random_split_200way']['ceiling200'] for u in U]),
                      fraction_200way_k40_card_session_split_40pt_sensitivity=frac([u['card_200way']['k40_bits'] for u in U], [u['card_200way']['k40_floor_bits'] for u in U], [u['card_200way']['ceiling_k40_bits'] for u in U]),
                      exemplar_noise_floor=float(np.mean([u['k40_random_split_200way']['exemplar_noise'] for u in U])),
                      exemplar_fraction_k40_session_split_sensitivity=frac(k(U, 'k40_session', 'exemplar'), k(U, 'k40_session', 'exemplar_floor'), k(U, 'k40_session', 'ceiling_exemplar')),
                      exemplar_k80=float(np.mean(k(U, 'k80', 'exemplar'))), exemplar_k40=float(np.mean(k(U, 'k40', 'exemplar'))), ceiling_exemplar_k40=float(np.mean(k(U, 'k40', 'ceiling_exemplar'))))
    lo = res['ubp']['exemplar_fraction_k40']['ci'][0]
    res['falsifier'] = dict(rule='UBP exemplar fraction (k40, floor-corrected) CI lower bound > 0.50', lower=lo, triggered=bool(lo > 0.50),
                            reading=('the best open decoder reaches most of the recording\'s exemplar reproducibility' if lo > 0.50 else
                                     f"UBP reaches {100 * res['ubp']['fraction_200way_k40_floor_corrected']['ratio_of_means']:.0f} % of 200-way but {100 * res['ubp']['exemplar_fraction_k40']['ratio_of_means']:.0f} % of exemplar"))
    S1f = f'{ROOT}/results/ext/s1_v5'; P = [json.load(open(f'{S1f}/{f}')) for f in sorted(os.listdir(S1f)) if f.startswith('sub-')] if os.path.isdir(S1f) else []
    if P:
        r40 = [p['rows']['nice_ga_k40'] for p in P]
        res['nice_ga_from_s1_v3'] = dict(exemplar_fraction_k40=frac([r['exemplar'] for r in r40], [r['exemplar_floor'] for r in r40], [r['ceiling_exemplar'] for r in r40]),
                                         exemplar_k80=float(np.mean([p['rows']['nice_ga_k80']['exemplar'] for p in P])), participants=[p['participant'] for p in P])
    json.dump(res, open(f'{a.out}/ubp_exemplar_card.json', 'w'), indent=1); print(json.dumps({x: y for x, y in res.items() if x != 'note'}, indent=1)[:3000])


if __name__ == '__main__':
    if a.summarize: summarize(); sys.exit(0)
    FT = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt')['img_features']
    N, POOL, _, COV = S1.pools_from(S1.categories())
    REG = {}
    for fn in ('ubp_card_full_a.json', 'ubp_card_full_b.json'):
        p = f'{ROOT}/results/baselines/ubp/{fn}'
        if os.path.exists(p): REG.update({str(s): v['released'] for s, v in json.load(open(p))['subjects'].items() if 'released' in v})
    for s in [int(x) for x in a.subs.split(',')]:
        if not os.path.exists(f'{a.out}/ubp_sub-{s:02d}.json'): subject(s)
