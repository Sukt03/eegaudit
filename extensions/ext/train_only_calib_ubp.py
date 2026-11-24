"""Train-only temperature calibration for the UBP released-decoder image yield (blind-review check).
Protocols (soft12 grid = 12-point logspace(-2.5,0), 200 test queries x 200 candidates, bits = log2 200 - CE):
  A  current: leave-one-query-out temperature over the 200 test queries (reproduces ubp_card soft_bits; G1 vs saved soft12 JSON, 1e-4).
  B  per-participant temperature = argmin of mean CE over TRAINING queries (200-concept candidate sets drawn from the 16540 training
     images, one random image per concept, NDRAW draws, same RN50 'medium' image space), applied unchanged to test queries.
  C  single temperature = argmin over the grid of the CE averaged over training queries of all participants, used for everyone.
Models: released decoder ('raw'); untrained-encoder + ridge baseline (rebuilt exactly as ubp_card: seed-0 EEGProjectLayer, ridge on a
held-out tenth). That baseline has NO PCA in ubp_card (no input PCA at all), so there is no PCA to refit; the temperature is its only test adaptation.
CAVEAT: the released decoders and the ridge were fitted on ALL 16540 training images (4 reps averaged), so training queries are in-sample and
no held-out training images/repetitions exist for them; train-query cosines are optimistic, and 4-rep training EEG differs in SNR from k=40/80 test.
Run: srun python -I -u train_only_calib_ubp.py --subs 1,2"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load
ap = argparse.ArgumentParser(); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--ndraw', type=int, default=200)
ap.add_argument('--out', default='/path/to/eegaudit/results/ext/train_only_calib_ubp'); ap.add_argument('--tag', default='')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda'); t0 = time.time(); LOG2 = math.log(2)
D = '/path/to/eegaudit/data_raw/third_party/ubp'; SAVED = '/path/to/eegaudit/results/baselines/ubp'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
TAUS = torch.logspace(-2.5, 0, 12, device=dev)
subs = [int(s) for s in a.subs.split(',')]
FT = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt')['img_features']
FTR = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_train.pt')['img_features']
class Res(torch.nn.Module):
    def __init__(s, f): super().__init__(); s.f = f
    def forward(s, x): return x + s.f(x)
class Enc(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024), Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))), torch.nn.LayerNorm(1024))
        s.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(s, x): return s.model(x.reshape(x.shape[0], -1))
def load_test(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt')
    E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])); ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0])
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA
    Y = torch.stack([FT['medium'][p] for p in imgs]).float()
    av = lambda Z: Z.mean(1).float()[..., 0:250].to(dev)
    return [av(E), av(E[:, hA]), av(E[:, hB])], Y.to(dev)
def load_train(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/train.pt')
    X = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])).mean(1)[..., 0:250].half(); imgs = list(np.asarray(t['img'])[:, 0]); return X, imgs
def ce_all(S, taus=TAUS):
    """S (..., n, n) cosines, truth on diagonal -> per-query CE bits (..., T, n)."""
    n = S.shape[-1]; r = torch.arange(n, device=S.device); out = []
    for t in taus:
        lp = F.log_softmax(S / t, -1); out.append(-lp[..., r, r] / LOG2)
    return torch.stack(out, -2)
def bitsA(CE):  # LOO temperature over the n queries, per-row bits (..., n)
    n = CE.shape[-1]; loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1)
    return math.log2(n) - CE.gather(-2, loo.argmin(-2, keepdim=True)).squeeze(-2)
def cosm(Z, Y): return F.normalize(Z.float(), dim=-1) @ F.normalize(Y.float(), dim=-1).T

# ---- per participant: test cosines (3 k-conditions x {raw, base}) and training-query CE per tau
Stest, trCE, trlam = {}, {}, {}
for s in subs:
    Xt, Y = load_test(s); Xtr, imgs = load_train(s)
    ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt'); enc = Enc().to(dev); enc.load_state_dict({k[6:]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}); enc.eval()
    Ytr = torch.stack([FTR['medium'][p] for p in imgs]).float().to(dev)
    torch.manual_seed(0); rnd = Enc().to(dev).eval()
    with torch.no_grad():
        H = torch.cat([rnd(Xtr[i:i + 4096].to(dev).float()) for i in range(0, len(Xtr), 4096)])
        Zr = torch.cat([enc(Xtr[i:i + 4096].to(dev).float()) for i in range(0, len(Xtr), 4096)])
    perm = torch.randperm(len(H), generator=torch.Generator().manual_seed(0)).to(dev); nv = len(H) // 10; va, tr = perm[:nv], perm[nv:]
    mu = H[tr].mean(0); Hc = H - mu; ym = Ytr[tr].mean(0)
    U, Sv, Vh = torch.linalg.svd(Hc[tr].double(), full_matrices=False); UtY = U.T @ (Ytr[tr] - ym).double(); best = (-1e9, None)
    for lam in torch.logspace(-2, 6, 25, dtype=torch.float64):
        W = Vh.T @ ((Sv / (Sv ** 2 + lam))[:, None] * UtY); sc = float(F.cosine_similarity(Hc[va].double() @ W + ym.double(), Ytr[va].double(), dim=1).mean())
        if sc > best[0]: best = (sc, float(lam))
    U, Sv, Vh = torch.linalg.svd(Hc.double(), full_matrices=False)
    W = (Vh.T @ ((Sv / (Sv ** 2 + best[1]))[:, None] * (U.T @ (Ytr - Ytr.mean(0)).double()))).float(); b0 = Ytr.mean(0)
    probe = lambda X: (rnd(X) - mu) @ W + b0
    with torch.no_grad():
        Stest[s] = torch.stack([torch.stack([cosm(enc(x), Y) for x in Xt]), torch.stack([cosm(probe(x), Y) for x in Xt])])  # (2 model, 3 k, 200, 200)
        Ptr = torch.cat([probe(Xtr[i:i + 4096].to(dev).float()) for i in range(0, len(Xtr), 4096)])
    # training candidate sets: 200 distinct concepts, one random image each, NDRAW draws (seed 0)
    conc = np.array([os.path.dirname(p) for p in imgs]); uc = np.unique(conc); byc = [np.where(conc == c)[0] for c in uc]
    rng = np.random.default_rng(0); idx = np.stack([[rng.choice(byc[j]) for j in rng.choice(len(uc), 200, replace=False)] for _ in range(a.ndraw)])
    idx = torch.from_numpy(idx).to(dev)
    trCE[s] = torch.stack([ce_all(torch.einsum('dnc,dmc->dnm', F.normalize(Zr[idx].float(), dim=-1), F.normalize(Ytr[idx], dim=-1))).mean((0, 2)),
                           ce_all(torch.einsum('dnc,dmc->dnm', F.normalize(Ptr[idx].float(), dim=-1), F.normalize(Ytr[idx], dim=-1))).mean((0, 2))])  # (2 model, T) mean CE
    trlam[s] = best[1]; print(f'sub-{s:02d} done ({time.time()-t0:.0f}s) train tauB raw {float(TAUS[trCE[s][0].argmin()]):.4f} base {float(TAUS[trCE[s][1].argmin()]):.4f}', flush=True)
    del H, Hc, U, Vh, Zr, Ptr; torch.cuda.empty_cache()

# ---- one batched scoring call: (P, model, k, 200, 200) x T
Sall = torch.stack([Stest[s] for s in subs]); P = len(subs)
CE = ce_all(Sall)                                               # (P, 2, 3, T, 200)
bitsApro = bitsA(CE).mean(-1)                                   # (P, 2, 3)  protocol A
CEm = CE.mean(-1)                                               # (P, 2, 3, T) mean test CE per tau
trstack = torch.stack([trCE[s] for s in subs])                  # (P, 2, T)
tB = trstack.argmin(-1)                                         # (P, 2)
tC = trstack.mean(0).argmin(-1)                                 # (2,)  pooled over participants, per model
bitsB = math.log2(200) - CEm.gather(-1, tB[:, :, None, None].expand(P, 2, 3, 1)).squeeze(-1)
bitsC = math.log2(200) - CEm.gather(-1, tC[None, :, None, None].expand(P, 2, 3, 1)).squeeze(-1)
# k80 = index 0, k40 = mean of halves (1, 2)
def kk(b): return {'k80': b[..., 0], 'k40': (b[..., 1] + b[..., 2]) / 2}
res = {p: kk(b) for p, b in (('A', bitsApro), ('B', bitsB), ('C', bitsC))}
# G1 vs saved
g1 = {}
for i, s in enumerate(subs):
    d = json.load(open(f'{SAVED}/ubp_card_soft12_sub{s}.json'))['subjects'][str(s)]
    for mi, (mn, key) in enumerate((('raw', 'released'), ('base', 'random_init_ridge'))):
        for k in ('k80', 'k40'): g1[f'sub{s}/{mn}/{k}'] = abs(float(res['A'][k][i, mi]) - d[key][f'{k}_bits'])
g1max = max(g1.values()); print('G1 max abs diff', g1max, 'PASS' if g1max < 1e-4 else 'FAIL', flush=True)

# ---- bootstrap (participants, 10000, GPU)
gen = torch.Generator(device=dev); gen.manual_seed(0); NB = 10000
bi = torch.randint(0, P, (NB, P), device=dev, generator=gen)
def ci(v):  # v (P,) -> mean, lo, hi
    bm = v[bi].mean(1); return [float(v.mean()), float(bm.quantile(0.025)), float(bm.quantile(0.975))]
out = {'code_sha256': CODE_SHA256, 'subs': subs, 'G1_max_abs_diff': g1max, 'G1_pass': g1max < 1e-4, 'ndraw': a.ndraw, 'tau_grid': TAUS.tolist(),
       'tau_B_per_participant': {str(s): {'raw': float(TAUS[tB[i, 0]]), 'base': float(TAUS[tB[i, 1]])} for i, s in enumerate(subs)},
       'tau_C': {'raw': float(TAUS[tC[0]]), 'base': float(TAUS[tC[1]])}, 'ridge_lambda': {str(s): trlam[s] for s in subs},
       'per_participant': {}, 'group': {}, 'status': ''}
for k in ('k80', 'k40'):
    for p in 'ABC':
        raw, base = res[p][k][:, 0], res[p][k][:, 1]; net = raw - base; res[p][k + '_net'] = net
        out['group'][f'{k}/{p}'] = {'raw': ci(raw), 'base': ci(base), 'net': ci(net)}
        for i, s in enumerate(subs): out['per_participant'].setdefault(str(s), {})[f'{k}/{p}'] = {'raw': float(raw[i]), 'base': float(base[i]), 'net': float(net[i])}
    for q in 'BC':
        pass
    for q in 'BC':
        d = res['A'][k + '_net'] - res[q][k + '_net']; netA = res['A'][k + '_net']
        pct = (netA[bi].mean(1) - res[q][k + '_net'][bi].mean(1)) / netA[bi].mean(1) * 100
        out['group'][f'{k}/infl_A-{q}'] = {'raw': ci(res['A'][k][:, 0] - res[q][k][:, 0]), 'base': ci(res['A'][k][:, 1] - res[q][k][:, 1]), 'net': ci(d),
                                          'net_pct_of_netA': [float(((netA.mean() - res[q][k + '_net'].mean()) / netA.mean()) * 100), float(pct.quantile(0.025)), float(pct.quantile(0.975))]}
        for i, s in enumerate(subs): out['per_participant'][str(s)][f'{k}/infl_A-{q}'] = {'raw': float(res['A'][k][i, 0] - res[q][k][i, 0]), 'base': float(res['A'][k][i, 1] - res[q][k][i, 1]), 'net': float(d[i])}
fn = os.path.join(a.out, f'train_only_calib_ubp{a.tag}.json'); json.dump(out, open(fn, 'w'), indent=1)
print(json.dumps(out['group'], indent=0)); print('tau_C', out['tau_C'], 'wrote', fn, f'{time.time()-t0:.0f}s')
