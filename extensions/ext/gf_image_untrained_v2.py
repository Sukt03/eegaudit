"""GPU-FILL v2 UNTRAINED-ENCODER baselines for NICE-GA and BrainFLORA on THINGS-EEG2 (, 23 Nov 2026).
Same recipe as the UBP random-init twin in experiments/baselines/ubp_card.py (frozen random-init encoder, GPU ridge probe, penalty chosen on a held-out tenth of the training images)
and the same scorer as the trained models (eegbits/run.py soft_bits: 12-point temperature grid, leave-one-query-out; log2 200 minus CE).
  nice_ga    run_acq_nice.load_classes('nice_ga') Enc_eeg+Proj_eeg (published source), weights_init_normal as at the start of its training, torch seed, eval mode; output 768-d; target CLIP ViT-L/14 train features
             (NICE dnn_feature/clip_feature_maps_training.npy), candidates center_clip.npy (the trained model's own 200 test centres)
  brainflora medformer_encoders.eeg_encoder (BrainFLORA source) at library init, eval mode; output 1024-d; target ViT-H-14 train features (bbeeg-atm hf_probe), candidates ViT-H-14_features_test.pt
Training EEG = preprocessed_eeg_training.npy averaged over repetitions (16540 x 63 x 250), as NICE trains; test = test_cache (200 x 80 x 63 x 250); k=80 uses all repetitions, k=40 the mean of the two halves' scores (as UBP card).
Floors per participant: noise input N(0,1)/sqrt(k) through the same frozen encoder + probe; shuffled candidates (mean of 20 permutations). Gate G1: the TRAINED NICE-GA checkpoint (seed 141) scored here with the same scorer reproduces
outputs/AGENT_LOOP_20261120/results/u1_native_v2/<sub>.json k=80 InfoNCE to 1e-4 (pipeline identity; float32 batch-size noise ~2e-6); floors <= 0.02 bits.
Usage: --out DIR --subs sub-01,sub-02 --models nice_ga,brainflora --seeds 0,1,2"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torch.nn.init as init
ROOT = Path('/path/to/eegaudit'); O = ROOT / 'outputs/AGENT_LOOP_20261120'
sys.path[:0] = [str(O / 'models/brainflora/source'), str(ROOT / 'experiments/eegbits')]
from run import soft_bits
import run_acq_nice as N
DEV = 'cuda'; t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)

def winit(m):                                              # nice_stand_ga_dgx_v2.weights_init_normal, verbatim semantics
    c = m.__class__.__name__
    if c.find('Conv') != -1 and hasattr(m, 'weight'): init.normal_(m.weight.data, 0.0, 0.02)
    elif c.find('Linear') != -1: init.normal_(m.weight.data, 0.0, 0.02)
    elif c.find('BatchNorm') != -1: init.normal_(m.weight.data, 1.0, 0.02); init.constant_(m.bias.data, 0.0)

def load_feat(p):
    v = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(v, dict): v = v['img_features']
    return v.float().numpy().reshape(len(v), -1)

def embed(fn, X, bs=500):
    with torch.inference_mode(): return torch.cat([fn(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(DEV)).float() for i in range(0, len(X), bs)]).cpu().numpy().astype(np.float64)

def fit_probe(H, Y, seed=0):
    H = torch.tensor(H, device=DEV, dtype=torch.float64); Y = torch.tensor(Y, device=DEV, dtype=torch.float64)
    perm = torch.randperm(len(H), generator=torch.Generator().manual_seed(seed)).to(DEV); nv = len(H) // 10; va, tr = perm[:nv], perm[nv:]
    mu = H[tr].mean(0); ym = Y[tr].mean(0); U, S, Vh = torch.linalg.svd(H[tr] - mu, full_matrices=False); UtY = U.T @ (Y[tr] - ym); best = (-9, None)
    for lam in torch.logspace(-2, 6, 25, dtype=torch.float64, device=DEV):
        W = Vh.T @ ((S / (S ** 2 + lam))[:, None] * UtY); sc = float(F.cosine_similarity((H[va] - mu) @ W + ym, Y[va], dim=1).mean())
        if sc > best[0]: best = (sc, float(lam))
    mu = H.mean(0); ym = Y.mean(0); U, S, Vh = torch.linalg.svd(H - mu, full_matrices=False); W = Vh.T @ ((S / (S ** 2 + best[1]))[:, None] * (U.T @ (Y - ym)))
    return (lambda X: ((torch.tensor(X, device=DEV, dtype=torch.float64) - mu) @ W + ym).cpu().numpy()), best

def nrm(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
def bits(P, C): return float(soft_bits(nrm(P) @ nrm(C).T))
def card(emb_fn, probe, x, C, rng, perms):
    """x [200, 80, 63, 250]. Returns dict k80, k40 (mean of two halves), noise floors, shuffled floor."""
    e80 = probe(emb_fn(x.mean(1))); ea = probe(emb_fn(x[:, :40].mean(1))); eb = probe(emb_fn(x[:, 40:].mean(1)))
    r = dict(k80=bits(e80, C), k40=0.5 * (bits(ea, C) + bits(eb, C)))
    nz = lambda k: rng.standard_normal((200, 63, 250)).astype(np.float32) / math.sqrt(k)
    r['noise_k80'] = bits(probe(emb_fn(nz(80))), C); r['noise_k40'] = bits(probe(emb_fn(nz(40))), C)
    r['shuffled_k80'] = float(np.mean([bits(e80[p], C) for p in perms])); return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default=','.join(f'sub-{i:02d}' for i in range(1, 11)))
    ap.add_argument('--models', default='nice_ga,brainflora'); ap.add_argument('--seeds', default='0,1,2'); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]; models = a.models.split(',')
    from model.medformer_encoders import eeg_encoder
    Enc, Proj = N.load_classes('nice_ga'); NICE = N.NICE
    Yc = np.squeeze(np.load(NICE / 'dnn_feature/center_clip.npy', allow_pickle=True)).astype(np.float64)
    Ytr_c = np.squeeze(np.load(NICE / 'dnn_feature/clip_feature_maps_training.npy', allow_pickle=True)).astype(np.float64)
    Yb_te = load_feat('/path/to/atm/data/hf_probe/ViT-H-14_features_test.pt').astype(np.float64); Yb_tr = load_feat('/path/to/atm/data/hf_probe/ViT-H-14_features_train.pt').astype(np.float64)
    assert Yb_tr.shape[0] == Ytr_c.shape[0] == 16540, (Yb_tr.shape, Ytr_c.shape)
    for sub in a.subs.split(','):
        dest = out / f'{sub}.json'
        if dest.exists(): continue
        tr = np.load(ROOT / f'data_raw/things_eeg2_250hz/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy', allow_pickle=True)
        tr = np.asarray((tr.item() if isinstance(tr, np.ndarray) else tr)['preprocessed_eeg_data'], dtype=np.float32).mean(1); assert tr.shape == (16540, 63, 250), tr.shape
        x = np.load(ROOT / f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy', mmap_mode='r'); x = np.asarray(x, dtype=np.float32); assert x.shape == (200, 80, 63, 250)
        rng = np.random.default_rng(31 + int(sub[-2:])); perms = [rng.permutation(200) for _ in range(20)]; res = dict(participant=sub, code_sha256=CODE_SHA256, models={})
        ref = json.loads((O / f'results/u1_native_v2/{sub}.json').read_text()) if (O / f'results/u1_native_v2/{sub}.json').exists() else None
        if 'nice_ga' in models and ref is not None:        # G1: trained NICE-GA checkpoint through this scorer == saved native k80 InfoNCE
            tag = f'nice_ga_dgx_v2_{sub.replace("-", "")}_seed141'; enc, proj = Enc().to(DEV).eval(), Proj().to(DEV).eval()
            enc.load_state_dict(torch.load(NICE / 'model' / f'{tag}_Enc_eeg_cls.pth', weights_only=True, map_location=DEV)); proj.load_state_dict(torch.load(NICE / 'model' / f'{tag}_Proj_eeg_cls.pth', weights_only=True, map_location=DEV))
            g1 = bits(embed(lambda t: proj(enc(t.unsqueeze(1))), x.mean(1)), Yc); sv = [r['eeg']['InfoNCE'] for r in ref['rows'] if r['model'] == 'nice_ga' and r['k'] == 80][0]
            res['G1_trained_nice_ga_k80'] = dict(this=g1, saved=sv, abs_diff=abs(g1 - sv)); assert abs(g1 - sv) < 1e-4, f'G1 FAIL {sub}: {g1} vs {sv}'; log(sub, 'G1', res['G1_trained_nice_ga_k80']); del enc, proj
        if ref is not None:
            for m in models:
                rr = [r for r in ref['rows'] if r['model'] == m]; res.setdefault('trained_saved', {})[m] = dict(k80=[r['eeg']['InfoNCE'] for r in rr if r['k'] == 80][0], k40_mean=float(np.mean([r['eeg']['InfoNCE'] for r in rr if r['k'] == 40])))
        for m in models:
            Yt, C = (Ytr_c, Yc) if m == 'nice_ga' else (Yb_tr, Yb_te); rows = []
            for sd in seeds:
                torch.manual_seed(7000 + sd)
                if m == 'nice_ga':
                    enc, proj = Enc().to(DEV), Proj().to(DEV); enc.apply(winit); proj.apply(winit); enc.eval(); proj.eval(); fn = lambda t: proj(enc(t.unsqueeze(1)))
                else:
                    enc = eeg_encoder().to(DEV).eval(); fn = lambda t: enc(t)
                emb_fn = lambda X: embed(fn, X); H = emb_fn(tr); probe, best = fit_probe(H, Yt)
                r = card(emb_fn, probe, x, C, rng, perms); r.update(seed=sd, ridge_lambda=best[1], ridge_dev_cos=best[0], emb_dim=H.shape[1]); rows.append(r)
                log(sub, m, f'seed {sd}', {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}); del enc
            res['models'][m] = dict(rows=rows, **{k: float(np.mean([r[k] for r in rows])) for k in ('k80', 'k40', 'noise_k80', 'noise_k40', 'shuffled_k80')})
        dest.write_text(json.dumps(res, indent=1))
    # summary
    R = [json.loads(p.read_text()) for p in sorted(out.glob('sub-*.json'))]
    def boot(v, B=10000):
        v = torch.tensor(np.asarray(v, np.float64), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(0); m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=g)].mean(1)
        return dict(mean=float(v.mean()), ci95=[float(m.quantile(.025)), float(m.quantile(.975))], n=len(v))
    Sm = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, n_participants=len(R), seeds=seeds, bootstrap='participant, 10000 resamples, GPU torch, seed 0', unit='bits per query (soft12 LOO InfoNCE, 200-way)', G1_max_abs_diff=max([r['G1_trained_nice_ga_k80']['abs_diff'] for r in R if 'G1_trained_nice_ga_k80' in r] or [None]) if any('G1_trained_nice_ga_k80' in r for r in R) else None)
    for m in models:
        if all(m in r['models'] for r in R) and R:
            d = {k: boot([r['models'][m][k] for r in R]) for k in ('k80', 'k40', 'noise_k80', 'noise_k40', 'shuffled_k80')}
            if all('trained_saved' in r for r in R):
                tk80 = [r['trained_saved'][m]['k80'] for r in R]; tk40 = [r['trained_saved'][m]['k40_mean'] for r in R]
                d['trained_k80'] = boot(tk80); d['trained_k40'] = boot(tk40); d['share_of_trained_k80'] = boot([r['models'][m]['k80'] / t for r, t in zip(R, tk80)]); d['share_of_trained_k40'] = boot([r['models'][m]['k40'] / t for r, t in zip(R, tk40)])
                d['ratio_of_means_k80'] = float(np.mean([r['models'][m]['k80'] for r in R]) / np.mean(tk80)); d['ratio_of_means_k40'] = float(np.mean([r['models'][m]['k40'] for r in R]) / np.mean(tk40))
            d['floors_ok_le_0.02'] = bool(max(abs(d[k]['mean']) for k in ('noise_k80', 'noise_k40', 'shuffled_k80')) <= 0.02); Sm[m] = d
    (out / 'summary.json').write_text(json.dumps(Sm, indent=1)); log(json.dumps(Sm, indent=1)[:3500])

if __name__ == '__main__': main()
