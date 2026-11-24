"""E2 nice_train_only_temp, STAGE 1 = assertions (RUNNER, 24 Nov 2026).
Pre-registration: results/ext/nice_train_only_temp/PREREG.json (+ PREREG_STAMP.json). The inflation stage is NOT in this file: per the stop rule it is written
only if A1, A2 and A3 all hold.
A1  the validation set of nice_stand_ga_dgx_v2.py (first 740 of np.random.permutation(16540) after np.random.seed(141)) is reconstructed and AUTHENTICATED: the saved
    seed-141 checkpoint (best-validation-loss epoch) must reproduce the validation loss written in the training log at its best epoch (5e-4); a decoy set of 740 other
    training trials must give a clearly different loss.
A2  the 740 validation trials are WHOLE concepts disjoint from training (concept = image index // 10; every concept with a validation image has all 10 images in validation).
A3  trained seed-141 NICE-GA scores through the gf_image_untrained_v2 loader/scorer reproduce outputs/AGENT_LOOP_20261120/results/u1_native_v2 k=80 InfoNCE to 1e-4.
Usage: python -I -u nice_train_only_temp.py --out DIR [--subs 1,2,..]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path('/path/to/eegaudit'); O = ROOT / 'outputs/AGENT_LOOP_20261120'
sys.path[:0] = [str(ROOT / 'experiments/ext')]
import gf_image_untrained_v2 as G           # loader and scorer of the protocol (embed, bits, N.load_classes, path setup)
N = G.N; NICE = N.NICE; DEV = G.DEV
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:6.1f}s]', *m, flush=True)
SEED = 141; NTR = 16540; NVAL = 740; LOGIT = float(np.exp(np.log(1 / 0.07)))
perm = np.random.RandomState(SEED).permutation(NTR); val_idx = np.sort(perm[:NVAL]); tr_idx = np.sort(perm[NVAL:]); decoy_idx = np.sort(perm[NVAL:2 * NVAL])
assert len(set(val_idx) & set(tr_idx)) == 0 and len(val_idx) == NVAL and len(tr_idx) == NTR - NVAL
res = dict(NOT_CITABLE='new output, no blind review yet', code_sha256=CODE_SHA256, script=str(Path(__file__).resolve()), seed=SEED, stage='1 (assertions only)',
           val_idx_sha256=_hl.sha256(val_idx.astype('<i8').tobytes()).hexdigest(), per_participant={})

# ---- A2: concept structure of the validation set (does not depend on the participant)
conc = np.arange(NTR) // 10
vc, vcount = np.unique(conc[val_idx], return_counts=True)
n_whole = int((vcount == 10).sum()); n_mate_in_train = int(sum(c for c in vcount if c < 10)); n_val_images_with_mate_in_train = int(sum(vcount[vcount < 10]))
Yc_tr = np.squeeze(np.load(NICE / 'dnn_feature/clip_feature_maps_training.npy', allow_pickle=True)).astype(np.float64); Yc_tr = Yc_tr / np.linalg.norm(Yc_tr, axis=1, keepdims=True)
G10 = Yc_tr.reshape(1654, 10, -1); within = float(np.mean([(G10[c] @ G10[c].T)[~np.eye(10, dtype=bool)].mean() for c in range(1654)]))
rr = np.random.RandomState(0); across = float(np.mean([Yc_tr[i] @ Yc_tr[j] for i, j in zip(rr.randint(0, NTR, 20000), rr.randint(0, NTR, 20000)) if i // 10 != j // 10]))
A2 = dict(n_validation_trials=NVAL, n_concepts_with_any_validation_image=int(len(vc)), n_concepts_with_all_10_images_in_validation=n_whole,
          n_validation_images_whose_concept_has_images_in_training=n_val_images_with_mate_in_train, frac_validation_images_with_concept_mates_in_training=n_val_images_with_mate_in_train / NVAL,
          count_per_concept_histogram={int(k): int((vcount == k).sum()) for k in np.unique(vcount)},
          n_training_concepts_also_in_validation=int(len(vc)), max_images_of_one_concept_in_val=int(vcount.max()),
          clip_cos_within_concept_groups_of_10=within, clip_cos_across_groups=across,
          note_concept_def='THINGS-EEG2 training images are 1654 concepts x 10 images in contiguous order; checked by CLIP cosine within vs across groups of 10',
          assertion='every concept that has a validation image has all 10 of its images in validation', passed=bool(len(vc) > 0 and n_whole == len(vc)))
res['A2_validation_trials_are_whole_concepts'] = A2
log('A2', json.dumps(A2))

# ---- per participant: A1 (authenticate reconstruction through the saved checkpoint's validation loss) and A3 (u1_native_v2 reproduction)
Enc, Proj = N.load_classes('nice_ga')
Yc = np.squeeze(np.load(NICE / 'dnn_feature/center_clip.npy', allow_pickle=True)).astype(np.float64)
Yt = torch.tensor(Yc_tr, dtype=torch.float32, device=DEV)
def val_loss(enc, proj, X, idx):
    with torch.no_grad():
        e = torch.cat([proj(enc(torch.from_numpy(np.ascontiguousarray(X[i:i + 370])).float().to(DEV).unsqueeze(1))) for i in range(0, len(X), 370)]); e = e / e.norm(dim=1, keepdim=True)
        im = Yt[torch.from_numpy(idx).to(DEV)]; im = im / im.norm(dim=1, keepdim=True)     # Proj_img.forward is identity in the v2 GA script; features are normalised in the loss
        lg = LOGIT * e @ im.T; lab = torch.arange(len(idx), device=DEV)
        return float((F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2)
a1_ok, a3_ok = True, True
for s in [int(x) for x in a.subs.split(',')]:
    sub = f'sub-{s:02d}'; tag = f'nice_ga_dgx_v2_sub{s:02d}_seed{SEED}'; R = {}
    enc, proj = Enc().to(DEV).eval(), Proj().to(DEV).eval()
    enc.load_state_dict(torch.load(NICE / 'model' / f'{tag}_Enc_eeg_cls.pth', weights_only=True, map_location=DEV)); proj.load_state_dict(torch.load(NICE / 'model' / f'{tag}_Proj_eeg_cls.pth', weights_only=True, map_location=DEV))
    lg = (Path('/path/to/image_benchmark/results/nice') / f'log_subject{s:02d}_ga_v2_seed{SEED}.txt').read_text()
    best = int(re.search(r'The best epoch is: (\d+)', lg).group(1)); logged = float(re.search(rf'^Epoch {best - 1}: .*loss val: ([0-9.]+)', lg, re.M).group(1))
    d = np.load(ROOT / f'data_raw/things_eeg2_250hz/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy', allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
    arr = d['preprocessed_eeg_data']; assert arr.shape[0] == NTR, arr.shape
    Xv = np.asarray(arr[val_idx], dtype=np.float32).mean(1); Xd = np.asarray(arr[decoy_idx], dtype=np.float32).mean(1); del d, arr
    lv, ld = val_loss(enc, proj, Xv, val_idx), val_loss(enc, proj, Xd, decoy_idx)
    R['A1'] = dict(best_epoch=best, logged_val_loss=logged, recomputed_val_loss_on_reconstructed_validation=lv, abs_diff=abs(lv - logged), decoy_740_training_trials_loss=ld,
                   passed=bool(abs(lv - logged) <= 5e-4 and abs(ld - logged) > 0.1)); a1_ok &= R['A1']['passed']
    # A3: u1_native_v2 reproduction through the gf_image_untrained_v2 scorer
    x = np.asarray(np.load(ROOT / f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy', mmap_mode='r'), dtype=np.float32); assert x.shape == (200, 80, 63, 250)
    g1 = G.bits(G.embed(lambda t: proj(enc(t.unsqueeze(1))), x.mean(1)), Yc)
    ref = json.loads((O / f'results/u1_native_v2/{sub}.json').read_text()); sv = [r['eeg']['InfoNCE'] for r in ref['rows'] if r['model'] == 'nice_ga' and r['k'] == 80][0]
    R['A3'] = dict(this_k80_infonce=g1, saved_u1_native_v2=sv, abs_diff=abs(g1 - sv), passed=bool(abs(g1 - sv) <= 1e-4)); a3_ok &= R['A3']['passed']
    res['per_participant'][sub] = R; log(sub, json.dumps(R)); del enc, proj, x, Xv, Xd; torch.cuda.empty_cache()

res['A1_passed_all'] = a1_ok; res['A3_passed_all'] = a3_ok; res['A2_passed'] = A2['passed']
res['status'] = 'CONTINUE_TO_STAGE_2' if (a1_ok and A2['passed'] and a3_ok) else 'STOPPED_BY_ASSERTION'
res['stop_reason'] = None if res['status'] == 'CONTINUE_TO_STAGE_2' else ('; '.join(
    (['A1 failed: the reconstructed validation set is not authenticated'] if not a1_ok else []) +
    ([f"A2 failed: the 740 validation trials are not whole concepts ({A2['n_concepts_with_any_validation_image']} concepts touched, {n_whole} whole, {A2['frac_validation_images_with_concept_mates_in_training']:.3f} of the validation images have concept mates in training)"] if not A2['passed'] else []) +
    (['A3 failed: u1_native_v2 not reproduced'] if not a3_ok else [])))
res['no_inflation_computed'] = True; res['runtime_s'] = time.time() - t0
(out / 'summary.json').write_text(json.dumps(res, indent=1)); log('STATUS', res['status'], res['stop_reason'])
