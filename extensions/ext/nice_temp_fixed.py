"""A4 (IDEATOR_LIFT_ABOVE_5_20261124): exposure control for temperature = yield at a FIXED, training-time temperature versus the test-calibrated yield.
NOT a rescue of the stopped E2 run (results/ext/nice_train_only_temp, STOPPED_BY_ASSERTION): no validation set, no training, a different pre-registered estimand
(Delta_L = (Y_L - Y_fixed) / Y_L, an UPPER BOUND on what calibrating the temperature on labelled test queries adds, because the training temperature was fitted to
4-repetition averages and is scored on 80-repetition averages). Pre-registration: results/ext/nice_temp_fixed/PREREG.json (+ PREREG_STAMP.json).
Decoders: NICE-GA (seed-141 checkpoints, tau = 0.07 = the training constant LOGIT = 1/0.07 of nice_stand_ga_dgx_v2.py, authenticated in E2 stage 1 by A1)
          BrainFLORA released checkpoint (tau = 1 / raw logit_scale stored in the checkpoint; used raw, as in the released losses), descriptive.
Scorer: eegbits/run.py soft_bits (LOO 12-point temperature; Y_L) versus the same cosine matrix scored at one fixed tau (bits = log2 200 - mean CE at S/tau).
Usage: python -I -u nice_temp_fixed.py --out DIR [--subs 1,2,..] [--smoke]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); O = ROOT / 'outputs/AGENT_LOOP_20261120'
sys.path[:0] = [str(ROOT / 'experiments/ext')]
import gf_image_untrained_v2 as G
import run as RUN                                              # eegbits/run.py (path set by G)
N = G.N; DEV = G.DEV
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:6.1f}s]', *m, flush=True)
MULTS = [0.5, 0.75, 1.0, 4 / 3, 2.0]; TAU_NICE = 0.07; FLOOR_MAX = 0.02; GATE = 1e-4; BOOT = 10000; SMALL_BOUND = 0.10
SUBS = [int(s) for s in a.subs.split(',')]


def fixed_bits(S, tau):
    Z = torch.as_tensor(np.asarray(S), dtype=torch.float64, device=DEV) / tau; n = Z.shape[0]
    return float(math.log2(n) + (Z.diagonal() - torch.logsumexp(Z, 1)).mean() / math.log(2))


def picks(S):
    """replica of soft_bits' leave-one-query-out temperature picks (to unit-check fixed_bits against soft_bits when every pick is the same tau)"""
    n = S.shape[0]; rows = np.arange(n)
    def ce_all(tau):
        Z = S / tau; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); return -lp[rows, rows] / np.log(2)
    CE = np.stack([ce_all(t) for t in RUN._TAUS]); tot = CE.sum(1, keepdims=True); loo = (tot - CE) / (n - 1); return loo.argmin(0)


def score_matrices(emb_fn, probe_fn, C, x, rng, perms):
    """cosine matrices for k=80, two k=40 halves, noise inputs (k=80, k=40, drawn exactly as gf_image_untrained_v2.card), shuffled-pairing perms of k=80"""
    nrm = G.nrm; P80 = emb_fn(x.mean(1)); Pa = emb_fn(x[:, :40].mean(1)); Pb = emb_fn(x[:, 40:].mean(1))
    nz = lambda k: rng.standard_normal((200, 63, 250)).astype(np.float32) / math.sqrt(k)
    Pn80 = emb_fn(nz(80)); Pn40 = emb_fn(nz(40))
    f = lambda P: nrm(P) @ nrm(C).T
    S = dict(k80=f(P80), a=f(Pa), b=f(Pb), n80=f(Pn80), n40=f(Pn40)); S['shuf'] = [S['k80'][p] for p in perms]; return S


def evaluate(S, taus):
    r = dict(YL80=float(RUN.soft_bits(S['k80'])), YL40=0.5 * (float(RUN.soft_bits(S['a'])) + float(RUN.soft_bits(S['b']))),
             floor_soft12=dict(noise80=float(RUN.soft_bits(S['n80'])), noise40=float(RUN.soft_bits(S['n40'])), shuffled80=float(np.mean([RUN.soft_bits(q) for q in S['shuf']]))))
    pk = picks(S['k80']); r['soft12_picks_k80'] = {f'{RUN._TAUS[i]:.4f}': int((pk == i).sum()) for i in np.unique(pk)}
    if len(np.unique(pk)) == 1:
        r['unit_check_fixed_at_single_pick_vs_soft12'] = abs(fixed_bits(S['k80'], RUN._TAUS[int(pk[0])]) - r['YL80'])
    r['fixed'] = {}
    for m, tau in taus.items():
        r['fixed'][m] = dict(tau=tau, Yf80=fixed_bits(S['k80'], tau), Yf40=0.5 * (fixed_bits(S['a'], tau) + fixed_bits(S['b'], tau)),
                             floor_noise80=fixed_bits(S['n80'], tau), floor_noise40=fixed_bits(S['n40'], tau), floor_shuffled80=float(np.mean([fixed_bits(q, tau) for q in S['shuf']])))
    return r


def boot_delta(yl, yf):
    yl = torch.tensor(yl, dtype=torch.float64, device=DEV); yf = torch.tensor(yf, dtype=torch.float64, device=DEV); n = len(yl)
    g = torch.Generator(device=DEV); g.manual_seed(0); ix = torch.randint(n, (BOOT, n), device=DEV, generator=g)
    d = (yl[ix].mean(1) - yf[ix].mean(1)) / yl[ix].mean(1); pt = float((yl.mean() - yf.mean()) / yl.mean())
    return dict(delta_ratio_of_means=pt, ci95_two_sided=[float(d.quantile(.025)), float(d.quantile(.975))], upper_one_sided_95=float(d.quantile(.95)),
                mean_of_per_participant_ratios=float(((yl - yf) / yl).mean()), n=n, mean_YL=float(yl.mean()), mean_Yf=float(yf.mean()))


def summarise(per, name, taus_keys, gate_ok):
    s = dict(decoder=name, n=len(per), per_participant=per, G1_all_pass=gate_ok)
    # floors: literal ideator reading = value at most 0.02 (one-sided); |.| reading reported beside it
    fl = {}
    for m in taus_keys:
        vals = [v for p in per.values() for v in (p['fixed'][m]['floor_noise80'], p['fixed'][m]['floor_noise40'], p['fixed'][m]['floor_shuffled80'])]
        fl[m] = {'max_value': max(vals), 'min_value': min(vals), 'one_sided_ok_le_0.02': bool(max(vals) <= FLOOR_MAX), 'abs_ok_le_0.02': bool(max(abs(v) for v in vals) <= FLOOR_MAX)}
    s['floors_at_fixed_tau'] = fl; s['floors_gate_one_sided_all_tau'] = all(v['one_sided_ok_le_0.02'] for v in fl.values())
    s['floors_gate_abs_all_tau_reported_not_the_gate'] = all(v['abs_ok_le_0.02'] for v in fl.values())
    s['floors_soft12_abs_max'] = max(abs(v) for p in per.values() for v in p['floor_soft12'].values())
    s['delta_k80'] = {m: boot_delta([p['YL80'] for p in per.values()], [p['fixed'][m]['Yf80'] for p in per.values()]) for m in taus_keys}
    s['delta_k40'] = {m: boot_delta([p['YL40'] for p in per.values()], [p['fixed'][m]['Yf40'] for p in per.values()]) for m in taus_keys}
    # floor-adjusted fixed yield, descriptive only: Yf - mean shuffled floor at the same tau
    s['descriptive_Yf80_minus_own_shuffled_floor'] = {m: float(np.mean([p['fixed'][m]['Yf80'] - p['fixed'][m]['floor_shuffled80'] for p in per.values()])) for m in taus_keys}
    um = [p['unit_check_fixed_at_single_pick_vs_soft12'] for p in per.values() if 'unit_check_fixed_at_single_pick_vs_soft12' in p]
    s['unit_check_fixed_scorer_equals_soft12_when_one_tau_picked'] = dict(n_participants_checked=len(um), max_abs_diff=max(um) if um else None, ok=bool(um and max(um) <= GATE))
    return s


res = dict(NOT_CITABLE='new output, no blind review yet', code_sha256=CODE_SHA256, script=str(Path(__file__).resolve()), smoke=a.smoke,
           not_a_rescue_of='results/ext/nice_train_only_temp (E2, STOPPED_BY_ASSERTION: the 740 validation trials are not whole concepts). This run uses no validation set and no training.',
           multipliers=MULTS, tau_nice=TAU_NICE, subs=SUBS)
taus_n = {f'{m:.3g}x': TAU_NICE * m for m in MULTS}; KEYS = list(taus_n)
Yc = np.squeeze(np.load(N.NICE / 'dnn_feature/center_clip.npy', allow_pickle=True)).astype(np.float64)

# ------------------------------------------------------------------ NICE-GA
Enc, Proj = N.load_classes('nice_ga'); per = {}; g1 = {}
for s in SUBS:
    sub = f'sub-{s:02d}'; tag = f'nice_ga_dgx_v2_sub{s:02d}_seed141'
    enc, proj = Enc().to(DEV).eval(), Proj().to(DEV).eval()
    enc.load_state_dict(torch.load(N.NICE / 'model' / f'{tag}_Enc_eeg_cls.pth', weights_only=True, map_location=DEV)); proj.load_state_dict(torch.load(N.NICE / 'model' / f'{tag}_Proj_eeg_cls.pth', weights_only=True, map_location=DEV))
    x = np.asarray(np.load(ROOT / f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy', mmap_mode='r'), dtype=np.float32); assert x.shape == (200, 80, 63, 250)
    rng = np.random.default_rng(31 + s); perms = [rng.permutation(200) for _ in range(20)]
    S = score_matrices(lambda X: G.embed(lambda t: proj(enc(t.unsqueeze(1))), X), None, Yc, x, rng, perms)
    r = evaluate(S, taus_n)
    ref = json.loads((O / f'results/u1_native_v2/{sub}.json').read_text()); sv = [q['eeg']['InfoNCE'] for q in ref['rows'] if q['model'] == 'nice_ga' and q['k'] == 80][0]
    r['G1_saved_u1_native_v2_k80'] = sv; r['G1_abs_diff'] = abs(r['YL80'] - sv); g1[sub] = r['G1_abs_diff']; per[sub] = r
    log('nice', sub, 'YL80 %.4f' % r['YL80'], 'Yf80@1x %.4f' % r['fixed']['1x']['Yf80'], 'G1 %.2e' % r['G1_abs_diff'], 'picks', r['soft12_picks_k80'])
    del enc, proj, x, S; torch.cuda.empty_cache()
res['NICE_GA'] = summarise(per, 'NICE-GA seed 141, tau fixed at 0.07', KEYS, bool(max(g1.values()) <= GATE)); res['NICE_GA']['G1_max_abs_diff'] = max(g1.values())

# ------------------------------------------------------------------ BrainFLORA (descriptive; only if the released checkpoint exposes its training logit scale)
try:
    sys.path[:0] = [str(O / 'models/brainflora/source')]
    from model.medformer_encoders import eeg_encoder
    ck = torch.load(O / 'models/brainflora/eeg_01-06_01-46_150.pth', map_location=DEV, weights_only=False); sd = ck.get('state_dict', ck.get('model', ck)) if isinstance(ck, dict) else ck
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    if 'logit_scale' not in sd: raise KeyError('logit_scale not in the released checkpoint')
    ls = float(sd['logit_scale']); tau_b = 1.0 / ls                      # the released losses multiply cosine by logit_scale raw (utils/losses.py ClipLoss, eval/reproduce_retrieval.py); no exp
    benc = eeg_encoder().to(DEV).eval(); lm = benc.load_state_dict(sd, strict=False)
    Yb = torch.load(O / 'models/brainflora/ViT-H-14_features_test.pt', map_location='cpu', weights_only=True); Yb = (Yb['img_features'] if isinstance(Yb, dict) else Yb).float().numpy().reshape(200, -1).astype(np.float64)
    taus_b = {f'{m:.3g}x': tau_b * m for m in MULTS}; perb = {}; g1b = {}
    def bemb(X, bs=100):
        with torch.inference_mode(): return torch.cat([benc(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(DEV)).float() for i in range(0, len(X), bs)]).cpu().numpy().astype(np.float64)
    for s in SUBS:
        sub = f'sub-{s:02d}'; x = np.asarray(np.load(ROOT / f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy', mmap_mode='r'), dtype=np.float32)
        rng = np.random.default_rng(31 + s); perms = [rng.permutation(200) for _ in range(20)]
        S = score_matrices(bemb, None, Yb, x, rng, perms); r = evaluate(S, taus_b)
        ref = json.loads((O / f'results/u1_native_v2/{sub}.json').read_text()); sv = [q['eeg']['InfoNCE'] for q in ref['rows'] if q['model'] == 'brainflora' and q['k'] == 80][0]
        r['G1_saved_u1_native_v2_k80'] = sv; r['G1_abs_diff'] = abs(r['YL80'] - sv); g1b[sub] = r['G1_abs_diff']; perb[sub] = r
        log('brainflora', sub, 'YL80 %.4f' % r['YL80'], 'Yf80@1x %.4f' % r['fixed']['1x']['Yf80'], 'G1 %.2e' % r['G1_abs_diff']); del x, S
    res['BrainFLORA'] = summarise(perb, 'BrainFLORA released checkpoint, tau fixed at 1/logit_scale', KEYS, bool(max(g1b.values()) <= GATE))
    res['BrainFLORA'].update(logit_scale_raw=ls, tau_train=tau_b, G1_max_abs_diff=max(g1b.values()), load_state_dict_missing=len(lm.missing_keys), load_state_dict_unexpected=len(lm.unexpected_keys))
except Exception as e:
    res['BrainFLORA'] = dict(status='not available', reason=repr(e)[:300])
    log('BrainFLORA not available', repr(e)[:300])

# ------------------------------------------------------------------ UBP: stored held-out-concepts estimate (not recomputed)
u = json.loads((ROOT / 'results/gpu_fill_20261123/ubp_heldout/summary.json').read_text())
res['UBP_reference_not_recomputed'] = dict(source='results/gpu_fill_20261123/ubp_heldout/summary.json (held-out training concepts, F4)', inflation_pct=u['inflation_pct'], n_unchanged=u['n_unchanged'],
                                           note='a different estimand (temperature fixed on held-out training concepts), not the same as Delta_L; do not put in one comparison with NICE-GA')

# ------------------------------------------------------------------ pre-registered verdict (NICE-GA, k=80, 1x only)
n = res['NICE_GA']; d = n['delta_k80']['1x']
void = (not n['G1_all_pass']) or (not n['floors_gate_one_sided_all_tau'])
res['verdict_NICE_GA_k80_1x'] = dict(delta=d['delta_ratio_of_means'], ci95=d['ci95_two_sided'], upper_bound_used=d['ci95_two_sided'][1], rule='small only if the upper 95% bound of Delta_L is at most 10%',
                                     void=bool(void), void_reasons=([] if n['G1_all_pass'] else ['G1 (reproduce u1_native_v2 soft12 k=80 to 1e-4 for all participants) failed']) + ([] if n['floors_gate_one_sided_all_tau'] else ['a noise or shuffled floor at a fixed tau exceeds 0.02']),
                                     small=bool((not void) and d['ci95_two_sided'][1] <= SMALL_BOUND), result=('VOID' if void else ('SMALL (upper bound <= 10%)' if d['ci95_two_sided'][1] <= SMALL_BOUND else 'NOT SMALL: state the larger number, called an upper bound')))
res['runtime_s'] = time.time() - t0
(out / 'summary.json').write_text(json.dumps(res, indent=1)); log('VERDICT', json.dumps(res['verdict_NICE_GA_k80_1x']))
