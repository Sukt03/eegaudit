"""E1 glim_mde (RUNNER, 24 Nov 2026).
Derived from experiments/ext/glim_within_task.py (G1; same loader, MiniLM, infonce, bootstrap, card12 grid). Question: what is the smallest
amount of target information injected into the released GLIM EEG-arm generations that the G1 within-task scorer detects (sentence-cluster
95% lower bound of the paired EEG-minus-noise difference above 0) in BOTH ZuCo1 <NR> reading tasks (task1, task2)?
Mixing: e_mix = normalize((1 - lam) * e_gen + lam * e_target), e_target = MiniLM embedding of the stimulus sentence (the candidate = Input Text).
The noise-input arm is left unchanged. This is SCORER-LEVEL power (what the scorer would see if the decoder leaked lam of the target into its
generation embedding), NOT EEG-level power. Pre-registration: results/ext/glim_mde/PREREG.json (+ PREREG_STAMP.json).
Usage: python -I -u glim_mde.py --out results/ext/glim_mde [--smoke]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, csv, json, math, os, time
from collections import Counter, defaultdict
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/path/to/eegaudit/results/ext/glim_mde')
ap.add_argument('--minilm', default='/path/to/eegaudit/data_raw/zuco/minilm')
ap.add_argument('--n_boot', type=int, default=10000)
ap.add_argument('--smoke', action='store_true', help='every 7th record, 300 resamples, gates not enforced')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda')
torch.manual_seed(0)
t0 = time.time()
LOG2 = math.log(2)
GLIM = '/path/to/eegaudit/data_raw/third_party/glim_results'
FILES = {'eeg': 'wandb_export_gen_samples_glim.csv', 'noise_test_time': 'wandb_export_gen_samples_noise_input_test.csv',
         'prompt_free': 'wandb_export_gen_samples_prompt_free_test.csv'}
TAUS = torch.logspace(-2.5, 0, 12, device=dev)      # card12
LAMS = [0.02, 0.05, 0.1, 0.2, 0.4]
ALL_LAMS = [0.0] + LAMS + [1.0]
TASKS = ['cell:<NR>ZuCo1|task1', 'cell:<NR>ZuCo1|task2']
SAVED = json.load(open('/path/to/eegaudit/results/ext/glim_within_task/summary.json'))['selections']
n_boot = 300 if a.smoke else a.n_boot

from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(a.minilm); mdl = AutoModel.from_pretrained(a.minilm).to(dev).eval()
_cache = {}


@torch.no_grad()
def emb(ts):
    new = [t for t in dict.fromkeys(ts) if t not in _cache]
    for i in range(0, len(new), 256):
        b = tok(new[i:i + 256], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
        h = mdl(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
        _cache.update(zip(new[i:i + 256], e))
    return torch.stack([_cache[t] for t in ts])


def infonce(S, true, cl, taus):
    """identical to glim_within_task.infonce (leave-one-sentence-out temperature)."""
    Z = S[None] / taus[:, None, None]
    lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, torch.arange(len(true), device=dev), true] / LOG2
    ncl = int(cl.max()) + 1
    csum = torch.zeros(len(taus), ncl, device=dev).index_add_(1, cl, CE); ccnt = torch.bincount(cl, minlength=ncl).float()
    loo = (CE.sum(1, keepdim=True) - csum) / (len(true) - ccnt).clamp(min=1)
    t_idx = loo.argmin(0)[cl]
    return math.log2(S.shape[1]) - CE[t_idx, torch.arange(len(true), device=dev)], t_idx


def boot_sentence(vals, cl, seed=0):
    """sentence-cluster bootstrap, same draw scheme as glim_within_task.boot_ci(kind='sentence') (fresh generator seed 0 per call)."""
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    ncl = int(cl.max()) + 1
    out = {k: [] for k in vals}; nan = 0; done = 0
    while done < n_boot:
        nb = min(2500, n_boot - done)
        pick = torch.randint(0, ncl, (nb, ncl), device=dev, generator=gen)
        w = torch.zeros(nb, ncl, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))
        wr = w[:, cl]; den = wr.sum(1); nan += int((den == 0).sum())
        for k, v in vals.items():
            out[k].append((wr * v[None]).sum(1) / den)
        done += nb
    res = {}
    for k in out:
        m = torch.cat(out[k]); m = m[torch.isfinite(m)]
        res[k] = [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]
    return res, nan


# ---------------- load the three released CSVs (same alignment as G1)
raw = {}
for arm, fn in FILES.items():
    rr = list(csv.DictReader(open(os.path.join(GLIM, fn), encoding='utf-8'))); occ = defaultdict(int); keyed = {}
    for r in rr:
        k = (r['Task'], r['Dataset'], r['Subject'], r['Input Text']); occ[k] += 1
        keyed[k + (occ[k],)] = dict(gen=r['Generated Text'], raw=r['Raw Task Key'], tgt=r['Target Texts'])
    assert len(keyed) == len(rr), (arm, 'duplicate keys')
    raw[arm] = keyed
ARMS = list(FILES)
common = sorted(set.intersection(*[set(raw[k]) for k in ARMS]))
assert all(len(raw[k]) == len(common) for k in ARMS), {k: len(raw[k]) for k in ARMS}
assert len(common) == 2208 or a.smoke, len(common)
for k in common:
    assert len({raw[arm][k]['raw'] for arm in ARMS}) == 1 and len({raw[arm][k]['tgt'] for arm in ARMS}) == 1, k
if a.smoke:
    common = common[::7]
REC = [dict(task=k[0], dataset=k[1], subject=k[2], sent=k[3], raw=raw['eeg'][k]['raw'], cell=k[0] + k[1], tgt=raw['eeg'][k]['tgt'],
            pid=k[1] + ':' + k[2], gen={arm: raw[arm][k]['gen'] for arm in ARMS}) for k in common]
print('records', len(REC), 'arms', ARMS, flush=True)
SEL = {}
for c, rk in sorted({(r['cell'], r['raw']) for r in REC}):
    SEL[f'cell:{c}|{rk}'] = [i for i, r in enumerate(REC) if r['cell'] == c and r['raw'] == rk]

gates = []
def gate(nm, exp, got, tol, informational=False):
    gates.append(dict(name=nm, expected=exp, got=got, tol=tol, abs_diff=abs(got - exp), pass_=bool(abs(got - exp) <= tol), informational=informational))

summary = dict(NOT_CITABLE='new output, no blind review yet', code_sha256=CODE_SHA256, script=os.path.abspath(__file__), smoke=a.smoke, seed=0,
               n_boot=n_boot, lams=ALL_LAMS, grid='card12', scorer='G1 scorer (MiniLM cosine InfoNCE, LOO-sentence temperature, card12 grid), temperature not re-picked inside a resample',
               level='SCORER-LEVEL power: generation embeddings of the released GLIM EEG arm are mixed toward the target embedding; this is not EEG-level power',
               tasks={})
perrec = []
for name in TASKS:
    idx = SEL.get(name, [])
    if len(idx) < 10:
        print('skip', name, len(idx)); continue
    tg = [REC[i]['sent'] for i in idx]; U = list(dict.fromkeys(tg)); pos = {u: j for j, u in enumerate(U)}
    true = torch.tensor([pos[t] for t in tg], device=dev); cl = true.clone(); N = len(U)
    EU = emb(U); Etgt = EU[true]
    Eeeg = emb([REC[i]['gen']['eeg'] or '.' for i in idx]); Enoi = emb([REC[i]['gen']['noise_test_time'] or '.' for i in idx])
    # sentences per task: the candidate (Input Text) and the stimulus target text must be the same string for the mixing target to be the stimulus
    n_tgt_equal = sum(REC[i]['tgt'] == REC[i]['sent'] for i in idx)
    g = torch.Generator(device=dev); g.manual_seed(0)
    perm = torch.randperm(len(idx), device=dev, generator=g)                # shuffled-pair floor (as G1)
    gs = torch.Generator(device=dev); gs.manual_seed(0)
    off = 1 + int(torch.randint(0, N - 1, (1,), device=dev, generator=gs))   # sentence-level cyclic shift, no fixed point, for the shuffled-target null
    Eshuf = EU[(true + off) % N]
    assert off % N != 0 and bool(((true + off) % N != true).all())
    y_noise, _ = infonce(Enoi @ EU.T, true, cl, TAUS)
    pids = list(dict.fromkeys(REC[i]['pid'] for i in idx)); ppos = {p: j for j, p in enumerate(pids)}
    part = torch.tensor([ppos[REC[i]['pid']] for i in idx], device=dev)
    T = dict(n_records=len(idx), n_candidates=N, n_participants=len(pids), n_target_text_equals_input_text=n_tgt_equal, noise_yield=float(y_noise.mean()), curve={}, null_shuffled_target={})
    y0 = None
    for lam in ALL_LAMS:
        for kind, Etar in (('curve', Etgt), ('null_shuffled_target', Eshuf)):
            if kind == 'null_shuffled_target' and lam == 1.0:
                continue
            Em = torch.nn.functional.normalize((1 - lam) * Eeeg + lam * Etar, dim=1)
            y, tix = infonce(Em @ EU.T, true, cl, TAUS)
            sh, _ = infonce((Em @ EU.T)[perm], true, cl, TAUS)
            if kind == 'curve' and lam == 0.0:
                y0 = y.clone()
            d = y - y_noise
            ci, nan = boot_sentence({'diff': d, 'yield': y, 'gain_vs_lam0': y - y0}, cl)
            pm = torch.zeros(len(pids), device=dev).index_add_(0, part, d) / torch.bincount(part, minlength=len(pids)).float()
            T[kind][f'{lam:g}'] = dict(lam=lam, yield_bits=float(y.mean()), diff_eeg_minus_noise=float(d.mean()), diff_ci95_sentence=ci['diff'],
                                       diff_lower_bound_above_0=bool(ci['diff'][0] > 0), yield_ci95_sentence=ci['yield'],
                                       gain_vs_lam0=float((y - y0).mean()), gain_vs_lam0_ci95_sentence=ci['gain_vs_lam0'],
                                       floor_shuffled_pair_bits=float(sh.mean()), n_participants_diff_positive=f'{int((pm > 0).sum())} of {len(pids)}',
                                       frac_records_tau_at_upper_edge=float((tix == len(TAUS) - 1).float().mean()), ci_nan_draws=nan)
            if kind == 'curve':
                for j, i in enumerate(idx):
                    perrec.append([name, f'{lam:g}', i, REC[i]['pid'], f'{y[j]:.6f}', f'{y_noise[j]:.6f}'])
    c = T['curve']
    print(name, f"N={N} n={len(idx)} noise {T['noise_yield']:+.4f}")
    for lam in ALL_LAMS:
        r = c[f'{lam:g}']
        print(f"  lam={lam:<5g} yield {r['yield_bits']:+.4f} diff {r['diff_eeg_minus_noise']:+.4f} {[round(x, 4) for x in r['diff_ci95_sentence']]} floor {r['floor_shuffled_pair_bits']:+.4f} "
              f"null_shuf diff {T['null_shuffled_target'].get(f'{lam:g}', {}).get('diff_eeg_minus_noise', float('nan')):+.4f}", flush=True)
    summary['tasks'][name] = T
    # ---- gates
    key = name.split('|')[1]
    sv = SAVED[name]['grids']['card12']
    gate(f'{key} lam=0 eeg-noise vs saved G1 summary', sv['diffs']['eeg_minus_noise_test_time']['mean'], c['0']['diff_eeg_minus_noise'], 1e-4)
    gate(f'{key} lam=0 eeg yield vs saved G1 summary', sv['arms']['eeg']['yield_bits'], c['0']['yield_bits'], 1e-4)
    gate(f'{key} noise yield vs saved G1 summary', sv['arms']['noise_test_time']['yield_bits'], T['noise_yield'], 1e-4)
    for j, side in enumerate(('lo', 'hi')):
        gate(f'{key} lam=0 sentence CI {side} vs saved G1 summary', sv['ci95']['sentence']['diff:eeg_minus_noise_test_time'][j], c['0']['diff_ci95_sentence'][j], 1e-4)
    gate(f'{key} lam=0 diff vs rounded protocol value', {'task1': 0.0072, 'task2': -0.0199}[key], c['0']['diff_eeg_minus_noise'], 1e-4)
    gate(f'{key} lam=1 yield = log2 N (N={N})', math.log2(N), c['1']['yield_bits'], 1e-3)
    gate(f'{key} lam=1 yield = log2 {({"task1": 40, "task2": 30})[key]}', math.log2({'task1': 40, 'task2': 30}[key]), c['1']['yield_bits'], 1e-3)
    ys = [c[f'{l:g}']['yield_bits'] for l in ALL_LAMS]
    summary.setdefault('monotone_check_informational', {})[key] = dict(yields=ys, monotone_nondecreasing=bool(all(y2 >= y1 - 1e-12 for y1, y2 in zip(ys, ys[1:]))))

# ---------------- decision, as pre-registered
dec = {}
if all(t in summary['tasks'] for t in TASKS):
    for lam in LAMS:
        k = f'{lam:g}'
        dec[k] = dict(both_lower_bounds_above_0=all(summary['tasks'][t]['curve'][k]['diff_lower_bound_above_0'] for t in TASKS),
                      diff_by_task={t.split('|')[1]: summary['tasks'][t]['curve'][k]['diff_eeg_minus_noise'] for t in TASKS},
                      lower_by_task={t.split('|')[1]: summary['tasks'][t]['curve'][k]['diff_ci95_sentence'][0] for t in TASKS})
    det = [lam for lam in LAMS if dec[f'{lam:g}']['both_lower_bounds_above_0']]
    lam_star = det[0] if det else None
    mde = max(dec[f'{lam_star:g}']['diff_by_task'].values()) if det else None
    nullbad = [lam for lam in LAMS if all(summary['tasks'][t]['null_shuffled_target'][f'{lam:g}']['diff_lower_bound_above_0'] for t in TASKS)]
    summary['decision'] = dict(per_lam=dec, lam_star=lam_star, mde_bits_max_over_tasks=mde, threshold_bits=0.10,
                               PASS_prereg=bool(lam_star is not None and mde <= 0.10), null_shuffled_target_detected_at=nullbad, null_ok=len(nullbad) == 0,
                               secondary_gain_vs_lam0_at_lam_star={t.split('|')[1]: summary['tasks'][t]['curve'][f'{lam_star:g}']['gain_vs_lam0'] for t in TASKS} if det else None)
    print('DECISION', json.dumps({k: v for k, v in summary['decision'].items() if k != 'per_lam'}), flush=True)
summary['gates'] = gates
summary['gate_all_pass'] = all(g['pass_'] for g in gates) if gates else None
summary['gate_failures'] = [g for g in gates if not g['pass_']]
for g in gates:
    print('GATE', 'PASS' if g['pass_'] else 'FAIL', g['name'], 'exp', round(g['expected'], 5), 'got', round(g['got'], 5), 'tol', g['tol'], flush=True)
summary['runtime_s'] = time.time() - t0
json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
with open(os.path.join(a.out, 'per_record.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['task', 'lam', 'record', 'participant', 'yield_eeg_mixed', 'yield_noise']); w.writerows(perrec)
print('wrote', a.out, f'{time.time()-t0:.0f}s', flush=True)
