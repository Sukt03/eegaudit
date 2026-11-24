"""G1 glim_within_task (RUNNER, 24 Nov 2026).
Scores the RELEASED GLIM generations (EEG input, test-time-noise input, prompt-free EEG input) with the scorer of
experiments/baselines/text_yield_card.py (score_set / infonce copied: MiniLM cosine InfoNCE log-score in bits, softmax over the unique
labelled test sentences of the scored set, temperature picked per sentence on the OTHER sentences' records = leave-one-sentence-out),
on: pooled (2208 records / 182 sentences / 30 participants), the four corpus cells (Task prompt x Dataset), and every cell x Raw Task Key
group (the two ZuCo1 <NR>-prompt reading tasks task1 = sentiment/movie reviews and task2 = Wikipedia/normal reading are the required ones).
Grids: card12 = logspace(-2.5,0,12) (the card's grid, used for the GATE), ext100 = logspace(-2.5,2,21) (primary new read-out, tau up to 100),
ext10 = logspace(-2.5,1,16) (point estimates only). Oracles: genre / task-identity labels alone with uniform choice inside the group
(bits per record = mean log2(N / n_candidates_in_group)). Intervals: sentence-cluster, participant (Dataset+Subject) and crossed
sentence x participant bootstraps, 10000 resamples, torch on the GPU, seed 0 (per selection and grid); the temperature is NOT re-picked
inside a resample (as in the card). Shuffled-pair floors are reported, never subtracted.
Usage: python -I -u glim_within_task.py --out results/ext/glim_within_task [--smoke]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, csv, json, math, os, re, time
from collections import Counter, defaultdict
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/path/to/eegaudit/results/ext/glim_within_task')
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
CARD_JSON = '/path/to/eegaudit/results/baselines/text_card_v2/text_yield_card.json'
FILES = {'eeg': 'wandb_export_gen_samples_glim.csv', 'noise_test_time': 'wandb_export_gen_samples_noise_input_test.csv',
         'prompt_free': 'wandb_export_gen_samples_prompt_free_test.csv'}
GRIDS = {'card12': torch.logspace(-2.5, 0, 12, device=dev), 'ext100': torch.logspace(-2.5, 2, 21, device=dev),
         'ext10': torch.logspace(-2.5, 1, 16, device=dev)}
BOOT_GRIDS = ('card12', 'ext100')
n_boot = 300 if a.smoke else a.n_boot

# ---------------- MiniLM (same model/pooling as the card)
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
    """identical maths to text_yield_card.infonce; also returns the chosen tau index per record."""
    Z = S[None] / taus[:, None, None]
    lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, torch.arange(len(true), device=dev), true] / LOG2
    ncl = int(cl.max()) + 1
    csum = torch.zeros(len(taus), ncl, device=dev).index_add_(1, cl, CE); ccnt = torch.bincount(cl, minlength=ncl).float()
    loo = (CE.sum(1, keepdim=True) - csum) / (len(true) - ccnt).clamp(min=1)
    t_idx = loo.argmin(0)[cl]
    return math.log2(S.shape[1]) - CE[t_idx, torch.arange(len(true), device=dev)], t_idx


def boot_ci(vals, cl, part, kind, gen):
    """vals name -> (Q,) ; weighted-mean bootstrap. kind sentence | participant | crossed. Returns name -> [2.5, 97.5] and the n of NaN draws."""
    ncl = int(cl.max()) + 1; npt = int(part.max()) + 1
    out = {k: [] for k in vals}; nan = 0; done = 0
    while done < n_boot:
        nb = min(2500, n_boot - done)
        wr = None
        if kind in ('sentence', 'crossed'):
            pick = torch.randint(0, ncl, (nb, ncl), device=dev, generator=gen)
            w = torch.zeros(nb, ncl, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))
            wr = w[:, cl]
        if kind in ('participant', 'crossed'):
            pick = torch.randint(0, npt, (nb, npt), device=dev, generator=gen)
            w = torch.zeros(nb, npt, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))
            wr = w[:, part] if wr is None else wr * w[:, part]
        den = wr.sum(1)
        nan += int((den == 0).sum())
        for k, v in vals.items():
            out[k].append((wr * v[None]).sum(1) / den)
        done += nb
    res = {}
    for k in out:
        m = torch.cat(out[k]); m = m[torch.isfinite(m)]
        res[k] = [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]
    return res, nan


# ---------------- load the three released CSVs, aligned on (Task, Dataset, Subject, Input Text, occurrence)
raw = {}
for arm, fn in FILES.items():
    p = os.path.join(GLIM, fn)
    if not os.path.exists(p):
        print('MISSING', p); continue
    rr = list(csv.DictReader(open(p, encoding='utf-8'))); occ = defaultdict(int); keyed = {}
    for r in rr:
        k = (r['Task'], r['Dataset'], r['Subject'], r['Input Text']); occ[k] += 1
        keyed[k + (occ[k],)] = dict(gen=r['Generated Text'], raw=r['Raw Task Key'], tgt=r['Target Texts'])
    assert len(keyed) == len(rr), (arm, 'duplicate keys')
    raw[arm] = keyed
ARMS = [k for k in FILES if k in raw]
common = sorted(set.intersection(*[set(raw[k]) for k in ARMS]))
assert all(len(raw[k]) == len(common) for k in ARMS), {k: len(raw[k]) for k in ARMS}
assert len(common) == 2208 or a.smoke, len(common)
for k in common:
    assert len({raw[arm][k]['raw'] for arm in ARMS}) == 1 and len({raw[arm][k]['tgt'] for arm in ARMS}) == 1, k
if a.smoke:
    common = common[::7]
REC = [dict(task=k[0], dataset=k[1], subject=k[2], sent=k[3], raw=raw['eeg'][k]['raw'], cell=k[0] + k[1],
            pid=k[1] + ':' + k[2], gen={arm: raw[arm][k]['gen'] for arm in ARMS}) for k in common]
print('records', len(REC), 'sentences', len({r['sent'] for r in REC}), 'participants', len({r['pid'] for r in REC}), 'arms', ARMS,
      'cells', dict(Counter(r['cell'] for r in REC)), 'cell x raw key', dict(Counter((r['cell'], r['raw']) for r in REC)), flush=True)

# ---------------- selections
SEL = {'pooled': list(range(len(REC)))}
for c in sorted({r['cell'] for r in REC}):
    SEL[f'cell:{c}'] = [i for i, r in enumerate(REC) if r['cell'] == c]
for c, rk in sorted({(r['cell'], r['raw']) for r in REC}):
    SEL[f'cell:{c}|{rk}'] = [i for i, r in enumerate(REC) if r['cell'] == c and r['raw'] == rk]
REQUIRED_WITHIN = ['cell:<NR>ZuCo1|task1', 'cell:<NR>ZuCo1|task2']


def oracle_bits(idx):
    """bits per record of knowing only a group label (uniform choice among that group's sentences); partitions: cell, raw task key, cell+raw key."""
    sents = [REC[i]['sent'] for i in idx]; N = len(set(sents)); out = {}
    for name, f in (('cell', lambda r: r['cell']), ('raw_task_key_genre', lambda r: r['raw']), ('cell_and_raw_task_key', lambda r: (r['cell'], r['raw']))):
        grp = [f(REC[i]) for i in idx]; ng = len(set(grp))
        if ng < 2:
            continue
        nsent = {g: len({s for s, gg in zip(sents, grp) if gg == g}) for g in set(grp)}
        # sentences are unique to a group (checked below), so the group's candidate count = its distinct sentences
        assert sum(nsent.values()) == N, (name, 'sentence spans several groups')
        out[name] = dict(bits_per_record=float(np.mean([math.log2(N / nsent[g]) for g in grp])), n_groups=ng)
    return out


def score_selection(name, idx, planted=None):
    """returns (result dict, per-record rows). planted: None | fn(record, arm) -> hypothesis text (positive control)."""
    tg = [REC[i]['sent'] for i in idx]; U = list(dict.fromkeys(tg)); pos = {u: j for j, u in enumerate(U)}
    true = torch.tensor([pos[t] for t in tg], device=dev); cl = true.clone(); N = len(U)
    pids = list(dict.fromkeys(REC[i]['pid'] for i in idx)); ppos = {p: j for j, p in enumerate(pids)}
    part = torch.tensor([ppos[REC[i]['pid']] for i in idx], device=dev)
    EU = emb(U)
    arms = ARMS if planted is None else ['planted']
    S = {}
    for arm in arms:
        hyp = [(planted(REC[i]) if planted else REC[i]['gen'][arm]) or '.' for i in idx]
        S[arm] = emb(hyp) @ EU.T
    g = torch.Generator(device=dev); g.manual_seed(0)
    perm = torch.randperm(len(idx), device=dev, generator=g)   # one shuffled-pair permutation shared by all arms (floor of the paired difference is defined)
    res = dict(n_records=len(idx), n_candidates=N, n_participants=len(pids), chance_top1=1 / N, oracle=oracle_bits(idx), grids={})
    rows = {}
    for gn, taus in GRIDS.items():
        y, sh, tix = {}, {}, {}
        for arm in arms:
            y[arm], tix[arm] = infonce(S[arm], true, cl, taus); sh[arm], _ = infonce(S[arm][perm], true, cl, taus)
        G = dict(tau_grid=[float(t) for t in taus], arms={}, diffs={})
        for arm in arms:
            G['arms'][arm] = dict(yield_bits=float(y[arm].mean()), floor_shuffled_bits=float(sh[arm].mean()),
                                  frac_records_at_grid_upper_edge=float((tix[arm] == len(taus) - 1).float().mean()),
                                  chosen_tau_median=float(taus[tix[arm]].median()))
        dpairs = [('eeg', 'noise_test_time'), ('eeg', 'prompt_free'), ('prompt_free', 'noise_test_time')] if planted is None else []
        dpairs = [(x, z) for x, z in dpairs if x in y and z in y]
        d = {f'{x}_minus_{z}': (y[x] - y[z]) for x, z in dpairs}
        dsh = {f'{x}_minus_{z}': (sh[x] - sh[z]) for x, z in dpairs}
        for k in d:
            G['diffs'][k] = dict(mean=float(d[k].mean()), floor_shuffled_mean=float(dsh[k].mean()), n_participants_positive=None)
        if 'eeg_minus_noise_test_time' in d:
            pm = torch.zeros(len(pids), device=dev).index_add_(0, part, d['eeg_minus_noise_test_time']) / torch.bincount(part, minlength=len(pids)).float()
            G['diffs']['eeg_minus_noise_test_time']['n_participants_positive'] = f'{int((pm > 0).sum())} of {len(pids)}'
        if gn in BOOT_GRIDS and planted is None:
            vals = {**{f'yield:{k}': v for k, v in y.items()}, **{f'floor:{k}': v for k, v in sh.items()}, **{f'diff:{k}': v for k, v in d.items()},
                    **{f'diff_floor:{k}': v for k, v in dsh.items()}}
            for kind in ('sentence', 'participant', 'crossed'):
                gen = torch.Generator(device=dev); gen.manual_seed(0)
                ci, nan = boot_ci(vals, cl, part, kind, gen)
                G.setdefault('ci95', {})[kind] = {k: v for k, v in ci.items()}; G.setdefault('ci_nan_draws', {})[kind] = nan
        res['grids'][gn] = G
        rows[gn] = {**{f'y_{k}': v.cpu().numpy() for k, v in y.items()}, **{f'floor_{k}': v.cpu().numpy() for k, v in sh.items()},
                    **{f'tau_{k}': taus[t].cpu().numpy() for k, t in tix.items()}}
    return res, rows, idx


summary = dict(NOT_CITABLE='new output, no blind review yet', code_sha256=CODE_SHA256, script=os.path.abspath(__file__), smoke=a.smoke, seed=0, n_boot=n_boot,
               inputs={k: os.path.join(GLIM, v) for k, v in FILES.items()}, n_records=len(REC), n_sentences=len({r['sent'] for r in REC}),
               n_participants=len({r['pid'] for r in REC}), arms=ARMS, grids={k: [float(t) for t in v] for k, v in GRIDS.items()},
               primary_grid_for_gate='card12', primary_grid_new_readout='ext100',
               scorer='copy of experiments/baselines/text_yield_card.py score_set/infonce (MiniLM cosine, InfoNCE bits, LOO-sentence temperature); scored text = Generated Text (free running); candidate = Input Text',
               note_bootstrap='temperature not re-picked inside a resample; participant = Dataset:Subject; crossed = product of independent sentence and participant multiplicities',
               selections={})
perrec = []
for name, idx in SEL.items():
    if len(idx) < 10:
        continue
    res, rows, idx = score_selection(name, idx)
    summary['selections'][name] = res
    for gn, R in rows.items():
        for j, i in enumerate(idx):
            perrec.append([name, gn, i, REC[i]['cell'], REC[i]['raw'], REC[i]['pid'], REC[i]['sent'][:60].replace(',', ' ').replace('"', ''),
                           *[f'{R[k][j]:.6f}' for k in sorted(R)]])
    perrec_cols = ['selection', 'grid', 'record', 'cell', 'raw_task_key', 'participant', 'sentence60', *sorted(R)]
    g12, g100 = res['grids']['card12'], res['grids']['ext100']
    print(f"{name:28s} n={res['n_records']:4d} N={res['n_candidates']:3d} | card12 eeg {g12['arms']['eeg']['yield_bits']:+.4f} noise {g12['arms']['noise_test_time']['yield_bits']:+.4f} "
          f"d {g12['diffs']['eeg_minus_noise_test_time']['mean']:+.4f} {g12['ci95']['sentence']['diff:eeg_minus_noise_test_time']} | ext100 d {g100['diffs']['eeg_minus_noise_test_time']['mean']:+.4f} "
          f"{g100['ci95']['sentence']['diff:eeg_minus_noise_test_time']} ({time.time()-t0:.0f}s)", flush=True)

# ---------------- planted controls (positive: generations that carry the information; genre-only: generations that carry ONLY the genre)
G1 = lambda r: 'This movie is a good film about love and life.' if r['raw'] == 'task1' else 'He was born in 1950 in the United States and served on the council.'
CTRL = {}
for nm, fn in (('planted_target_text_as_generation', lambda r: r['sent']), ('planted_genre_only_constant_text', G1)):
    CTRL[nm] = {}
    for sel in ['pooled', 'cell:<NR>ZuCo1', *REQUIRED_WITHIN]:
        if sel in SEL and len(SEL[sel]) >= 10:
            r_, _, _ = score_selection(sel, SEL[sel], planted=fn)
            CTRL[nm][sel] = {gn: r_['grids'][gn]['arms']['planted']['yield_bits'] for gn in GRIDS} | {'oracle': r_['oracle'], 'n_candidates': r_['n_candidates']}
    print('CONTROL', nm, json.dumps({k: {g: round(v[g], 4) for g in GRIDS} for k, v in CTRL[nm].items()}), flush=True)
summary['planted_controls'] = CTRL

# ---------------- gates
gates = []
def gate(nm, exp, got, tol):
    gates.append(dict(name=nm, expected=exp, got=got, tol=tol, abs_diff=abs(got - exp), pass_=bool(abs(got - exp) <= tol)))
def S_(sel, grid='card12'): return summary['selections'][sel]['grids'][grid]
if not a.smoke:
    D = 'eeg_minus_noise_test_time'
    # (1) saved card JSON, 1e-4 on the means (same grid, same scorer)
    card = json.load(open(CARD_JSON))['models']['glim_released']['yield']['free_running']
    for sel, ck in (('pooled', 'pooled'), ('cell:<NR>ZuCo1', 'within_<NR>ZuCo1'), ('cell:<NR>ZuCo2', 'within_<NR>ZuCo2'), ('cell:<TSR>ZuCo1', 'within_<TSR>ZuCo1'), ('cell:<TSR>ZuCo2', 'within_<TSR>ZuCo2')):
        c = card[ck]
        gate(f'card_json {sel} eeg yield', c['arms']['eeg']['yield_bits'], S_(sel)['arms']['eeg']['yield_bits'], 1e-4)
        gate(f'card_json {sel} noise yield', c['arms']['noise_test_time']['yield_bits'], S_(sel)['arms']['noise_test_time']['yield_bits'], 1e-4)
        gate(f'card_json {sel} eeg-noise', c['eeg_minus_noise_test_time']['mean'], S_(sel)['diffs'][D]['mean'], 1e-4)
    # (2) reviewer table, 0.005 bit
    for sel, e in (('pooled', 0.2436), ('cell:<NR>ZuCo1', 0.3151), ('cell:<NR>ZuCo2', -0.010), ('cell:<TSR>ZuCo1', 0.0018), ('cell:<TSR>ZuCo2', 0.0027),
                   ('cell:<NR>ZuCo1|task1', 0.007), ('cell:<NR>ZuCo1|task2', -0.020)):
        gate(f'reviewer {sel} eeg-noise (card12)', e, S_(sel)['diffs'][D]['mean'], 0.005)
    for sel, e in (('pooled', [0.140, 0.346]), ('cell:<NR>ZuCo1', [0.138, 0.502])):   # card CI, 3 decimals, 5000 resamples vs 10000 here
        c = S_(sel)['ci95']['sentence']['diff:' + D]
        gate(f'reviewer {sel} sentence CI lo', e[0], c[0], 0.005); gate(f'reviewer {sel} sentence CI hi', e[1], c[1], 0.005)
    for sel, e in (('cell:<NR>ZuCo1|task1', [-0.05, 0.07]), ('cell:<NR>ZuCo1|task2', [-0.06, 0.02])):   # reviewer CI rounded to 2 decimals
        c = S_(sel)['ci95']['sentence']['diff:' + D]
        gate(f'reviewer {sel} sentence CI lo (2-dec rounded)', e[0], c[0], 0.01); gate(f'reviewer {sel} sentence CI hi (2-dec rounded)', e[1], c[1], 0.01)
    gate('reviewer ZuCo1 <NR> genre oracle', 0.970, summary['selections']['cell:<NR>ZuCo1']['oracle']['raw_task_key_genre']['bits_per_record'], 0.005)
    gate('reviewer ZuCo1 <NR> prompt-free yield', 0.258, S_('cell:<NR>ZuCo1')['arms']['prompt_free']['yield_bits'], 0.005)
    gate('reviewer ZuCo1 <NR> EEG yield', 0.294, S_('cell:<NR>ZuCo1')['arms']['eeg']['yield_bits'], 0.005)
    gate('reviewer pooled prompt-free yield', 0.263, S_('pooled')['arms']['prompt_free']['yield_bits'], 0.005)
    gate('reviewer pooled EEG yield', 0.250, S_('pooled')['arms']['eeg']['yield_bits'], 0.005)
    # (3) reviewer extended-grid sensitivities (tau 10 / 100), reported only
    for sel, e in (('pooled', 0.244), ('cell:<NR>ZuCo1', 0.297), ('cell:<NR>ZuCo2', -0.019), ('cell:<TSR>ZuCo1', 0.008), ('cell:<TSR>ZuCo2', -0.011)):
        gate(f'reviewer {sel} eeg-noise (tau 10 grid, informational)', e, S_(sel, 'ext10')['diffs'][D]['mean'], 0.005)
    for sel, e in (('pooled', 0.248), ('cell:<NR>ZuCo1', None), ('cell:<NR>ZuCo2', -0.017), ('cell:<TSR>ZuCo1', 0.006), ('cell:<TSR>ZuCo2', -0.009)):
        if e is not None:
            gate(f'reviewer {sel} eeg-noise (tau 100 grid, informational)', e, S_(sel, 'ext100')['diffs'][D]['mean'], 0.005)
summary['gates'] = gates
summary['gate_all_pass_core'] = all(g['pass_'] for g in gates if 'informational' not in g['name']) if gates else None
summary['gate_failures'] = [g for g in gates if not g['pass_']]
for g in gates:
    print('GATE', 'PASS' if g['pass_'] else 'FAIL', g['name'], 'exp', g['expected'], 'got', round(g['got'], 5), 'tol', g['tol'], flush=True)

summary['runtime_s'] = time.time() - t0
sfn = os.path.join(a.out, 'summary.json'); json.dump(summary, open(sfn, 'w'), indent=1)
with open(os.path.join(a.out, 'per_record.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(perrec_cols); w.writerows(perrec)
print('wrote', sfn, f'{time.time()-t0:.0f}s', flush=True)
