"""Aggregate results/track_text/*.json (PROTOCOL schema) into a markdown price table: mean +- 95% CI over draws
(m=1: subjects; m=3: random triples x seeds; m=all: seeds). Also writes summary.json for ledger registration."""
import json, glob, os, sys
import numpy as np
res_dir = sys.argv[1]; out_md = sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(res_dir, '*.json'))):
    if os.path.basename(f).startswith(('meta', 'reading_times', 'summary')):
        continue
    d = json.load(open(f))
    for r in d['rows']:
        r = dict(r); r['decoder'] = d['decoder']; r['seed'] = d['seed']; r['N'] = d['n_candidates']; rows.append(r)
meta = json.load(open(os.path.join(res_dir, 'meta.json')))
N = meta['n_candidates']
def ci(x):
    x = np.asarray(x, float); m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    return m, h
order_k = {1: 0, 2: 1, 4: 2, 8: 3, 'all': 4}
summary = {}
counts = meta.get('test_counts_per_task', {})
if len(counts) > 1:
    tot = sum(counts.values())
    # 2-way score of a decoder that knows only which task/dataset a query comes from: beats every other-task candidate, ties within task
    dom = sum(c * ((tot - c) / (tot - 1) + 0.5 * (c - 1) / (tot - 1)) for c in counts.values()) / tot
else:
    dom = None
lines = [f'# TRACK text price table  (N = {N} candidate sentences, chance top-1 = {1/N:.3f}, log2 N = {np.log2(N):.2f} bits)', '',
         f"Sentence encoder: {meta['sentence_encoder']}. EEG feature: {meta['eeg_feature']}.", f"Split: {meta['split']}.",
         (f"Test pool per task: {counts}; 2-way score of a dataset-only (domain-only) decoder = {dom:.3f}; chance = 0.500." if dom else 'Single-task pool: no dataset cue available.'),
         f"Reading-time constants: mean sentence on-screen {meta['reading_times'].get('mean_sentence_duration_s')} s, mean gaze duration/word {meta['reading_times'].get('mean_GD_s')} s.", '',
         '| decoder | words | subjects avg (m) | condition | n draws | top-1 | 2-way | bits/query | bits/s | norm. eff. |', '|---|---|---|---|---|---|---|---|---|---|']
groups = {}
for r in rows:
    groups.setdefault((r['decoder'], r['words'], r['subjects_averaged'], r['condition']), []).append(r)
def skey(k):
    dec, w, m, c = k
    mode = dec.split('_')[1]
    return ({'sentlevel': 2, 'mean': 0, 'concat': 1}[mode], order_k[w], {1: 0, 3: 1, 'all': 2}[m], {'eeg': 0, 'noise': 1, 'trial_shuffled': 2, 'shuffled_pair': 3}[c])
for k in sorted(groups, key=skey):
    g = groups[k]
    t1 = ci([r['top1'] for r in g]); tw = ci([r['two_way'] for r in g]); b = ci([r['bits'] for r in g]); ne = ci([r['norm_eff'] for r in g])
    bps = [r['bits_per_second'] for r in g if r['bits_per_second'] is not None]
    bps_s = f'{np.mean(bps):.4f}' if bps else '--'
    lines.append(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(g)} | {t1[0]:.3f} ± {t1[1]:.3f} | {tw[0]:.3f} ± {tw[1]:.3f} | {b[0]:.3f} ± {b[1]:.3f} | {bps_s} | {ne[0]:.3f} |')
    summary['|'.join(map(str, k))] = {'top1': t1[0], 'top1_ci': t1[1], 'two_way': tw[0], 'two_way_ci': tw[1], 'bits': b[0], 'bits_ci': b[1],
                                      'bits_per_second': float(np.mean(bps)) if bps else None, 'norm_eff': ne[0], 'n': len(g)}
# EEG minus controls, per decoder x m (headline deltas)
lines += ['', '## EEG minus noise control (two-way accuracy, mean over draws; positive = EEG carries sentence information)', '',
          '| decoder | words | m | 2-way EEG | 2-way noise | 2-way shuffled pair | delta vs noise |', '|---|---|---|---|---|---|---|']
for k in sorted(groups, key=skey):
    if k[3] != 'eeg':
        continue
    e = np.mean([r['two_way'] for r in groups[k]]); n_ = groups.get((k[0], k[1], k[2], 'noise')); s_ = groups.get((k[0], k[1], k[2], 'shuffled_pair'))
    nv = np.mean([r['two_way'] for r in n_]) if n_ else float('nan'); sv = np.mean([r['two_way'] for r in s_]) if s_ else float('nan')
    lines.append(f'| {k[0]} | {k[1]} | {k[2]} | {e:.3f} | {nv:.3f} | {sv:.3f} | {e-nv:+.3f} |')
    summary['|'.join(map(str, k))]['two_way_minus_noise'] = e - nv
open(out_md, 'w').write('\n'.join(lines) + '\n'); json.dump(summary, open(os.path.join(res_dir, 'summary.json'), 'w'), indent=1)
print('\n'.join(lines))
