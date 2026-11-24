"""Cache word-level GD features from Wang & Ji pickles into one compact pickle for BrainTranslator training.
Reuses load_task from pilot_retrieval.py. Split = Wang & Ji (first 80/10/10 sentence indices per task), duplicates across tasks dropped
the same way pilot_retrieval.py does."""
import os, sys, pickle, time, argparse
import numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import load_task, TASKS

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/path/to/eegaudit/data_raw/zuco/eeg2text_pickles')
ap.add_argument('--out', required=True)
ap.add_argument('--tasks', default='SR,NR,NR2')
ap.add_argument('--max_subjects', type=int, default=None)
ap.add_argument('--keep_dups', action='store_true', help='keep cross-task duplicate sentences (the ORIGINAL Wang-Ji reference split, leaky; for the G2 reproduction only)')
a = ap.parse_args()
t0 = time.time()
all_texts, recs, split, task_of = [], [], {}, {}
for t in a.tasks.split(','):
    texts, sp, rc, subjects = load_task(os.path.join(a.data, TASKS[t]), t, max_subjects=a.max_subjects)
    off = len(all_texts); idx_map = {}
    for i, tx in enumerate(texts):
        if tx is not None:
            idx_map[i] = len(all_texts); all_texts.append(tx); split[idx_map[i]] = sp[i]; task_of[idx_map[i]] = t
    for (sj, i), (wv, sv) in rc.items():
        recs.append({'subject': t + ':' + sj, 'g': idx_map[i], 'words': wv.astype(np.float32)})
    print(f'{t}: {len(idx_map)} sentences, {len(subjects)} subjects, {len(rc)} records ({time.time()-t0:.0f}s)', flush=True)
first = {}
for g, tx in enumerate(all_texts):
    first.setdefault(tx, g)
for g in range(len(all_texts)):
    if first[all_texts[g]] != g and not a.keep_dups:
        split[g] = 'drop'
for r in recs:
    r['split'] = split[r['g']]; r['task'] = task_of[r['g']]
recs = [r for r in recs if r['split'] != 'drop']
n_w = [r['words'].shape[0] for r in recs]
print('records kept', len(recs), 'mean words', np.mean(n_w), 'max words', max(n_w), flush=True)
for s in ['train', 'dev', 'test']:
    print(s, 'records', sum(r['split'] == s for r in recs), 'sentences', len({r['g'] for r in recs if r['split'] == s}))
pickle.dump({'texts': all_texts, 'recs': recs}, open(a.out, 'wb'), protocol=4)
print('wrote', a.out, os.path.getsize(a.out) / 1e6, 'MB in', time.time() - t0, 's')
