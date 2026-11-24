"""Condition 2 (METHOD): which test sentences of GLIM's released generations also occur in GLIM's train/dev split?
Reads GLIM's label table (data/tmp/zuco_label_8variants.df, pandas pickle) only after scanning every pickle global and accepting only
pandas/numpy/builtins-container globals. Writes results/baselines/text/glim_split_audit.json (leaked test texts, counts)."""
import io, json, pickletools, sys
ALLOW = ('pandas.', 'numpy.', 'builtins.slice', 'builtins.list', 'builtins.dict', 'builtins.tuple', 'builtins.object', 'copyreg._reconstructor', 'datetime.', 'collections.OrderedDict', '_codecs.encode')
p = '/path/to/eegaudit/data_raw/third_party/glim_repo/zuco_label_8variants.df'
data = open(p, 'rb').read(); found = set(); pushed = []; memo = {}
for op, arg, _ in pickletools.genops(io.BytesIO(data)):
    if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'): pushed.append(arg)
    elif op.name == 'MEMOIZE': memo[len(memo)] = pushed[-1] if pushed else None
    elif op.name in ('BINPUT', 'LONG_BINPUT', 'PUT'): memo[arg] = pushed[-1] if pushed else None
    elif op.name in ('BINGET', 'LONG_BINGET', 'GET'): pushed.append(memo.get(arg))
    elif op.name == 'GLOBAL': found.add(arg.replace(' ', '.')); pushed.append(None)
    elif op.name == 'STACK_GLOBAL': found.add(f'{pushed[-2]}.{pushed[-1]}'); pushed.append(None)
    else: pushed.append(None)
bad = [g for g in found if not g.startswith(ALLOW)]
print('globals', sorted(found)); assert not bad, f'refusing: {bad}'
import pandas as pd
df = pd.read_pickle(p)
import csv, re
norm = lambda t: ' '.join(re.findall(r"[a-z0-9]+", str(t).lower()))
by = {}
for t, u in zip(df['input text'], df['text uid']):
    by.setdefault(norm(t), set()).add(int(u))
rel = list(csv.DictReader(open('/path/to/eegaudit/data_raw/third_party/glim_results/wandb_export_gen_samples_glim.csv', encoding='utf-8')))
test_texts = sorted({r['Input Text'] for r in rel})
uid_of = {}
for t, u in zip(df['input text'], df['text uid']):
    uid_of.setdefault(t, set()).add(int(u))
leaked = [t for t in test_texts if len(by.get(norm(t), set())) > 1]
dup_uid_counts = df['text uid'].value_counts()
out = {'n_label_rows': len(df), 'n_unique_uids': int(df['text uid'].nunique()), 'n_uids_shared_across_rows': int((dup_uid_counts > 1).sum()),
       'n_released_test_texts': len(test_texts), 'n_test_texts_with_same_normalised_text_under_another_uid': len(leaked),
       'leaked_test_texts': leaked,
       'rule': 'GLIM STEP4: split by text uid; uids shared by >1 label row go to train. A released test sentence is flagged if its normalised text also occurs under a DIFFERENT text uid (that copy may sit in train/val). Conservative: flagged texts are excluded in the re-score.'}
json.dump(out, open('/path/to/eegaudit/results/baselines/text/glim_split_audit.json', 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'leaked_test_texts'})
