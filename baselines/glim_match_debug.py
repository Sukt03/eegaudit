"""Debug: how our cache sentences match GLIM's label table (dataset, task, normalised text). CPU only, label table loaded after the global scan."""
import pickle, re, sys
sys.argv = [sys.argv[0]]
exec(open('/path/to/eegaudit/experiments/baselines/glim_split_audit.py').read().split("import pandas as pd")[0])  # global scan
import pandas as pd
df = pd.read_pickle('/path/to/eegaudit/data_raw/third_party/glim_repo/zuco_label_8variants.df')
print(df.groupby(['dataset', 'task']).size())
norm = lambda t: ' '.join(re.findall(r"[a-z0-9]+", str(t).lower()))
D = pickle.load(open('/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl', 'rb'))
texts = D['texts']; blocks = {'SR': (0, 400), 'NR': (400, 700), 'NR2': (700, 1049)}
for col in ('input text', 'raw text'):
    keys = {}
    for _, r in df.iterrows():
        keys.setdefault(norm(r[col]), set()).add((r['dataset'], r['task']))
    for t, (s, e) in blocks.items():
        hit = [keys.get(norm(texts[g])) for g in range(s, e)]
        print(col, t, 'matched', sum(h is not None for h in hit), 'of', e - s, 'where', pd.Series([str(sorted(h)) for h in hit if h]).value_counts().head(3).to_dict())
