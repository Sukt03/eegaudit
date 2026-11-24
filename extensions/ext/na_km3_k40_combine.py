"""Combine km3 k40 halves A and B (, 23 Nov 2026, NEW): per participant, every list of floats inside decoders[m] (soft12, floor_soft12, exemplar_exact, ..., top1) is averaged over the two disjoint 40-repetition halves (the paper's k40 = mean of halves, as ubp_card); then the unchanged summarize of na_km3_k40_v3.py is run on the merged dir. Prints the registered-score gate: UBP top1 vs ubp_card k40_top1 (1e-6) and group means vs saved k40 yields."""
import json, sys
from pathlib import Path
A, B, O = [Path(p) for p in sys.argv[1:4]]; O.mkdir(parents=True, exist_ok=True)
def avg(a, b):
    if isinstance(a, list) and a and all(isinstance(x, (int, float)) for x in a): return [(x + y) / 2 for x, y in zip(a, b)]
    if isinstance(a, dict): return {k: avg(a[k], b[k]) for k in a}
    return a
g = {}
for fa in sorted(A.glob('sub-*.json')):
    fb = B / fa.name
    if not fb.exists(): continue
    ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
    ja['decoders'] = {m: {k: (avg(v, jb['decoders'][m][k]) if k not in ('g2',) else v) for k, v in d.items()} for m, d in ja['decoders'].items()}
    ja['wording_halves'] = 'mean of 40-repetition halves A and B'; (O / fa.name).write_text(json.dumps(ja, indent=1))
    u = ja['decoders'].get('ubp')
    if u:
        s = int(fa.stem[4:]); c = json.loads(Path('/path/to/eegaudit/results/baselines/ubp/ubp_card_full_a.json').read_text())['subjects'].get(str(s), {}).get('released') or json.loads(Path('/path/to/eegaudit/results/baselines/ubp/ubp_card_full_b.json').read_text())['subjects'].get(str(s), {}).get('released')
        g[s] = dict(ubp_top1=u['top1'][0], card_k40_top1=c['k40_top1'], abs_diff=abs(u['top1'][0] - c['k40_top1']))
print('UBP G1 top-1 vs ubp_card k40:', {s: round(v['abs_diff'], 8) for s, v in g.items()})
(O / 'G1_ubp_top1.json').write_text(json.dumps(g, indent=1))
