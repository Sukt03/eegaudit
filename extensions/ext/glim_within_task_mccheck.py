"""Diagnostic (not part of the protocol run): Monte-Carlo spread of the sentence-cluster CI of EEG-minus-noise in the ZuCo1 <NR> cell (card12 grid),
from results/ext/glim_within_task/per_record.csv. Tests whether the 0.0054 miss of the CI lower edge in the G1 gate is bootstrap-seed noise."""
import csv, torch, numpy as np
dev = torch.device('cuda')
rows = [r for r in csv.DictReader(open('/path/to/eegaudit/results/ext/glim_within_task/per_record.csv')) if r['selection'] == 'cell:<NR>ZuCo1' and r['grid'] == 'card12']
d = torch.tensor([float(r['y_eeg']) - float(r['y_noise_test_time']) for r in rows], device=dev)
s = list(dict.fromkeys(r['sentence60'] for r in rows)); print('records', len(rows), 'unique sentence60', len(s))
cl = torch.tensor([s.index(r['sentence60']) for r in rows], device=dev); ncl = len(s)
def ci(nb, seed):
    g = torch.Generator(device=dev); g.manual_seed(seed); ms = []
    for _ in range(nb // 2500):
        pick = torch.randint(0, ncl, (2500, ncl), device=dev, generator=g)
        w = torch.zeros(2500, ncl, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))[:, cl]
        ms.append((w * d[None]).sum(1) / w.sum(1))
    m = torch.cat(ms); return float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))
print('mean', float(d.mean()))
for nb in (10000, 100000):
    v = np.array([ci(nb, sd) for sd in range(10)]); print(nb, 'lo mean/min/max', v[:, 0].mean().round(4), v[:, 0].min().round(4), v[:, 0].max().round(4), '| hi mean/min/max', v[:, 1].mean().round(4), v[:, 1].min().round(4), v[:, 1].max().round(4))
