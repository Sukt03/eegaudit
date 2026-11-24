"""G2 (round_20261124): reads the existing rf2 MI matched random-derivation arms (no rerun) and reports per-family percentiles of the kept share
next to the proxy value. All numerics in torch on the GPU.
Same count (5 columns) and same construction (4 bipolar differences + 1 three-channel mean, 11 distinct channels) as the proxies -> verified from arms.json pairs/mean3."""
import json, sys
from pathlib import Path
import torch
R = Path('/path/to/eegaudit/results/runner_fixes_20261123/rf2_mi'); O = Path('/path/to/eegaudit/results/runner_fixes_20261123/rf9_mi_random_derivations'); O.mkdir(exist_ok=True)
S = json.loads((R / 'summary.json').read_text()); AJ = json.loads((R / 'arms.json').read_text()); A = AJ['arms']; dev = 'cuda'
q = torch.tensor([.05, .5, .95], dtype=torch.float64, device=dev); out = dict(NOT_CITABLE='until independent review', source='rf2_mi (reused, not rerun)', gate=S['gate'], gate_pass=S['gate_pass'],
    gate_max_abs_diff=max(v['abs_diff'] for v in S['gate'].values()), raw_bits=S['raw_bits'], proxy_kept=S['proxy']['kept'], proxy_ci95=S['proxy']['ci95'], proxy_bits=S['proxy']['bits'], families={})
raw = torch.tensor(A['raw']['bits'], dtype=torch.float64, device=dev); IX = torch.randint(0, len(raw), (10000, len(raw)), generator=torch.Generator(device=dev).manual_seed(0), device=dev)
for fam in ['frontal', 'nonfrontal', 'any']:
    ks = sorted(k for k in A if k.startswith(fam + '_d') and '|' not in k); bits = torch.stack([torch.tensor(A[k]['bits'], dtype=torch.float64, device=dev) for k in ks])
    kept = bits.mean(1) / raw.mean(); p = torch.quantile(kept, q); pr = float(S['proxy']['kept'])
    # construction check: 4 pairs + 3-channel mean, 11 distinct channels, all from the family pool
    pool = set(AJ['pools'][fam]); ok = all(len(A[k]['pairs']) == 4 and len(A[k]['mean3']) == 3 and len({c for pp in A[k]['pairs'] for c in pp} | set(A[k]['mean3'])) == 11 and ({c for pp in A[k]['pairs'] for c in pp} | set(A[k]['mean3'])) <= pool for k in ks)
    # participant-bootstrap CI of each percentile (resample participants, recompute kept per draw, then percentile over draws)
    kb = bits[:, IX].mean(2) / raw[IX].mean(1)[None]           # [draws, B]
    pb = torch.quantile(kb, q, dim=0)                           # [3, B]
    blank = {k: A[k]['mean'] for k in A if '|' in k and k.startswith(fam)}
    out['families'][fam] = dict(n_draws=len(ks), construction_ok=ok, p05=float(p[0]), p50=float(p[1]), p95=float(p[2]),
        p05_ci95=[float(pb[0].quantile(.025)), float(pb[0].quantile(.975))], p50_ci95=[float(pb[1].quantile(.025)), float(pb[1].quantile(.975))], p95_ci95=[float(pb[2].quantile(.025)), float(pb[2].quantile(.975))],
        min=float(kept.min()), max=float(kept.max()), proxy_inside_p05_p95=bool(p[0] <= pr <= p[2]), median_above_0_5=bool(p[1] > 0.5), n_draws_le_proxy=int((kept <= pr).sum()), blanks=blank,
        blanks_max_abs=max(abs(v) for v in blank.values()), blanks_within_0_02=all(abs(v) <= 0.02 for v in blank.values()))
out['all_blanks_within_0_02'] = S['nulls_ok_abs_le_0.02']; out['planted_control'] = S['planted_control']
(O / 'summary.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
