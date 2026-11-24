"""Synthetic proof for km3_decoders_v3.score: global-permutation floor ~ 0, within-profile floor > 0 (the cause of the failed smoke), unpermuted score = D1.gscore to 1e-6."""
import sys, numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/ext")
import km3_decoders_v3 as M
Ctr, Cte, _, _, _ = M.K.categories(); Ct = torch.as_tensor(Cte, dtype=torch.float64, device="cuda"); g = torch.Generator(device="cuda").manual_seed(0)
W = torch.randn(Ct.shape[1], 64, device="cuda", dtype=torch.float64, generator=g)
Y = Ct @ W + 0.5 * torch.randn(200, 64, device="cuda", dtype=torch.float64, generator=g)          # candidate embeddings, category-structured
for noise in (1.0, 4.0, 12.0):
    P = Y + noise * torch.randn(200, 64, device="cuda", dtype=torch.float64, generator=g)
    S = torch.stack([M.D1.gcos(P, Y)] * 2)
    r = M.score(S, Ct, Cte); d = abs(r["soft12"][0] - M.D1.gscore(S)["soft12"][0].item())
    print(f"noise {noise}: soft12 {r['soft12'][0]:.4f} global_floor {r['floor_soft12'][0]:+.4f} within_profile_floor {r['within_profile_shuffle_soft12'][0]:+.4f} unperm_diff {d:.2e}")
    m30 = float(np.mean([M.D1.gscore(S[0].index_select(-1, torch.as_tensor(np.random.default_rng(k).permutation(200), device="cuda")))["soft12"].item() for k in range(30)]))
    print(f"   mean of 30 global perms {m30:+.4f}")
    assert -0.08 < m30 < 0.02 and abs(r["floor_soft12"][0]) < 0.08 and r["within_profile_shuffle_soft12"][0] > 0.2 and d < 1e-6
print("PASS")
