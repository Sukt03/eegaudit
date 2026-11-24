"""Round 5, attack 3: is the split-half ceiling (trained NICE-GA space, 96 % top-1) a session artefact?
smoke_splithalf.py splits repetitions 0-39 vs 40-79; THINGS-EEG2's 80 test repetitions are 4 sessions x 20 in order, so that is
sessions 1-2 vs 3-4. Same logic as smoke_splithalf.py (copied, not modified) with several splits, in the three spaces
(raw z-scored voltage, random-init NICE-GA, trained NICE-GA seed 141), 200-way, shuffled-pair floor each:
  sess12_vs_34        reps 0-39 vs 40-79            (the paper's split; k = 40 per half, different sessions)
  even_vs_odd         reps 0,2,.. vs 1,3,..         (interleaved; both halves hold all four sessions; k = 40)
  sess1_vs_sess2      reps 0-19 vs 20-39            (the requested "within-session" split: adjacent sessions, k = 20)
  within_sess_1st_2nd reps {0-9,20-29,40-49,60-69} vs {10-19,30-39,50-59,70-79}  (first vs second half of every session; k = 40)
  sess1a_vs_sess1b    reps 0-9 vs 10-19             (one session only, k = 10)
  sess1a_vs_sess3a    reps 0-9 vs 40-49             (two sessions, same k = 10: the k-matched between-session partner)
  halves_k10_random   10 random k=10 vs k=10 disjoint draws from all 80 (mixed sessions, k = 10)
Also a session-structure diagnostic that needs no labels: the 80 x 80 correlation matrix of the per-repetition grand-mean ERP
(mean over the 200 images), summarised as mean within-block-of-20 vs between-block correlation.
Output: results/method_pilots/r5_splithalf_sessions_<subject>.json ; usage: python r5_splithalf_sessions.py sub-01 [sub-05 ...]"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import load_test, score, save, _norm
import run_acq_nice as N
dev = torch.device("cuda"); t0 = time.time()
Enc, Proj = N.load_classes("nice_ga")
torch.manual_seed(0); enc_r, proj_r = Enc().to(dev).eval(), Proj().to(dev).eval()       # same random init as smoke_splithalf.py


@torch.no_grad()
def emb(x, enc, proj):
    return proj(enc(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).float().cpu().numpy()


def raw(x):
    f = x.reshape(len(x), -1); return (f - f.mean(0)) / (f.std(0) + 1e-6)


SPLITS = {"sess12_vs_34": (np.arange(0, 40), np.arange(40, 80)), "even_vs_odd": (np.arange(0, 80, 2), np.arange(1, 80, 2)),
          "sess1_vs_sess2": (np.arange(0, 20), np.arange(20, 40)),
          "within_sess_1st_2nd": (np.concatenate([np.arange(s, s + 10) for s in (0, 20, 40, 60)]), np.concatenate([np.arange(s + 10, s + 20) for s in (0, 20, 40, 60)])),
          "sess1a_vs_sess1b": (np.arange(0, 10), np.arange(10, 20)), "sess1a_vs_sess3a": (np.arange(0, 10), np.arange(40, 50))}
for subject in sys.argv[1:]:
    X = np.asarray(load_test(subject), dtype=np.float32); rng = np.random.default_rng(0)
    enc_t, proj_t = Enc().to(dev).eval(), Proj().to(dev).eval(); tag = f"nice_ga_dgx_v2_{subject.replace('-', '')}_seed141"
    enc_t.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
    proj_t.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
    spaces = {"raw_voltage": raw, "random_init_encoder": lambda x: emb(x, enc_r, proj_r), "trained_encoder": lambda x: emb(x, enc_t, proj_t)}
    rows = []
    splits = dict(SPLITS)
    for d in range(10):
        p = rng.permutation(80); splits[f"halves_k10_random_{d}"] = (np.sort(p[:10]), np.sort(p[10:20]))
    for name, (ia, ib) in splits.items():
        A, B = X[:, ia].mean(1), X[:, ib].mean(1)
        for sp, f in spaces.items():
            fa, fb = f(A), f(B); S = _norm(fa) @ _norm(fb).T
            rows.append(dict(split=name, space=sp, k=int(len(ia)), condition="eeg", **score(S)))
            rows.append(dict(split=name, space=sp, k=int(len(ia)), condition="shuffled_pair", **score(S[:, rng.permutation(200)])))
    # label-free session diagnostic: per-repetition grand-mean ERP (80, 63*250), correlation matrix, within vs between blocks of 20
    G = X.mean(0).reshape(80, -1); G = (G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True); C = (G @ G.T) / G.shape[1]
    blk = np.repeat(np.arange(4), 20); same = blk[:, None] == blk[None, :]; off = ~np.eye(80, dtype=bool)
    diag_ = dict(within_block_corr=float(C[same & off].mean()), between_block_corr=float(C[~same].mean()),
                 block_means=[[float(C[np.ix_(blk == i, blk == j)][off[np.ix_(blk == i, blk == j)]].mean()) for j in range(4)] for i in range(4)],
                 adjacent_rep_corr=float(np.mean([C[i, i + 1] for i in range(79)])))
    # same diagnostic on image-specific signal: per-rep correlation of the (image x feature) matrix after removing the per-rep grand mean
    Xi = X - X.mean(0, keepdims=True); Xi = Xi.reshape(200, 80, -1); Xi = Xi - Xi.mean(0, keepdims=True)
    Ci = np.zeros((80, 80), np.float32)
    for a in range(80):
        for b in range(a + 1, 80):
            Ci[a, b] = Ci[b, a] = float((Xi[:, a] * Xi[:, b]).sum() / np.sqrt((Xi[:, a] ** 2).sum() * (Xi[:, b] ** 2).sum()))
    diag_["image_signal_within_block_corr"] = float(Ci[same & off].mean()); diag_["image_signal_between_block_corr"] = float(Ci[~same].mean())
    out = dict(subject=subject, note=__doc__, rows=rows, session_diagnostic=diag_, runtime_s=round(time.time() - t0, 1))
    save(f"r5_splithalf_sessions_{subject}.json", out)
    def g(split, sp, cond="eeg", key="top1"):
        v = [r[key] for r in rows if r["split"].startswith(split) and r["space"] == sp and r["condition"] == cond]; return float(np.mean(v))
    print(subject, "grand-ERP corr within/between session blocks: %.3f / %.3f ; image-signal within/between: %.4f / %.4f" % (diag_["within_block_corr"], diag_["between_block_corr"], diag_["image_signal_within_block_corr"], diag_["image_signal_between_block_corr"]))
    for split in list(SPLITS) + ["halves_k10_random"]:
        print(f"   {split:22s} " + " | ".join(f"{sp[:7]} top1 {100*g(split, sp):5.1f}% 2way {100*g(split, sp, key='two_way'):5.1f}% soft {g(split, sp, key='soft_bits'):.2f} (floor {100*g(split, sp, 'shuffled_pair'):.1f}%)" for sp in spaces), flush=True)
    del X, Xi
print(f"done ({time.time()-t0:.0f}s)")
