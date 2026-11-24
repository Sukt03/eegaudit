"""Protocol audit A: is Wolpaw bits-on-top-1 a sound estimator of the information an EEG query
carries about the image?  Compares, on ATM-S test embeddings (10 subjects, 200 candidates):
  wolpaw      : log2 N + P log2 P + (1-P) log2((1-P)/(N-1)) on top-1 accuracy P  (= Fano lower bound)
  fano        : identical formula written as log2 N - h(Pe) - Pe log2(N-1); printed to make the identity explicit
  mi_hard     : plug-in I(Y; argmax) from the N x N hard confusion matrix, naive / Miller-Madow / jackknife
  mi_soft     : I(Y; X) lower bound from a softmax posterior over cosine similarities, temperature fit by
                leave-one-out log-likelihood (Barber-Agakov / InfoNCE-style variational bound), bits
  rank_bits   : log2 N - mean log2(rank of correct item)   (rank-based)
  mrr, mrr_bits: mean reciprocal rank and log2(N * MRR)
  two_way, two_way_bits: pairwise identification accuracy and 1 - h(two_way) (Wolpaw at N=2)
Then: Spearman rank correlation of estimators across subjects, and behaviour under candidate-set subsampling
(N = 50, 100, 200) with efficiency = bits / log2 N.
Read-only on data; writes only the report under outputs/.
"""
import sys, glob, re, json, time
import numpy as np, torch
from scipy.stats import spearmanr

EEG_GLOB = "/path/to/atm/data/hf_probe/emb_eeg/ATM_S_eeg_features_sub-*_test.pt"
TGT = "/path/to/atm/data/hf_probe/ViT-H-14_features_test.pt"
OUT_DIR = "/path/to/eegaudit/outputs"
LOG2 = np.log(2)

def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def wolpaw_bits(P, N):
    P = min(max(P, 1e-9), 1 - 1e-9)
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))

def fano_bits(P, N):
    Pe = 1 - P
    return float(np.log2(N) - h2(Pe) - Pe * np.log2(N - 1))

def plugin_mi(counts):
    """I(Y; Yhat) in bits from a count matrix (rows = true, cols = predicted)."""
    n = counts.sum()
    pj = counts / n
    py = pj.sum(1, keepdims=True); pyh = pj.sum(0, keepdims=True)
    nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / (py @ pyh)[nz])).sum())

def mi_hard_estimates(y, yhat, N):
    n = len(y)
    C = np.zeros((N, N)); np.add.at(C, (y, yhat), 1)
    naive = plugin_mi(C)
    # Miller-Madow: H_mm = H_plug + (m-1)/(2n ln2) for each entropy term; I = H(Y)+H(Yh)-H(Y,Yh)
    m_y = (C.sum(1) > 0).sum(); m_yh = (C.sum(0) > 0).sum(); m_j = (C > 0).sum()
    mm = naive + ((m_y - 1) + (m_yh - 1) - (m_j - 1)) / (2 * n * LOG2)
    # jackknife (leave-one-sample-out)
    loo = []
    for i in range(n):
        C2 = C.copy(); C2[y[i], yhat[i]] -= 1
        loo.append(plugin_mi(C2))
    jk = n * naive - (n - 1) * np.mean(loo)
    return naive, float(mm), float(jk)

def soft_mi(S, idx):
    """Variational lower bound on I(image; EEG): fit a softmax temperature on the candidate set by LOO
    log-likelihood (each query is its own test; tau chosen on the other queries), then
    I >= log2 N - mean cross-entropy (bits). S is the full cosine matrix, idx the candidate subset."""
    Ss = S[np.ix_(idx, idx)]
    n = len(idx)
    taus = np.logspace(-2.5, 0, 40)
    def ce(Sm, tau, rows):
        Z = Sm[rows] / tau
        Z = Z - Z.max(1, keepdims=True)
        lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -lp[np.arange(len(rows)), rows] / LOG2
    all_rows = np.arange(n)
    # choose tau once on all rows (tiny bias, one scalar) and also strict LOO
    ce_all = np.array([ce(Ss, t, all_rows).mean() for t in taus])
    tau_star = taus[ce_all.argmin()]
    loo = np.empty(n)
    for i in range(n):
        others = np.delete(all_rows, i)
        ce_o = np.array([ce(Ss, t, others).mean() for t in taus])
        t_i = taus[ce_o.argmin()]
        loo[i] = ce(Ss, t_i, np.array([i]))[0]
    return float(np.log2(n) - loo.mean()), float(tau_star)

def estimators(S, idx):
    """All estimators restricted to candidate set idx (queries = the same items)."""
    Ss = S[np.ix_(idx, idx)]
    n = len(idx)
    diag = np.diag(Ss)
    rank = (Ss > diag[:, None]).sum(1) + 1
    top1 = float((rank == 1).mean())
    yhat = Ss.argmax(1); y = np.arange(n)
    naive, mm, jk = mi_hard_estimates(y, yhat, n)
    soft, tau = soft_mi(S, idx)
    two_way = float(((Ss < diag[:, None]).sum(1) / (n - 1)).mean())
    out = dict(N=n, top1=top1, wolpaw=wolpaw_bits(top1, n), fano=fano_bits(top1, n),
               mi_hard_naive=naive, mi_hard_mm=mm, mi_hard_jk=jk, mi_soft=soft, tau=tau,
               rank_bits=float(np.log2(n) - np.log2(rank).mean()),
               mrr=float((1 / rank).mean()), mrr_bits=float(np.log2(n * (1 / rank).mean())),
               two_way=two_way, two_way_bits=1 - h2(two_way), mean_rank=float(rank.mean()))
    return out

def main(smoke=False):
    t0 = time.time()
    tgt = torch.load(TGT, map_location="cpu")["img_features"].float().numpy()
    tgt = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    files = sorted(glob.glob(EEG_GLOB))
    if smoke: files = files[:2]
    subs, full, sub_rows = [], [], []
    rng = np.random.default_rng(0)
    draws = 3 if smoke else 10
    chance = []
    for f in files:
        sid = re.search(r"sub-(\d+)", f).group(1)
        e = torch.load(f, map_location="cpu").float().numpy()
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        S = e @ tgt.T
        r = estimators(S, np.arange(200)); r["subject"] = sid
        full.append(r); subs.append(sid)
        # shuffled-pairs control: permute which EEG row is scored against which target (chance floor)
        Sp = S[rng.permutation(200)]
        c = estimators(Sp, np.arange(200)); c["subject"] = sid; chance.append(c)
        for N in (50, 100, 200):
            for d in range(draws if N < 200 else 1):
                idx = np.sort(rng.choice(200, N, replace=False)) if N < 200 else np.arange(200)
                rr = estimators(S, idx); rr.update(subject=sid, draw=d)
                sub_rows.append(rr)
        print(f"sub-{sid} top1={r['top1']:.3f} wolpaw={r['wolpaw']:.2f} mi_hard={r['mi_hard_naive']:.2f}/"
              f"{r['mi_hard_mm']:.2f}/{r['mi_hard_jk']:.2f} soft={r['mi_soft']:.2f} rank={r['rank_bits']:.2f} "
              f"2way={r['two_way']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    keys = ["top1", "wolpaw", "mi_hard_naive", "mi_hard_mm", "mi_hard_jk", "mi_soft", "rank_bits", "mrr_bits",
            "two_way_bits"]
    # ---- report
    L = []
    L.append("# Protocol audit A: estimators of yield (bits) on ATM-S test embeddings\n")
    L.append(f"Data: {len(files)} subjects, 200 test images, ATM-S EEG embeddings vs ViT-H-14 image features "
             f"(cosine). Script: `experiments/protocol_audit/A_estimators.py`. Run on GPU node 0 via srun.\n")
    L.append("## 1. Per-subject estimates, N = 200 (bits unless stated)\n")
    L.append("| subj | top-1 | Wolpaw | Fano | MI hard naive | MI hard Miller-Madow | MI hard jackknife | MI soft (LOO softmax) | rank bits | MRR bits | 2-way acc | 2-way bits |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in full:
        L.append(f"| {r['subject']} | {r['top1']:.3f} | {r['wolpaw']:.2f} | {r['fano']:.2f} | {r['mi_hard_naive']:.2f} | "
                 f"{r['mi_hard_mm']:.2f} | {r['mi_hard_jk']:.2f} | {r['mi_soft']:.2f} | {r['rank_bits']:.2f} | "
                 f"{r['mrr_bits']:.2f} | {r['two_way']:.3f} | {r['two_way_bits']:.3f} |")
    A = {k: np.array([r[k] for r in full]) for k in keys + ["two_way", "fano"]}
    L.append(f"| **mean** | {A['top1'].mean():.3f} | {A['wolpaw'].mean():.2f} | {A['fano'].mean():.2f} | "
             f"{A['mi_hard_naive'].mean():.2f} | {A['mi_hard_mm'].mean():.2f} | {A['mi_hard_jk'].mean():.2f} | "
             f"{A['mi_soft'].mean():.2f} | {A['rank_bits'].mean():.2f} | {A['mrr_bits'].mean():.2f} | "
             f"{A['two_way'].mean():.3f} | {A['two_way_bits'].mean():.3f} |")
    Cc = {k: np.array([c[k] for c in chance]) for k in keys}
    L.append("| **shuffled pairs (chance)** | " + f"{Cc['top1'].mean():.3f} | {Cc['wolpaw'].mean():.2f} | {Cc['wolpaw'].mean():.2f} | "
             f"{Cc['mi_hard_naive'].mean():.2f} | {Cc['mi_hard_mm'].mean():.2f} | {Cc['mi_hard_jk'].mean():.2f} | "
             f"{Cc['mi_soft'].mean():.2f} | {Cc['rank_bits'].mean():.2f} | {Cc['mrr_bits'].mean():.2f} | "
             f"{np.mean([c['two_way'] for c in chance]):.3f} | {Cc['two_way_bits'].mean():.3f} |")
    L.append(f"\nlog2 N = {np.log2(200):.3f} bits. Maximum Wolpaw-Fano difference across subjects: "
             f"{np.abs(A['wolpaw']-A['fano']).max():.2e} (algebraically identical).\n")
    L.append("## 2. Spearman rank correlation across subjects (N = 200)\n")
    L.append("| | " + " | ".join(keys) + " |"); L.append("|---|" + "---|" * len(keys))
    for a in keys:
        row = []
        for b in keys:
            rho = spearmanr(A[a], A[b]).correlation
            row.append(f"{rho:.2f}")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## 3. Candidate-set subsampling: efficiency = bits / log2 N (mean ± sd over subjects; "
             f"{draws} random subsets per N<200)\n")
    L.append("| N | top-1 | Wolpaw bits | Wolpaw/log2N | MI hard MM | MI hard MM/log2N | MI soft | MI soft/log2N | rank bits | rank/log2N | 2-way bits |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sub_summary = {}
    for N in (50, 100, 200):
        rows = [r for r in sub_rows if r["N"] == N]
        # average over draws within subject first, then over subjects
        per_sub = {}
        for r in rows: per_sub.setdefault(r["subject"], []).append(r)
        agg = {k: np.array([np.mean([x[k] for x in v]) for v in per_sub.values()]) for k in
               ["top1", "wolpaw", "mi_hard_mm", "mi_soft", "rank_bits", "two_way_bits"]}
        lg = np.log2(N)
        sub_summary[N] = agg
        f = lambda k: f"{agg[k].mean():.3f} ± {agg[k].std():.3f}"
        L.append(f"| {N} | {f('top1')} | {f('wolpaw')} | {(agg['wolpaw']/lg).mean():.3f} ± {(agg['wolpaw']/lg).std():.3f} | "
                 f"{f('mi_hard_mm')} | {(agg['mi_hard_mm']/lg).mean():.3f} ± {(agg['mi_hard_mm']/lg).std():.3f} | "
                 f"{f('mi_soft')} | {(agg['mi_soft']/lg).mean():.3f} ± {(agg['mi_soft']/lg).std():.3f} | "
                 f"{f('rank_bits')} | {(agg['rank_bits']/lg).mean():.3f} ± {(agg['rank_bits']/lg).std():.3f} | {f('two_way_bits')} |")
    # subject ranking stability across N
    L.append("\nSubject-ranking stability across N (Spearman between per-subject estimate at N=50 and at N=200):\n")
    L.append("| estimator | rho(N=50, N=200) | rho(N=100, N=200) |"); L.append("|---|---|---|")
    for k in ["top1", "wolpaw", "mi_hard_mm", "mi_soft", "rank_bits", "two_way_bits"]:
        L.append(f"| {k} | {spearmanr(sub_summary[50][k], sub_summary[200][k]).correlation:.2f} | "
                 f"{spearmanr(sub_summary[100][k], sub_summary[200][k]).correlation:.2f} |")
    L.append("")
    stamp = time.strftime("%Y%m%d-%H%M")
    path = f"{OUT_DIR}/PROTOCOL_AUDIT_A_estimators_{stamp}{'_SMOKE' if smoke else ''}.md"
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(path.replace(".md", ".json"), "w") as fh:
        json.dump(dict(full=full, chance=chance, subsample=sub_rows), fh, indent=1)
    print("wrote", path)

if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
