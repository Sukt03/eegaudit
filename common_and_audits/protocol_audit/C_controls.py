"""Protocol audit C: controls and leakage.

(1) Category-aware null for THINGS-EEG2 200-way identification. Candidate pools restricted to the target's CLIP-nearest
    neighbours (pool sizes 2, 10, 50) versus random pools of the same size, for the native ATM-S embeddings and for the
    eegbits linear probe (run.py functions reused: PCA -> ridge -> cosine scoring), with the protocol's trial_shuffled
    and shuffled_pair controls recomputed inside each pool.
(2) Leakage checks that can be made from the data: train/test concept disjointness via the CLIP text features released
    with the dataset (one text vector per concept; a shared concept gives cosine 1.0) and via image-feature nearest
    neighbours; pick_alpha uses training rows only (read from run.py, asserted here by calling it with no test input).
(3) Decision EEG: session-permutation null (as in group_stats.py) recomputed with the same code next to a circular-shift
    null and a block-permutation null of the participant's own prediction-error series, per decoder.
(4) Existing random-projection and target-rank controls aggregated from results/eegbits/controls_sub-*.json.

Read-only on all existing files. Writes results/protocol_audit/C_controls.json.
Usage (inside a compute node):
  srun --overlap -N1 -n1 --cpus-per-task=2 --mem=6G -t 20 \
    python experiments/protocol_audit/C_controls.py [--part 1,2,3,4] [--smoke]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/path/to/eegaudit")
sys.path.insert(0, str(ROOT / "experiments/eegbits"))
sys.path.insert(0, str(ROOT))
import run as R  # eegbits pipeline functions (load_torch, pca_scores, ridge_predict, pick_alpha, metrics, wolpaw_bits)

OUT = ROOT / "results/protocol_audit/C_controls.json"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
POOLS = [2, 10, 50, 200]
N_RANDOM_POOLS = 20


# ----------------------------------------------------------------------------------------------------- part 1 helpers
def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pool_metrics(S, pools):
    """S: (n queries x n candidates) similarity, pools: list of index arrays, each containing the query index itself.
    Returns top-1 within pool, 2-way (fraction of pool negatives below the target), Wolpaw bits with N = pool size."""
    n = S.shape[0]
    top1, two = [], []
    for i in range(n):
        p = np.asarray(pools[i])
        s = S[i, p]; st = S[i, i]
        top1.append(float(st >= s.max()) if (s > st).sum() == 0 else 0.0)
        neg = s[p != i]
        two.append(float((neg < st).mean()))
    N = len(pools[0])
    t1 = float(np.mean(top1))
    return dict(N=N, top1=t1, two_way=float(np.mean(two)), bits=R.wolpaw_bits(t1, N), chance_top1=1.0 / N)


def hard_pools(C, N):
    """C: (200 x 200) cosine among targets. Pool for query i = i plus its N-1 nearest other targets."""
    pools = []
    for i in range(C.shape[0]):
        order = np.argsort(-C[i])
        order = order[order != i][: N - 1]
        pools.append(np.r_[i, order])
    return pools


def random_pools(n, N, rng):
    pools = []
    for i in range(n):
        others = rng.choice([j for j in range(n) if j != i], N - 1, replace=False)
        pools.append(np.r_[i, others])
    return pools


def score_all_pools(S, C, rng_seed=0):
    """Score a similarity matrix under hard and random pools of every size."""
    out = {}
    for N in POOLS:
        if N == 200:
            pools = [np.arange(200)] * 200
            out[str(N)] = dict(hard=pool_metrics(S, pools), random=pool_metrics(S, pools))
            continue
        hard = pool_metrics(S, hard_pools(C, N))
        rng = np.random.default_rng(rng_seed)
        rnd = [pool_metrics(S, random_pools(200, N, rng)) for _ in range(N_RANDOM_POOLS)]
        out[str(N)] = dict(hard=hard, random={k: float(np.mean([r[k] for r in rnd])) for k in ("top1", "two_way", "bits")} | dict(N=N, chance_top1=1.0 / N))
    return out


def part1(smoke=False):
    import torch
    te = torch.load(R.CLIP / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)
    Yte = te["img_features"].float().numpy()
    Ytr = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt")
    Yte_u = unit(Yte)
    C = Yte_u @ Yte_u.T
    np.fill_diagonal(C, -np.inf)
    nn_cos = C.max(1)
    np.fill_diagonal(C, 1.0)
    res = dict(target_structure=dict(nearest_neighbour_cosine_mean=float(nn_cos.mean()), nearest_neighbour_cosine_min=float(nn_cos.min()),
                                     nearest_neighbour_cosine_max=float(nn_cos.max()),
                                     offdiag_cosine_mean=float(C[~np.eye(200, dtype=bool)].mean()),
                                     pool10_mean_cosine=float(np.mean([C[i, p[1:]].mean() for i, p in enumerate(hard_pools(C, 10))])),
                                     pool50_mean_cosine=float(np.mean([C[i, p[1:]].mean() for i, p in enumerate(hard_pools(C, 50))])),
                                     random_pool_mean_cosine=float(C[~np.eye(200, dtype=bool)].mean())),
               atm_s={}, probe={})
    subs = SUBJECTS[:2] if smoke else SUBJECTS
    rng_pair = np.random.default_rng(200)
    perm = rng_pair.permutation(200)
    for s in subs:
        t0 = time.time()
        Xte = R.load_torch(R.EMB / f"ATM_S_eeg_features_{s}_test.pt")
        S_atm = unit(Xte) @ Yte_u.T
        res["atm_s"][s] = dict(eeg=score_all_pools(S_atm, C), shuffled_pair=score_all_pools(unit(Xte) @ Yte_u[perm].T, C))
        # linear probe, exactly as run.py: mean of 4 reps, standardise+PCA on train, ridge, alpha picked on train rows only
        Xtr = R.load_torch(R.EMB / f"ATM_S_eeg_features_{s}_train.pt").reshape(len(Ytr), 4, -1).mean(1)
        if smoke:
            Xtr, Ytr_s = Xtr[:3000], Ytr[:3000]
        else:
            Ytr_s = Ytr
        Ztr, (Zte,), _ = R.pca_scores(Xtr, [Xte])
        rng = np.random.default_rng(0)
        res["probe"][s] = {}
        for L in (50, min(1024, Ztr.shape[1], len(Ztr))):
            alpha = R.pick_alpha(Ztr[:, :L], Ytr_s, rng)
            pred = R.ridge_predict(Ztr[:, :L], Ytr_s, Zte[:, :L], alpha)
            S_eeg = unit(pred) @ Yte_u.T
            r = np.random.default_rng(200)
            pred_sh = R.ridge_predict(Ztr[r.permutation(len(Ztr)), :L], Ytr_s, Zte[:, :L], alpha)
            res["probe"][s][str(L)] = dict(alpha=alpha, eeg=score_all_pools(S_eeg, C),
                                           trial_shuffled=score_all_pools(unit(pred_sh) @ Yte_u.T, C),
                                           shuffled_pair=score_all_pools(unit(pred) @ Yte_u[perm].T, C))
        print(f"[1] {s} atm top1 200/50/10/2: " + " ".join(f"{res['atm_s'][s]['eeg'][str(N)]['hard']['top1']:.3f}" for N in (200, 50, 10, 2))
              + f" | probe L1024 hard: " + " ".join(f"{res['probe'][s][str(max(int(k) for k in res['probe'][s]))]['eeg'][str(N)]['hard']['top1']:.3f}" for N in (200, 50, 10, 2))
              + f" ({time.time() - t0:.0f}s)", flush=True)
    # aggregate
    def agg(getter):
        vals = {str(N): {} for N in POOLS}
        for N in POOLS:
            for kind in ("hard", "random"):
                for m in ("top1", "two_way", "bits"):
                    v = [getter(s)[str(N)][kind][m] for s in subs]
                    vals[str(N)][f"{kind}_{m}_mean"] = float(np.mean(v)); vals[str(N)][f"{kind}_{m}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            vals[str(N)]["chance_top1"] = 1.0 / N
        return vals
    res["summary"] = dict(atm_s_eeg=agg(lambda s: res["atm_s"][s]["eeg"]), atm_s_shuffled_pair=agg(lambda s: res["atm_s"][s]["shuffled_pair"]))
    Lfull = max(int(k) for k in res["probe"][subs[0]])
    for L in (50, Lfull):
        res["summary"][f"probe_L{L}_eeg"] = agg(lambda s, L=L: res["probe"][s][str(L)]["eeg"])
        res["summary"][f"probe_L{L}_trial_shuffled"] = agg(lambda s, L=L: res["probe"][s][str(L)]["trial_shuffled"])
        res["summary"][f"probe_L{L}_shuffled_pair"] = agg(lambda s, L=L: res["probe"][s][str(L)]["shuffled_pair"])
    return res


# ----------------------------------------------------------------------------------------------------- part 2
def part2():
    import torch
    te = torch.load(R.CLIP / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)
    tr = torch.load(R.CLIP / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)
    Tte, Ttr = unit(te["text_features"].float().numpy()), unit(tr["text_features"].float().numpy())
    Ite, Itr = unit(te["img_features"].float().numpy()), unit(tr["img_features"].float().numpy())
    text_max = (Tte @ Ttr.T).max(1)                       # one text vector per concept ("This picture is <concept>")
    img_max = (Ite @ Itr.T).max(1)                        # each test image's nearest training image
    # within-concept image cosine in training: images are stored in blocks of 10 per concept (loader: samples_per_class=10)
    G = Itr.reshape(1654, 10, -1)
    within = np.array([(g @ g.T)[np.triu_indices(10, 1)].mean() for g in G])
    # between-concept: nearest training image from a different concept, for a random 400 training images
    rng = np.random.default_rng(0)
    idx = rng.choice(16540, 400, replace=False)
    Sx = Itr[idx] @ Itr.T
    for k, i in enumerate(idx):
        c = i // 10
        Sx[k, c * 10:(c + 1) * 10] = -np.inf
    between_nn = Sx.max(1)
    # pick_alpha signature: (Ztr, Ytr, rng) -> no test argument can be passed; the 90/10 split is over training rows
    import inspect
    sig = str(inspect.signature(R.pick_alpha))
    src = inspect.getsource(R.pick_alpha)
    return dict(train_concepts=int(Ttr.shape[0]), test_concepts=int(Tte.shape[0]), train_images=int(Itr.shape[0]), test_images=int(Ite.shape[0]),
                text_cosine_test_to_nearest_train_concept=dict(max=float(text_max.max()), mean=float(text_max.mean()), n_at_or_above_0_99=int((text_max >= 0.99).sum()),
                                                               n_at_or_above_0_95=int((text_max >= 0.95).sum())),
                image_cosine_test_to_nearest_train_image=dict(max=float(img_max.max()), mean=float(img_max.mean()), n_above_0_9=int((img_max > 0.9).sum())),
                train_within_concept_image_cosine=dict(mean=float(within.mean()), p05=float(np.percentile(within, 5))),
                train_between_concept_nearest_image_cosine=dict(mean=float(between_nn.mean()), p95=float(np.percentile(between_nn, 95)), max=float(between_nn.max())),
                pick_alpha_signature=sig, pick_alpha_source=src)


# ----------------------------------------------------------------------------------------------------- part 3
def part3(decoders, smoke=False):
    import torch
    from belief_bottleneck import cv_predict
    from cogmodels import models as M
    Q = np.load(ROOT / "results/full_36318/behaviour_quantities_flow.npz")
    model = {**M.MODELS, **M.EXTRA_MODELS}["RWD"]
    Z = torch.tensor(np.load(ROOT / "results/map_RWD.npy")).float()
    probs = M.reward_schedule(480)
    nS = Z.shape[0]
    full_q = []
    for i in range(nS):
        with torch.no_grad():
            full_q.append(M.run(model, Z[i][None], probs, torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None],
                                torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy())

    def resid(v, cols):
        Cm = np.c_[np.ones(len(v)), np.column_stack(cols)]
        b, *_ = np.linalg.lstsq(Cm, v, rcond=None)
        return v - Cm @ b

    def qtil(j, tri):
        c, r = Q["choice"][j][tri], Q["reward"][j][tri]
        cols = [r] + [(c == k).astype(float) for k in range(3)] + [(c == k).astype(float) * r for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
        v = resid(full_q[j][tri].astype(float), cols)
        return (v - v.mean()) / (v.std() + 1e-9)

    def corr_cols(P, Y):
        P = P - P.mean(0); Y = Y - Y.mean(0)
        return (P * Y).sum(0) / np.sqrt((P * P).sum(0) * (Y * Y).sum(0) + 1e-18)

    res = {}
    subs = range(3) if smoke else range(nS)
    MIN_LAG = 10
    BLOCK = 24
    for dec in decoders:
        t0 = time.time()
        rows = []
        for i in subs:
            fz = np.load(ROOT / f"results/features/{dec}" / f"sub-{i + 1:03d}.npz")
            F, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
            n = len(tri)
            y = qtil(i, tri)
            Ysess = np.column_stack([qtil(j, tri) for j in range(nS)])                    # column i = own; others = session null
            shifts = np.arange(MIN_LAG, n - MIN_LAG)
            Ycirc = np.column_stack([np.roll(y, int(s)) for s in shifts])                   # circular shift of own series
            rng = np.random.default_rng(1000 + i)
            nb = int(np.ceil(n / BLOCK))
            Yblk = []
            for _ in range(200):
                order = rng.permutation(nb)
                perm = np.concatenate([np.arange(b * BLOCK, min((b + 1) * BLOCK, n)) for b in order])
                Yblk.append(y[perm])
            Yblk = np.column_stack(Yblk)
            Yall = np.column_stack([Ysess, Ycirc, Yblk])
            P = cv_predict(F, Yall, "pca", 64, colw=colw)                                  # per-target ridge: exact batching
            rr = corr_cols(P, Yall)
            r = float(rr[i])
            r_sess = np.delete(rr[:nS], i)
            r_circ = rr[nS:nS + len(shifts)]
            r_blk = rr[nS + len(shifts):]
            lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
            rows.append(dict(subject=i + 1, n=n, r=r, lag1_autocorr_q0=lag1,
                             sess=dict(mean=float(r_sess.mean()), sd=float(r_sess.std(ddof=1)), q95=float(np.quantile(r_sess, .95)), max=float(r_sess.max()), mean_r2=float((r_sess ** 2).mean()), n=int(len(r_sess))),
                             circ=dict(mean=float(r_circ.mean()), sd=float(r_circ.std(ddof=1)), q95=float(np.quantile(r_circ, .95)), max=float(r_circ.max()), mean_r2=float((r_circ ** 2).mean()), n=int(len(r_circ)),
                                       p=float((1 + (r_circ >= r).sum()) / (len(r_circ) + 1))),
                             block=dict(mean=float(r_blk.mean()), sd=float(r_blk.std(ddof=1)), q95=float(np.quantile(r_blk, .95)), max=float(r_blk.max()), mean_r2=float((r_blk ** 2).mean()), n=int(len(r_blk)),
                                        p=float((1 + (r_blk >= r).sum()) / (len(r_blk) + 1)))))
        R_ = np.array([w["r"] for w in rows])
        summ = dict(n_subjects=len(rows), r_mean=float(R_.mean()), r_min=float(R_.min()))
        for nm in ("sess", "circ", "block"):
            sd = np.array([w[nm]["sd"] for w in rows]); q95 = np.array([w[nm]["q95"] for w in rows]); mx = np.array([w[nm]["max"] for w in rows])
            mr2 = np.array([w[nm]["mean_r2"] for w in rows])
            summ[nm] = dict(null_sd_mean=float(sd.mean()), null_q95_mean=float(q95.mean()), null_max_mean=float(mx.mean()), null_mean_r2=float(mr2.mean()),
                            n_above_q95=int((R_ > q95).sum()), n_above_max=int((R_ > mx).sum()),
                            snr_mean=float(np.mean([(w["r"] ** 2 - w[nm]["mean_r2"]) / max(1 - w["r"] ** 2, 1e-6) for w in rows])))
        summ["snr_ratio_circ_over_sess"] = float(summ["circ"]["snr_mean"] / summ["sess"]["snr_mean"]) if summ["sess"]["snr_mean"] else None
        summ["snr_ratio_block_over_sess"] = float(summ["block"]["snr_mean"] / summ["sess"]["snr_mean"]) if summ["sess"]["snr_mean"] else None
        res[dec] = dict(summary=summ, rows=rows)
        print(f"[3] {dec}: r={summ['r_mean']:.3f} | null sd sess {summ['sess']['null_sd_mean']:.3f} circ {summ['circ']['null_sd_mean']:.3f} block {summ['block']['null_sd_mean']:.3f}"
              f" | above q95: sess {summ['sess']['n_above_q95']} circ {summ['circ']['n_above_q95']} block {summ['block']['n_above_q95']} /{len(rows)}"
              f" | SNR ratio circ/sess {summ['snr_ratio_circ_over_sess']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # what the paper's results.json says, for cross-reference
    gs = json.load(open(ROOT / "results/group_stats/results.json"))
    res["_paper_results_json"] = {d: dict(r_mean=float(np.mean(v["r"])), null_sd=float(np.mean([np.std(z, ddof=1) for z in v["null_z"]])),
                                          indiv_above_95=v["n_individually_above_95pct_null"], prevalence=v["prevalence"], meta_mu=v["meta"]["mu"])
                                  for d, v in gs["decoders"].items() if d in decoders}
    return res


# ----------------------------------------------------------------------------------------------------- part 4
def part4():
    out = dict(per_subject={}, summary={})
    Ls = None
    for f in sorted(glob.glob(str(ROOT / "results/eegbits/controls_sub-*.json"))):
        d = json.load(open(f)); s = d["subject"]
        emb = json.load(open(ROOT / f"results/eegbits/embedding_{s}.json"))
        eeg = {r["L"]: r for r in emb["rows"] if r["condition"] == "eeg"}
        tr = {r["L"]: r for r in d["rows"] if r["condition"] == "clip_rank_L_perfect_decoder"}
        rp = {}
        for r in d["rows"]:
            if r["condition"] == "eeg_random_projection":
                rp.setdefault(r["L"], []).append(r)
        Ls = sorted(tr)
        out["per_subject"][s] = {str(L): dict(eeg_top1=eeg[L]["top1"], eeg_two_way=eeg[L]["two_way"], eeg_bits=eeg[L]["bits"],
                                               target_rank_top1=tr[L]["top1"], target_rank_bits=tr[L]["bits"],
                                               random_proj_top1=float(np.mean([r["top1"] for r in rp[L]])), random_proj_two_way=float(np.mean([r["two_way"] for r in rp[L]])),
                                               random_proj_bits=float(np.mean([r["bits"] for r in rp[L]])),
                                               eeg_adj2way_over_randproj_adj2way=float((eeg[L]["two_way"] - .5) / max(np.mean([r["two_way"] for r in rp[L]]) - .5, 1e-9)),
                                               ceiling_binds=bool(eeg[L]["top1"] > 0.5 * tr[L]["top1"]))
                                 for L in Ls}
        out["per_subject"][s]["native"] = d["native"]; out["per_subject"][s]["native_efficiency"] = d["native_efficiency"]
        out["per_subject"][s]["frac_at_L50"] = emb["frac_at_L50"]
    for L in Ls:
        keys = ["eeg_top1", "eeg_bits", "target_rank_top1", "target_rank_bits", "random_proj_top1", "random_proj_bits", "eeg_adj2way_over_randproj_adj2way"]
        out["summary"][str(L)] = {k: float(np.mean([out["per_subject"][s][str(L)][k] for s in out["per_subject"]])) for k in keys}
    out["summary"]["frac_at_L50_mean"] = float(np.mean([out["per_subject"][s]["frac_at_L50"] for s in out["per_subject"]]))
    out["summary"]["native_efficiency_mean"] = float(np.mean([out["per_subject"][s]["native_efficiency"] for s in out["per_subject"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--decoders", default="raw,ens,fmtok,eegpt,labram")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    parts = a.part.split(",")
    res = {"smoke": a.smoke}
    if "1" in parts:
        res["part1_category_null"] = part1(a.smoke)
    if "2" in parts:
        res["part2_leakage"] = part2()
    if "3" in parts:
        res["part3_temporal_null"] = part3(a.decoders.split(","), a.smoke)
    if "4" in parts:
        res["part4_existing_controls"] = part4()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("->", out)


if __name__ == "__main__":
    main()
