"""Two controls asked for by the round-2 blind review, per subject, embedding input (GPU if visible):
  (1) target-rank control: bottleneck on the CLIP image embeddings THEMSELVES (no EEG). Project the 200 test targets and the
      train targets onto their own top-L principal directions and run the 200-way identification with the projected targets
      as queries. If this is already high at small L, "few dimensions suffice" is a property of the target set, not of EEG.
  (2) random-projection control: replace the PCA basis for the EEG embedding by a random orthonormal L-subspace (3 seeds),
      then ridge as usual. If random directions do as well as PCA, the EEG information is spread, not concentrated.
Writes results/eegbits/controls_<subject>.json. Also computes the normalised efficiency numbers for the visual target:
native yield / log2(200).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run import EMB, CLIP, load_torch, metrics, ridge_predict, pick_alpha, pca_scores

LS = [2, 5, 10, 25, 50, 100, 250, 1024]


def main(subject):
    Ytr, Yte = load_torch(CLIP / "ViT-H-14_features_train.pt"), load_torch(CLIP / "ViT-H-14_features_test.pt")
    Xtr = load_torch(EMB / f"ATM_S_eeg_features_{subject}_train.pt").reshape(len(Ytr), 4, -1).mean(1)
    Xte = load_torch(EMB / f"ATM_S_eeg_features_{subject}_test.pt")
    rows = []
    # (1) target-rank control: identification with rank-L-projected targets as queries (perfect decoder, L-dim target)
    mu = Ytr.mean(0); U, S, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    for L in LS:
        P = Vt[:L].T @ Vt[:L]
        q = (Yte - mu) @ P + mu
        rows.append(dict(condition="clip_rank_L_perfect_decoder", L=L, seed=0, **metrics(q, Yte)))
    # (2) random orthonormal projection of the standardised EEG embedding, then ridge
    rng = np.random.default_rng(0)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr_full, Zte_full = (Xtr - mu_x) / sd_x, (Xte - mu_x) / sd_x
    for seed in range(3):
        r = np.random.default_rng(10 + seed)
        Q, _ = np.linalg.qr(r.standard_normal((Ztr_full.shape[1], Ztr_full.shape[1])).astype(np.float32))
        for L in LS:
            Ztr, Zte = Ztr_full @ Q[:, :L], Zte_full @ Q[:, :L]
            alpha = pick_alpha(Ztr, Ytr, rng)
            rows.append(dict(condition="eeg_random_projection", L=L, seed=seed, alpha=alpha, **metrics(ridge_predict(Ztr, Ytr, Zte, alpha), Yte)))
    native = metrics(Xte, Yte)
    out = dict(subject=subject, rows=rows, native=native, target_entropy_bits=float(np.log2(200)),
               native_efficiency=native["bits"] / float(np.log2(200)))
    Path("/path/to/eegaudit/results/eegbits").mkdir(exist_ok=True, parents=True)
    Path(f"/path/to/eegaudit/results/eegbits/controls_{subject}.json").write_text(json.dumps(out, indent=1))
    c = {r["L"]: round(100 * r["two_way"], 1) for r in rows if r["condition"].startswith("clip")}
    rp = {L: round(100 * np.mean([r["two_way"] for r in rows if r["condition"] == "eeg_random_projection" and r["L"] == L]), 1) for L in LS}
    print(f"{subject} target-rank control 2-way by L: {c}\n{subject} random-projection EEG 2-way by L: {rp}\n{subject} native efficiency {100*out['native_efficiency']:.1f}% of log2(200)")


if __name__ == "__main__":
    for s in (sys.argv[1:] or [f"sub-{i:02d}" for i in range(1, 11)]):
        main(s)
