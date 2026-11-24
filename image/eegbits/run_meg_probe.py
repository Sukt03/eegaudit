"""EEG-versus-MEG on the same target space (ideation card B2, minimal form): for THINGS-MEG (4 subjects, 271 sensors, 200 Hz, 0-1 s,
training 19,848 images x 1 repetition, test 200 images x 12 repetitions) and THINGS-EEG2 (63 channels, 250 Hz, 0-1 s), the SAME
linear read-out: standardised raw signal -> dual ridge -> DINOv2 ViT-B/14 image embedding, 200-way identification on the official test
images; plus each modality's own raw split-half ceiling (half the repetitions vs the other half, correlation identification) and the
noise and shuffled-pair controls. Yield = Wolpaw bits and InfoNCE soft bound (run.metrics).
Usage: python run_meg_probe.py meg sub-01 | python run_meg_probe.py eeg sub-01
Output: results/eegbits/modprobe_<modality>_<subject>.json"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics, load_raw
ROOT = Path("/path/to/eegaudit"); OUT = ROOT / "results/eegbits"
MEG = ROOT / "data_raw/things_meg/preprocessed_MEG"; MEGF = OUT / "meg_dinov2_feature"; EEGF = OUT / "dinov2_feature"
dev = torch.device("cuda")


def load_meg(subject):
    tr = pickle.load(open(MEG / subject / "preprocessed_meg_training.pkl", "rb")); te = pickle.load(open(MEG / subject / "preprocessed_meg_zs_test.pkl", "rb"))
    Xtr = np.asarray(tr["meg_data"], dtype=np.float32); Xte = np.asarray(te["meg_data"], dtype=np.float32)
    # training pickle is (1654 concepts, 12 images, 271, 201): one trial per image -> flatten to (19848, 271, 201) in concept-major,
    # image-minor order, which is the sorted image order used for the DINOv2 features
    Xtr = Xtr.reshape(-1, Xtr.shape[-2], Xtr.shape[-1])                                  # (1654, 12, 1, 271, 201) -> (19848, 271, 201)
    Xte = Xte.reshape(Xte.shape[0], -1, Xte.shape[-2], Xte.shape[-1])                   # (200, 12, 271, 201)
    return Xtr, Xte, dict(ch=len(te["ch_names"]), sfreq="200 Hz (201 samples, 0-1.0 s)")


def ridge_dual(Ftr, Ytr, Ftes, seed=0):
    fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.from_numpy((Ftr - fm) / fs).to(dev); Ym = Ytr.mean(0); Y = torch.from_numpy(Ytr - Ym).to(dev)
    n, D = A.shape; idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(dev); cut = int(0.9 * n); tr, va = idx[:cut], idx[cut:]
    K = A @ A.T; best, ba = -1, None
    for alpha in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
        coef = torch.linalg.solve(K[tr][:, tr] + alpha * D * torch.eye(len(tr), device=dev), Y[tr])
        s = metrics((K[va][:, tr] @ coef).cpu().numpy() + Ym, Ytr[va.cpu().numpy()])["mrr"]
        if s > best: best, ba = s, alpha
    coef = torch.linalg.solve(K + ba * D * torch.eye(n, device=dev), Y)
    preds = [((torch.from_numpy((F - fm) / fs).to(dev) @ A.T) @ coef).cpu().numpy() + Ym for F in Ftes]
    del K, A; torch.cuda.empty_cache(); return ba, preds


def main(mod, subject):
    t0 = time.time(); rng = np.random.default_rng(0)
    if mod == "meg":
        Xtr, Xte, meta = load_meg(subject); Ytr = np.load(MEGF / "dinov2_feature_maps_training.npy"); Yte = np.load(MEGF / "dinov2_feature_maps_test.npy")
    else:
        Xtr4, Xte, meta = load_raw(subject); Xtr = Xtr4.mean(1); del Xtr4
        Ytr = np.load(EEGF / "dinov2_feature_maps_training.npy"); Yte = np.load(EEGF / "dinov2_feature_maps_test.npy")
    assert len(Xtr) == len(Ytr), (Xtr.shape, Ytr.shape)
    nrep = Xte.shape[1]; Ftr = Xtr.reshape(len(Xtr), -1); Fte_all = Xte.mean(1).reshape(200, -1)
    Fte_1 = [Xte[:, rng.choice(nrep, 1)].mean(1).reshape(200, -1) for _ in range(5)]
    alpha, (p_all, *p1) = ridge_dual(Ftr, Ytr, [Fte_all] + Fte_1)
    rows = [dict(condition="eeg", k=nrep, draw=0, **metrics(p_all, Yte))] + [dict(condition="eeg", k=1, draw=d, **metrics(p, Yte)) for d, p in enumerate(p1)]
    rows.append(dict(condition="shuffled_pair", k=nrep, draw=0, **metrics(p_all, Yte[rng.permutation(200)])))
    sd, mu = Ftr.std(), Ftr.mean()
    _, (pn,) = ridge_dual(rng.standard_normal(Ftr.shape).astype(np.float32) * sd + mu, Ytr, [rng.standard_normal(Fte_all.shape).astype(np.float32) * sd + mu])
    rows.append(dict(condition="noise", k=nrep, draw=0, **metrics(pn, Yte)))
    # raw split-half ceiling: first half of repetitions vs second half, z-scored, correlation identification
    h = nrep // 2; A = Xte[:, :h].mean(1).reshape(200, -1); B = Xte[:, h:].mean(1).reshape(200, -1)
    z = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6)
    rows.append(dict(condition="splithalf_raw", k=h, draw=0, **metrics(z(A), z(B))))
    rows.append(dict(condition="splithalf_raw_pair", k=h, draw=0, **metrics(z(A), z(B)[rng.permutation(200)])))
    out = dict(modality=mod, subject=subject, n_train=int(len(Ftr)), feature_dim=int(Ftr.shape[1]), n_test_reps=int(nrep), alpha=alpha, target="DINOv2 ViT-B/14 CLS (768-d, L2)",
               meta={k: (v if isinstance(v, (int, str)) else str(v)) for k, v in meta.items()}, rows=rows, runtime_s=round(time.time() - t0, 1))
    (OUT / f"modprobe_{mod}_{subject}.json").write_text(json.dumps(out, indent=1))
    e, n_, s_, sh = rows[0], rows[-3], rows[-4], rows[-2]
    print(f"{mod} {subject} n={len(Ftr)} D={Ftr.shape[1]} alpha={alpha:g}: k={nrep} 2-way {100*e['two_way']:.1f}% top1 {100*e['top1']:.1f}% bits {e['bits']:.2f} soft {e.get('soft_bits', float('nan')):.2f} | noise 2-way {100*n_['two_way']:.1f}% pair {100*s_['two_way']:.1f}% | raw split-half top-1 {100*sh['top1']:.1f}% 2-way {100*sh['two_way']:.1f}%  ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
