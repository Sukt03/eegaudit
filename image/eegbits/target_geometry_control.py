"""No-EEG target-geometry control for the target swap (ideation soundness check, gap A2): in each target space (CLIP test features,
DINOv2, pixel-PCA), identify each of the 200 test images from a noisy copy of its own embedding, y = (e + sigma * eps) with eps
standard normal, at a sequence of noise levels; report Wolpaw bits vs sigma per space, and the bits in DINOv2 / pixel space at the sigma
that reproduces the CLIP arm's observed yield (0.72 bits). If the spaces differ at matched noise, part of the target-swap effect is
geometry (how the 200 images are spread), not the EEG. 200 draws per sigma. Output: results/eegbits/target_geometry_control.json"""
import json
from pathlib import Path
import numpy as np, sys
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import metrics
R = Path("/path/to/eegaudit/results/eegbits"); NICE = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature")
spaces = {"clip": np.squeeze(np.load(NICE / "clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32),
          "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_test.npy"), "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_test.npy")}
rng = np.random.default_rng(0); sigmas = np.logspace(-0.5, 1.8, 24); out = {"sigmas": sigmas.tolist(), "bits": {}, "top1": {}}      # noise norm relative to a unit-norm embedding; out = {"sigmas": sigmas.tolist(), "bits": {}, "top1": {}}
for name, E in spaces.items():
    E = E / np.linalg.norm(E, axis=1, keepdims=True); bits, top1 = [], []
    for sg in sigmas:
        b = [metrics(E + sg * rng.standard_normal(E.shape).astype(np.float32) / np.sqrt(E.shape[1]), E) for _ in range(50)]
        bits.append(float(np.mean([m["bits"] for m in b]))); top1.append(float(np.mean([m["top1"] for m in b])))
    out["bits"][name] = bits; out["top1"][name] = top1
# matched-noise comparison: sigma at which CLIP reproduces the observed CLIP yield
clip_obs = json.load(open(R / "retrain_summary.json"))["clip_testfeat"]["k80_bits"]["mean"]
cb = np.array(out["bits"]["clip"]); i = int(np.argmin(np.abs(cb - clip_obs)))
out["matched"] = {"clip_observed_bits": clip_obs, "sigma": float(sigmas[i]), "clip_bits_at_sigma": float(cb[i]), "dinov2_bits_at_sigma": out["bits"]["dinov2"][i], "pixpca_bits_at_sigma": out["bits"]["pixpca"][i]}
# also report the geometry: mean nearest-neighbour cosine and mean off-diagonal cosine per space
for name, E in spaces.items():
    En = E / np.linalg.norm(E, axis=1, keepdims=True); S = En @ En.T; np.fill_diagonal(S, -np.inf)
    out.setdefault("geometry", {})[name] = {"mean_nn_cosine": float(np.max(S, axis=1).mean()), "mean_offdiag_cosine": float(S[np.isfinite(S)].mean())}
(R / "target_geometry_control.json").write_text(json.dumps(out, indent=1))
print("matched noise:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["matched"].items()}); print("geometry:", out["geometry"])
