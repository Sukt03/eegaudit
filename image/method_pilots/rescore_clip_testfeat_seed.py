"""CLIP arm of the PAIRED target swap for any released NICE-GA seed (method image loop, 2026-11-20): score the released NICE-GA
(CLIP target, seed <seed>) against the CLIP features of the 200 TEST IMAGES themselves (same candidate set as the DINOv2 and pixel
arms), k = 80 and k = 1 (5 draws). Logic copied from experiments/eegbits/rescore_clip_testfeat.py; only the seed is a parameter and the
output goes to results/method_pilots so the existing aggregate's glob over results/eegbits/clip_testfeat_sub-*.json is untouched.
Usage: python rescore_clip_testfeat_seed.py sub-01 297  ->  results/method_pilots/clip_testfeat_sub-01_seed297.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import metrics
import run_acq_controls as C
from run_acq_nice import NICE
subject, seed = sys.argv[1], int(sys.argv[2]); dev = torch.device("cuda"); t0 = time.time(); rng = np.random.default_rng(seed % 1000)
embed, Ycent, tag, X = C.load_encoder("nice_ga", subject, seed, dev)
Yte = np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32)
assert Yte.shape == (200, 768), Yte.shape
rows = [dict(axis="native", candidates="test_image_features", k=80, draw=0, **metrics(embed(X.mean(1)), Yte)),
        dict(axis="native", candidates="concept_centres", k=80, draw=0, **metrics(embed(X.mean(1)), Ycent))]
for d in range(5):
    rows.append(dict(axis="repetitions", candidates="test_image_features", k=1, draw=d, **metrics(embed(X[:, rng.choice(80, 1)].mean(1)), Yte)))
out = dict(subject=subject, seed=seed, rows=rows, runtime_s=round(time.time() - t0, 1))
Path(f"/path/to/eegaudit/results/method_pilots/clip_testfeat_{subject}_seed{seed}.json").write_text(json.dumps(out, indent=1))
print(subject, f"seed{seed}", "vs test features: top-1 %.1f%% bits %.2f | vs centres: top-1 %.1f%% bits %.2f" % (100*rows[0]["top1"], rows[0]["bits"], 100*rows[1]["top1"], rows[1]["bits"]), flush=True)
