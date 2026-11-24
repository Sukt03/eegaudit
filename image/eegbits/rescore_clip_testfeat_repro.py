"""Score the trained NICE-GA (CLIP target, seed 141) against the CLIP features of the 200 TEST IMAGES themselves (clip_feature_maps_test.npy)
instead of NICE's concept centres, so the CLIP arm of the target swap uses the same candidate set as the DINOv2 and pixel arms
(code review 2026-11-20). k = 80 and k = 1 (5 draws). Output: results/eegbits/clip_testfeat_<subject>.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics
import run_acq_controls as C
from run_acq_nice import NICE
subject = sys.argv[1]; dev = torch.device("cuda"); t0 = time.time(); rng = np.random.default_rng(141 % 1000)
embed, Ycent, tag, X = C.load_encoder("nice_ga", subject, 141, dev)
Yte = np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32)
assert Yte.shape == (200, 768), Yte.shape
rows = [dict(axis="native", candidates="test_image_features", k=80, draw=0, **metrics(embed(X.mean(1)), Yte)),
        dict(axis="native", candidates="concept_centres", k=80, draw=0, **metrics(embed(X.mean(1)), Ycent))]
for d in range(5):
    rows.append(dict(axis="repetitions", candidates="test_image_features", k=1, draw=d, **metrics(embed(X[:, rng.choice(80, 1)].mean(1)), Yte)))
out = dict(subject=subject, seed=141, rows=rows, runtime_s=round(time.time() - t0, 1))
Path(f"/path/to/eegaudit/results/eegbits/repro_check/clip_testfeat_{subject}.json").write_text(json.dumps(out, indent=1))
print(subject, "vs test features: top-1 %.1f%% bits %.2f | vs centres: top-1 %.1f%% bits %.2f" % (100*rows[0]["top1"], rows[0]["bits"], 100*rows[1]["top1"], rows[1]["bits"]), flush=True)
