"""ATM's native zero-shot retrieval (no probe): cosine between the ATM test EEG embedding and the 200 CLIP test embeddings,
per subject. This is the published method's own full-width rung of the visual ladder. Writes results/eegbits/native_retrieval.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run import EMB, CLIP, load_torch, metrics
Yte = load_torch(CLIP / "ViT-H-14_features_test.pt")
out = {}
for i in range(1, 11):
    s = f"sub-{i:02d}"
    Xte = load_torch(EMB / f"ATM_S_eeg_features_{s}_test.pt")
    out[s] = metrics(Xte, Yte)
m = {k: float(np.mean([v[k] for v in out.values()])) for k in ["top1", "top5", "two_way", "mrr", "bits"]}
json.dump(dict(per_subject=out, mean=m, method="ATM-S embedding -> cosine nearest CLIP ViT-H/14 image embedding, 200-way, 80-rep test average"), open("results/eegbits/native_retrieval.json", "w"), indent=1)
print("ATM native retrieval, mean over 10 subjects:", {k: round(v, 3) for k, v in m.items()})
print({s: round(100 * v["top1"], 1) for s, v in out.items()})
