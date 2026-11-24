"""Split-half EEG-to-EEG identification WITHOUT a trained read-out (smoke): does the EEG identify its own image before any encoder is
trained on it?  For each subject: repetitions 0-39 vs 40-79 averaged; identify image A among the 200 B halves by correlation
(top-1, 2-way, Wolpaw bits) in three spaces: raw voltage (z-scored 63x250), a randomly initialised NICE-GA embedding (seed 0),
the trained NICE-GA embedding (seed 141). Also k = 1 vs 40-rep half (5 draws). Shuffled-pair floor for each.
Output: results/eegbits/splithalf_<subject>.json ; usage: python smoke_splithalf.py sub-01"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import metrics
import run_acq_nice as N
from run_acq_controls import load_test
subject = sys.argv[1]; dev = torch.device("cuda"); t0 = time.time()
X = np.asarray(load_test(subject), dtype=np.float32)                 # (200, 80, 63, 250)
rng = np.random.default_rng(0)
Enc, Proj = N.load_classes("nice_ga")
torch.manual_seed(0); enc_r, proj_r = Enc().to(dev).eval(), Proj().to(dev).eval()
enc_t, proj_t = Enc().to(dev).eval(), Proj().to(dev).eval()
tag = f"nice_ga_dgx_v2_{subject.replace('-', '')}_seed141"
enc_t.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
proj_t.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)


@torch.no_grad()
def emb(x, enc, proj):
    return proj(enc(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).float().cpu().numpy()


def raw(x):
    f = x.reshape(len(x), -1); return (f - f.mean(0)) / (f.std(0) + 1e-6)


spaces = {"raw_voltage": raw, "random_init_encoder": lambda x: emb(x, enc_r, proj_r), "trained_encoder": lambda x: emb(x, enc_t, proj_t)}
A, B = X[:, :40].mean(1), X[:, 40:].mean(1)
rows = []
for name, f in spaces.items():
    fa, fb = f(A), f(B)
    m = metrics(fa, fb); rows.append(dict(space=name, k=40, condition="eeg", **m))
    rows.append(dict(space=name, k=40, condition="shuffled_pair", **metrics(fa, fb[rng.permutation(200)])))
    for d in range(5):
        idx = rng.choice(40, 1); rows.append(dict(space=name, k=1, condition="eeg", draw=d, **metrics(f(X[:, idx].mean(1)), fb)))
out = dict(subject=subject, note="split-half EEG-to-EEG identification, 200-way, correlation/cosine in three spaces; no read-out trained on this subject's test data (the trained encoder was trained on the training images only)", rows=rows, runtime_s=round(time.time() - t0, 1))
Path(f"/path/to/eegaudit/results/eegbits/splithalf_{subject}.json").write_text(json.dumps(out, indent=1))
print(subject, {r["space"]: (round(100 * r["top1"], 1), round(100 * r["two_way"], 1), round(r["bits"], 2)) for r in rows if r["k"] == 40 and r["condition"] == "eeg"}, "| k=1 top-1:", {s: round(100 * np.mean([r["top1"] for r in rows if r["space"] == s and r["k"] == 1]), 1) for s in spaces}, "| pair floor top-1:", {r["space"]: round(100 * r["top1"], 1) for r in rows if r["condition"] == "shuffled_pair"}, f"({out['runtime_s']}s)", flush=True)
