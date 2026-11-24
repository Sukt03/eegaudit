"""Group summary of results/gpu_fill_20261123/ubp_heldout/ubp_card.json (held-out-concept UBP retrain, eeg arm, k80):
participant means of A (test-query temperature) and B' (held-out-concept temperature), their gap, participant-bootstrap CI (10000, seed 1),
and the count of unchanged participants. Same computation as the independent review's recompute. Writes summary.json next to the input."""
import json, hashlib, numpy as np
from pathlib import Path
D = Path("/path/to/eegaudit/results/gpu_fill_20261123/ubp_heldout")
U = json.load(open(D / "ubp_card.json"))
rows = [v for v in U["retrain"].values() if v["arm"] == "eeg"]; P = sorted({v["participant"] for v in rows})
A = np.array([np.mean([v["bits_A_k80"] for v in rows if v["participant"] == p]) for p in P])
B = np.array([np.mean([v["bits_Bprime_k80"] for v in rows if v["participant"] == p]) for p in P])
gap = A - B; bs = np.random.default_rng(1).choice(gap, (10000, len(gap))).mean(1)
q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
out = {"NOT_CITABLE": "until independent review (see review: cite the mean, 9 of 10 unchanged)",
       "n_participants": len(P), "A_k80": {"mean": float(A.mean())}, "Bprime_k80": {"mean": float(B.mean())},
       "gap": {"mean": float(gap.mean()), "ci95": q(bs)}, "inflation_pct": {"mean": float(100 * gap.mean() / B.mean()), "ci95": [float(100 * x / B.mean()) for x in q(bs)]},
       "n_unchanged": int((np.abs(gap) < 1e-5).sum()), "source": "ubp_card.json", "source_code_sha256": U.get("code_sha256"),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
json.dump(out, open(D / "summary.json", "w"), indent=1); print({k: out[k] for k in ("n_participants", "n_unchanged")}, round(out["A_k80"]["mean"], 3), round(out["Bprime_k80"]["mean"], 3), [round(x, 3) for x in out["gap"]["ci95"]])
