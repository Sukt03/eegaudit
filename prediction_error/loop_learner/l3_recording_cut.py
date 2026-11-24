"""L3 (ideation C2): recording cut on the learner. The raw-ERP read-out (results/features/raw: 64 channels x 23 samples,
150-600 ms, channel-major columns, PCA-64 ridge within participant) is recomputed from the 32 occipito-parietal electrodes
of the image track (run_acq_nice.OCC_PAR) and, as the control, from the complementary 32 anterior electrodes; same target,
same exact null, same statistics, paired differences; the EEG increment in bits at each coupling from the exact curve.
EEGNet retraining on 32 channels is not run here (23 LOPO folds exceed the 20-minute budget)."""
import json, time, numpy as np
import sys; sys.path.insert(0, "/path/to/eegaudit/experiments/loop/learner"); from common import *
t0 = time.time(); OUT = ROOT / "results/loop/learner"; OUT.mkdir(parents=True, exist_ok=True)
OCC_PAR = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
           "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP9", "TP10", "C5", "C3", "C1", "Cz"]
E = ROOT / "results/eeg_measures"; common = None
for s in range(1, 24):
    c = [str(x) for x in np.load(E / f"sub-{s:03d}_epochs.npz")["ch"]]
    common = c if common is None else [x for x in common if x in c]
times = np.load(E / "sub-001_epochs.npz")["times"]; nt = int(((times >= 0.15) & (times <= 0.60)).sum())
F0 = load_feat("raw", 1)[0]; assert F0.shape[1] == len(common) * nt, (F0.shape, len(common), nt)
post = [k for k, ch in enumerate(common) if ch in OCC_PAR]; ant = [k for k in range(len(common)) if k not in post]
assert len(post) == 32, len(post)
cols = {"all64": None, "posterior32": np.concatenate([np.arange(k * nt, (k + 1) * nt) for k in post]), "anterior32": np.concatenate([np.arange(k * nt, (k + 1) * nt) for k in ant])}
incr, rmax = make_incr(ROOT / "results/ceiling_RWD_v2/results.json")
res = {"decoder": "raw (64 ch x %d samples 150-600 ms, PCA-64 ridge, 10-fold within participant)" % nt, "posterior32": [common[k] for k in post], "anterior32": [common[k] for k in ant],
       "ceiling_curve": "results/ceiling_RWD_v2/results.json", "sets": {}}
C = {}
for name, cc in cols.items():
    C[name] = coupling("raw", "pe_resid", cc); c = strip(C[name])
    c["increment_bits"] = incr(c["rho_eff_pooled"]); c["increment_bits_ci"] = [incr(x) for x in c["rho_eff_ci"]]
    c["efficiency_pct_at_0.9"] = 100 * c["increment_bits"] / incr(0.9); c["n_columns"] = int(F0.shape[1] if cc is None else len(cc))
    res["sets"][name] = c
    print(f"raw/{name}: rho_eff {c['rho_eff_pooled']:.3f} [{c['rho_eff_ci'][0]:.3f}, {c['rho_eff_ci'][1]:.3f}] increment {c['increment_bits']:.3f} bits "
          f"[{c['increment_bits_ci'][0]:.3f}, {c['increment_bits_ci'][1]:.3f}] gamma0 {c['prevalence']['gamma0']:.2f} ({time.time() - t0:.0f}s)", flush=True)
res["pairs"] = {"all64-posterior32": paired(C["all64"], C["posterior32"]), "all64-anterior32": paired(C["all64"], C["anterior32"]), "posterior32-anterior32": paired(C["posterior32"], C["anterior32"])}
for k, p in res["pairs"].items(): print(f"paired {k}: dz {p['mu']:.3f} [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] p {p['p']:.2g}")
res["eegnet_32ch"] = "not run (23 LOPO retrainings exceed the 20-minute budget)"; res["runtime_s"] = round(time.time() - t0, 1)
(OUT / "l3_recording_cut.json").write_text(json.dumps(res, indent=1)); print("done", res["runtime_s"])
