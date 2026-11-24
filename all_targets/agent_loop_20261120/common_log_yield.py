"""Score three EEG target types by log predictive gain over a named reference."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from i2_complementarity import atomic_json, folds, targets

DATA = ROOT / "results"
OUT = ROOT / "outputs/AGENT_LOOP_20261120/results/common_log_yield.json"


def density_gain(y, x, fold):
    """Held-out Gaussian log-density ratio; both variances fit on training rows."""
    scores = np.full(len(y), np.nan)
    for block in np.unique(fold):
        train = fold != block
        test = ~train
        design = np.column_stack([np.ones(len(x)), x])
        coeff = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        ref_mean = y[train].mean()
        prediction = design @ coeff
        residual_var = np.mean((y[train] - prediction[train])**2)
        ref_var = np.mean((y[train] - ref_mean)**2)
        assert min(residual_var, ref_var) > 0
        scores[test] = (np.log(ref_var / residual_var) + (y[test]-ref_mean)**2/ref_var
                        - (y[test]-prediction[test])**2/residual_var) / (2*np.log(2))
    assert np.isfinite(scores).all()
    return float(scores.mean())


def main():
    evaluate, participants, fpath, _ = targets()
    ds = {}
    for model in ("eegnet","reve_lopo"):
        rows = []
        for subject in participants:
            with np.load(fpath(model,subject)) as cache:
                x = cache["F"].astype(float)
                trial = cache["tri"]
            assert x.shape == (len(trial),1) and (np.diff(trial)>0).all()
            y = evaluate(subject,trial)
            fold = folds(len(y))
            gain = density_gain(y,x,fold)
            units = density_gain(100*y,x,fold)
            perm = np.random.default_rng(20261120+subject).permutation(len(y))
            assert abs(gain-units) < 1e-9
            rows.append(dict(subject=subject, trials=len(y), gain=gain, shuffled=density_gain(y,x[perm],fold),
                             unit_error=abs(gain-units)))
        ds[model] = rows
    image = json.loads((DATA/"loop/s3_fraction/s3_fraction_k40.json").read_text())
    text = {}
    candidate_counts = {}
    for task in ("SR","NR2"):
        d = json.loads((DATA/f"loop/x6_text_bottleneck/x6_{task}.json").read_text())
        candidate_counts[task] = d["N"]
        cond, = [r for r in d["curve"] if r["basis"]=="pca" and r["L"]==840]
        text[task] = {subject:{k:r[k]["nce"] for k in ("eeg","noise","shuffled_pair")}
                      for subject,r in cond["per_participant"].items()}
    report = dict(unit="log2 probability (categorical) or density (continuous) gain per evaluated query",
                  allocation=os.getenv("SLURM_JOB_ID"), step=os.getenv("SLURM_STEP_ID"),
                  image={"N":200,"k":40,"rows":{r["subject"]:{"gain":r["yield_k40"]["I_NCE"],"shuffled":r["floor"]["I_NCE"]} for r in image["rows"]}},
                  sentence={"candidate_counts":candidate_counts,"rows":text}, behavioral={"target":"historical globally residualized PE per feedback trial","rows":ds},
                  calibration="Historical retrieval calibration procedures differ. New density calibration uses train-only blocked folds within each participant.",
                  interpretation="Model-dependent held-out predictive gain relative to the declared baseline. No universal EEG constant, information upper bound, or pooled price calculated.")
    atomic_json(OUT,report)
    print(OUT)
if __name__=="__main__":
    main()
