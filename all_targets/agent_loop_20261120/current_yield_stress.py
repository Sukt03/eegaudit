"""Retrospective sensitivity audits on cached participant results."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results"
GPT = ROOT / "outputs/AGENT_LOOP_20261120/results"


def read(path):
    return json.loads(path.read_text())


def crossing(points, target):
    ordered = sorted(points.items())
    previous_x, previous_y = ordered[0]
    if previous_y >= target:
        return previous_x
    for x, y in ordered[1:]:
        if y >= target:
            return previous_x + (x - previous_x) * (target - previous_y) / (y - previous_y)
        previous_x, previous_y = x, max(previous_y, y)
    raise ValueError("Unreachable threshold")


def by_width(rows, field):
    eligible = [r for r in rows if r["condition"] == "eeg" and r["k"] == 80]
    return {width: mean(r[field] for r in eligible if r["L"] == width)
            for width in sorted({r["L"] for r in eligible})}


def audit_q8():
    ratios = {name: {threshold: [] for threshold in (0.8, 0.9, 0.95)}
              for name in ("embedding", "trainedenc")}
    widths = []
    paired = {width: [] for width in (100, 250, 500)}
    for path in sorted((GPT / "q8").glob("q8_sub-??.json")):
        d = read(path)
        assert hashlib.sha256(Path(d["pca_source"]).read_bytes()).hexdigest() == d["pca_source_sha256"]
        curve = by_width(d["pca_rows"], "soft_bits")
        curve[15750] = d["full_width"]["soft_bits"]
        widths.append(crossing(curve, 0.9 * curve[15750]))
        for width, diffs in paired.items():
            control, = [r for r in d["rows"] if r["L"] == width and r["k"] == 80
                        and r["condition"] == "random_projection_eeg"]
            diffs.append(curve[width] - control["soft_bits"])
        input_curve = by_width(d["pca_rows"], "bits")
        input_curve[15750] = d["full_width"]["bits"]
        for name, thresholds in ratios.items():
            rep = DATA / f"eegbits/{name}_{d['subject']}.json"
            if not rep.exists():
                continue
            rep_curve = by_width(read(rep)["rows"], "bits")
            full = rep_curve[max(rep_curve)]
            assert full > 0 and input_curve[15750] > 0
            for fraction, participants in thresholds.items():
                top = crossing(input_curve, fraction * input_curve[15750])
                bottom = crossing(rep_curve, fraction * full)
                participants.append({"subject": d["subject"], "input": top, "representation": bottom, "ratio": top / bottom})
    assert len(widths) == 10
    rng = np.random.default_rng(20261120)
    medians = np.median(rng.choice(widths, (10000, len(widths))), axis=1)
    interval = np.quantile(medians, [0.025, 0.975]).tolist()
    # Two-sided exact sign tests. All three contrasts have the same number of positives.
    assert all(len(v) == 10 and min(v) > 0 for v in paired.values())
    return dict(input_nce_L90_median=median(widths), bootstrap95_median=interval,
                octave_ratio=interval[1]/interval[0], holm_sign_p=3*2/2**10,
                mean_pca_minus_random={k:mean(v) for k,v in paired.items()}, wolpaw_ratios=ratios,
                median_ratios={name:{k:median(p["ratio"] for p in v) for k,v in sets.items()} for name,sets in ratios.items()})


def audit_q7():
    expected = {("eeg", 1, i) for i in range(5)} | {(c,80,0) for c in ("eeg","noise","trial_shuffled","shuffled_pair")}
    by_subject = {}
    for p in sorted((GPT / "q7").glob("*sub-??.json")):
        d = read(p)
        assert d["subject"] == p.stem[-6:] and d["model"] == "reve" and d["slice"] == 0 and d["n_train"] == 16540
        assert len(d["rows"]) == len(expected)
        assert {(r["condition"],r["k"],r["draw"]) for r in d["rows"]} == expected
        assert all(math.isfinite(r["soft_bits"]) for r in d["rows"])
        v = {r["condition"]:r["soft_bits"] for r in d["rows"] if r["k"] == 80}
        controls = [v[x] for x in ("noise","shuffled_pair","trial_shuffled")]
        by_subject[d["subject"]] = dict(margin_over_largest_control=v["eeg"]-max(controls),
                                       max_abs_control=max(map(abs,controls)), scores=v)
    return by_subject


def modalities():
    image = read(DATA/"loop/s3_fraction/s3_fraction_k40.json")["rows"]
    fraction = {k:[] for k in ("I_F","I_NCE")}
    for row in image:
        for metric, vals in fraction.items():
            q = (row["yield_k40"][metric]-row["floor"][metric])/(row["ceiling_k40"][metric]-row["floor_ceiling"][metric])
            assert abs(q-row["fraction"][metric]) < 1e-6
            vals.append(q)
    text = {}
    for task in ("SR","NR2"):
        d = read(DATA/f"loop/x6_text_bottleneck/x6_{task}.json")
        row, = [r for r in d["curve"] if r["basis"] == "pca" and r["L"] == 840]
        vals = [r["eeg"]["nce"]-r["noise"]["nce"] for r in row["per_participant"].values()]
        text[task] = dict(n=len(vals), mean_eeg_minus_noise_nce=mean(vals), reference_fraction=None)
    behavior = read(DATA/"loop/p6/p6_pe_partial.json")
    return dict(image_fraction_at_k40={m:mean(v) for m,v in fraction.items()},
                image_participants_over_quarter={m:sum(x>=0.25 for x in v) for m,v in fraction.items()},
                image_n=len(image),text=text, behavior_partial={name:dict(n=len(d["rows"]),
                    mean=d and mean(r["all"]["bits_partial"] for r in d["rows"]),
                    above_own_nuisance_floor=sum(r["all"]["above_floor"] for r in d["rows"])) for name,d in behavior.items()},
                pooled_price=None)


if __name__ == "__main__":
    result = dict(q7=audit_q7(),q8=audit_q8(),cross_modality=modalities())
    path=GPT/"current_yield_stress.json"
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)
    print(path, flush=True)
