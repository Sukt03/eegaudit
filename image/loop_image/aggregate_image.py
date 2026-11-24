"""Tables for S2 (NICE encoding order), S3 (ATM-S input bottleneck vs NICE-GA r1) and S4 (occpar16 / w04 retrains, 6 subjects) from
the files the smokes wrote. Writes results/loop/image/s{2,3,4}_table.md.  Usage: python aggregate_image.py s2|s3|s4|all"""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path("/path/to/eegaudit/results/loop/image"); MP = Path("/path/to/eegaudit/results/method_pilots")
EB = Path("/path/to/eegaudit/results/eegbits")
what = sys.argv[1] if len(sys.argv) > 1 else "all"


def s2():
    files = sorted(OUT.glob("s2_order_nice_sub-*_seed141.json"))
    if not files:
        return
    L = ["# S2: NICE (standard, seed 141): average-then-encode vs encode-then-average", "", "| subject | k | avg->enc Wolpaw | enc->avg Wolpaw | diff | avg->enc soft | enc->avg soft | diff | shuffled-pair soft floor |", "|---|---|---|---|---|---|---|---|---|"]
    acc = {}
    for f in files:
        v = json.loads(f.read_text())
        for k in (8, 80):
            a = [r for r in v["rows"] if r["k"] == k and r["order"] == "avg_then_encode"]; b = [r for r in v["rows"] if r["k"] == k and r["order"] == "encode_then_avg"]
            aw, bw = np.mean([r["bits"] for r in a]), np.mean([r["bits"] for r in b]); as_, bs = np.mean([r["soft_bits"] for r in a]), np.mean([r["soft_bits"] for r in b])
            fl = np.mean([r["soft_bits_shuffled_pair"] for r in a + b])
            L.append(f"| {v['subject']} | {k} | {aw:.3f} | {bw:.3f} | {bw-aw:+.3f} | {as_:.3f} | {bs:.3f} | {bs-as_:+.3f} | {fl:.2f} |")
            acc.setdefault(k, []).append((bw - aw, bs - as_, aw, as_))
    L.append("")
    for k in (8, 80):
        d = np.array(acc[k]); L.append(f"mean over {len(d)} subjects, k={k}: enc->avg minus avg->enc = {d[:,0].mean():+.3f} Wolpaw bits ({d[:,0].min():+.3f}..{d[:,0].max():+.3f}), {d[:,1].mean():+.3f} soft bits ({d[:,1].min():+.3f}..{d[:,1].max():+.3f}); relative to avg->enc: {100*d[:,0].mean()/d[:,2].mean():+.0f} % Wolpaw, {100*d[:,1].mean()/d[:,3].mean():+.0f} % soft")
    L.append("Reference (METHOD_ITERATIONS round 2, NICE-GA, 6 files): -0.070 Wolpaw / -0.26 soft at k=8; -0.089 / -0.26 at k=80.")
    (OUT / "s2_table.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def s3():
    f = OUT / "s3_atm_input_bottleneck_sub-08.json"
    if not f.exists():
        return
    v = json.loads(f.read_text()); R = v["rows"]
    full = [r for r in R if r["L"] == "full" and r["k"] == 80 and r["condition"] == "eeg"][0]
    full1 = [r for r in R if r["L"] == "full" and r["k"] == 1]
    # NICE-GA r1 files with single-trial own-subject basis (k=80), for the comparison column
    ga = {}
    for g in MP.glob("r1_input_bottleneck_sub-*_basis-sub-*.json"):
        d = json.loads(g.read_text())
        if d["basis_subject"] != d["subject"] or "smoke" in g.name:
            continue
        F = [r for r in d["rows"] if r["L"] == "full" and r["k"] == 80 and r["condition"] == "eeg"][0]
        for r in d["rows"]:
            if r["k"] == 80 and r["condition"] == "eeg" and r["L"] != "full":
                ga.setdefault(int(r["L"]), []).append((r["bits"] / F["bits"] if F["bits"] > 0 else np.nan, r["soft_bits"] / F["soft_bits"]))
    L = [f"# S3: input-side PCA bottleneck through frozen ATM-S (sub-08), single-trial basis; full k=80: top-1 {full['top1']:.3f}, Wolpaw {full['bits']:.2f}, soft {full['soft_bits']:.2f} bits; full k=1 (3 draws): Wolpaw {np.mean([r['bits'] for r in full1]):.3f}, soft {np.mean([r['soft_bits'] for r in full1]):.3f}", "",
         "| L | var. expl. | k=80 top-1 | Wolpaw bits | share W | soft bits | share soft | k=1 share W | k=1 share soft | shuffled-pair soft | noise soft | NICE-GA r1 share W (n) | NICE-GA r1 share soft |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for Lr in v["Ls"]:
        e = [r for r in R if r["L"] == Lr and r["k"] == 80 and r["condition"] == "eeg"][0]
        sp = [r for r in R if r["L"] == Lr and r["k"] == 80 and r["condition"] == "shuffled_pair"][0]
        nz = [r for r in R if r["L"] == Lr and r["k"] == 80 and r["condition"] == "noise"][0]
        e1 = [r for r in R if r["L"] == Lr and r["k"] == 1]
        sw, ss = e["bits"] / full["bits"], e["soft_bits"] / full["soft_bits"]
        sw1 = np.mean([r["bits"] for r in e1]) / np.mean([r["bits"] for r in full1]); ss1 = np.mean([r["soft_bits"] for r in e1]) / np.mean([r["soft_bits"] for r in full1])
        g = np.array(ga.get(Lr, [])); gs = f"{np.nanmean(g[:,0]):.2f} ({len(g)})" if len(g) else "-"; gs2 = f"{np.nanmean(g[:,1]):.2f}" if len(g) else "-"
        out[Lr] = dict(var=e["var_explained"], top1=e["top1"], bits=e["bits"], soft=e["soft_bits"], share_w=sw, share_soft=ss, k1_share_w=sw1, k1_share_soft=ss1, shuf_soft=sp["soft_bits"], noise_soft=nz["soft_bits"], nicega_share_w=float(np.nanmean(g[:,0])) if len(g) else None, nicega_share_soft=float(np.nanmean(g[:,1])) if len(g) else None)
        L.append(f"| {Lr} | {e['var_explained']:.3f} | {e['top1']:.3f} | {e['bits']:.2f} | {sw:.2f} | {e['soft_bits']:.2f} | {ss:.2f} | {sw1:.2f} | {ss1:.2f} | {sp['soft_bits']:.2f} | {nz['soft_bits']:.2f} | {gs} | {gs2} |")
    l90w = min([Lr for Lr in v["Ls"] if out[Lr]["share_w"] >= 0.9], default=">1000"); l90s = min([Lr for Lr in v["Ls"] if out[Lr]["share_soft"] >= 0.9], default=">1000")
    L += ["", f"Smallest L with 90 % of the full yield (ATM-S, k=80): Wolpaw {l90w}, soft {l90s}. NICE-GA (r1, own single-trial basis, k=80, mean over subjects) shown in the last two columns."]
    (OUT / "s3_table.md").write_text("\n".join(L) + "\n"); (OUT / "s3_table.json").write_text(json.dumps(dict(full=full, by_L=out, L90_wolpaw=l90w, L90_soft=l90s), indent=1)); print("\n".join(L))


def s4():
    L = ["# S4: 16-electrode and 0.4-s retrains (NICE-GA, seed 141), share of the full-input retrain-matched reference (same subject, seed 141 standard NICE-GA; from s1_full files)", "", "| variant | subject | k=80 top-1 | Wolpaw bits | share W | soft bits | share soft | k=1 Wolpaw share | k=1 soft share | noise top-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for v in ("occpar16", "w04"):
        for f in sorted(EB.glob(f"retrain_{v}_sub-*_seed141.json")):
            d = json.loads(f.read_text()); s = d["subject"]; full = OUT / f"s1_full_{s}_seed141.json"
            if not full.exists() or not all("soft_bits" in r for r in d["rows"]):
                var = OUT / f"s1_var_{v}_{s}_seed141.json"
                if not (full.exists() and var.exists()):
                    continue
                d = json.loads(var.read_text()); d["rows"] = [dict(axis="native" if r["k"] == 80 and r.get("condition") != "noise" else ("control" if r.get("condition") == "noise" else "repetitions"), **r) for r in d["rows"]]
            F = json.loads(full.read_text())
            n = [r for r in d["rows"] if r["k"] == 80 and r.get("condition") != "noise"][0]; r1 = [r for r in d["rows"] if r["k"] == 1]
            fn = [r for r in F["rows"] if r["k"] == 80 and r.get("condition") != "noise"][0]; f1 = [r for r in F["rows"] if r["k"] == 1]
            nz = [r for r in d["rows"] if r.get("condition") == "noise"]
            rec = dict(top1=n["top1"], bits=n["bits"], soft=n["soft_bits"], share_w=n["bits"] / fn["bits"], share_soft=n["soft_bits"] / fn["soft_bits"],
                       k1_share_w=np.mean([r["bits"] for r in r1]) / np.mean([r["bits"] for r in f1]), k1_share_soft=np.mean([r["soft_bits"] for r in r1]) / np.mean([r["soft_bits"] for r in f1]), noise_top1=nz[0]["top1"] if nz else None)
            rows.setdefault(v, {})[s] = rec
            L.append(f"| {v} | {s} | {rec['top1']:.3f} | {rec['bits']:.2f} | {rec['share_w']:.2f} | {rec['soft']:.2f} | {rec['share_soft']:.2f} | {rec['k1_share_w']:.2f} | {rec['k1_share_soft']:.2f} | {rec['noise_top1'] if rec['noise_top1'] is None else round(rec['noise_top1'],3)} |")
    L.append("")
    for v, R in rows.items():
        a = np.array([[r["share_w"], r["share_soft"], r["k1_share_w"], r["k1_share_soft"]] for r in R.values()])
        L.append(f"{v}: n={len(R)} subjects ({', '.join(sorted(R))}); k=80 share Wolpaw {a[:,0].mean():.2f} ({a[:,0].min():.2f}-{a[:,0].max():.2f}), soft {a[:,1].mean():.2f} ({a[:,1].min():.2f}-{a[:,1].max():.2f}); k=1 share Wolpaw {a[:,2].mean():.2f}, soft {a[:,3].mean():.2f}")
    (OUT / "s4_table.md").write_text("\n".join(L) + "\n"); (OUT / "s4_table.json").write_text(json.dumps(rows, indent=1)); print("\n".join(L))


{"s2": s2, "s3": s3, "s4": s4}.get(what, lambda: (s2(), s3(), s4()))()
