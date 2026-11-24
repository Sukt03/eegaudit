"""G2-IMAGE: reproduction check of 200-way zero-shot top-1 / top-5 on THINGS-EEG2 (intra-subject test split, 200 images x 80
repetitions, all 80 averaged = k80) for our EEG-to-image models, against each paper's published per-participant numbers.
G2-IMAGE agent, 21 Nov 2026. GPU only. Existing loaders are imported unchanged:
  NICE-GA seed 141 (our retrain) and BrainFLORA (released single-EEG checkpoint): experiments/ext/s1_averaging.models() / embed()
  ATM-S (released test embeddings): experiments/eegbits/run.py EMB / CLIP / load_torch (as native_retrieval.py)
  UBP (released checkpoints): read from results/baselines/ubp (released-checkpoint top-1 reproduced by ubp_card.py) and the released
      per-subject test_results.json (top-5); no forward pass here.
Candidates:
  NICE-GA  'centres' = dnn_feature/center_clip.npy (CLIP ViT-L/14 template per test concept from THINGS images NOT shown in the EEG
           experiment; NICE paper Sec. 3.1 + repo README step 2: this IS the paper's protocol, and what S1/r1/p1 use);
           'testimg' = dnn_feature/clip_feature_maps_test.npy (CLIP ViT-L/14 of the 200 shown test images; NOT the NICE protocol).
  BrainFLORA  released ViT-H-14_features_test.pt (native candidates, as U1 / S1).
  ATM-S   ViT-H-14_features_test.pt (as native_retrieval.py).
Scores per (model, candidate set): top-1, top-5 (rank = 1 + #candidates with strictly higher cosine), plus two InfoNCE columns
for the NICE-GA k80 G2 diagnosis: 'soft12' = S1.info200 (locked 12-point LOO grid, run._TAUS) and 'ls40' = method_pilots
common.soft_bits (40-point grid, the estimator that wrote the registered r1 k80 file).
Pass rule (per model): |mean top-1 ours - published| <= 2 pp.
Usage: python g2_image_topk.py --subjects 1   |   python g2_image_topk.py --summarize
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT / "experiments/method_pilots", ROOT / "experiments/eegbits", ROOT / "experiments/ext"):
    sys.path.insert(0, str(p))
import s1_averaging as S1          # models(), embed(), info200() (imports run, m3_yield_map; unchanged)
import run as R                    # EMB, CLIP, load_torch
import common as MP                # soft_bits with the 40-point grid (r1's estimator)
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
OUT = ROOT / "results/ext/g2_image_topk"; DEV = "cuda"
NICE_DIR = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG")
UBP_D = ROOT / "data_raw/third_party/ubp"

# Published intra-subject numbers, read with the alphaXiv tool on 21 Nov 2026 (per participant 1..10, then the paper's average).
PUB = {
    "nice_ga": dict(source="Song et al., ICLR 2024, arXiv 2308.13234v3, Table 2 (subject dependent, NICE-GA)",
                    top1=[15.2, 13.9, 14.7, 17.6, 9.0, 16.4, 14.9, 20.3, 14.1, 19.6], top5=[40.1, 40.1, 42.7, 48.9, 29.7, 44.4, 43.1, 52.1, 39.7, 46.7],
                    avg_top1=15.6, avg_top5=42.8),
    "atm_s": dict(source="Li et al., NeurIPS 2024, arXiv 2403.07721v7, Table 8 (subject dependent, batch 1024; mean +- SD only)",
                  top1=None, top5=None, avg_top1=28.64, avg_top5=58.47, sd_top1=6.39, sd_top5=8.97,
                  note="per-participant values are not in the paper text; UBP (arXiv 2503.04207 Table 1) cites ATM-S as 28.5 / 60.4"),
    "brainflora": dict(source="Li et al., ACM MM 2025, arXiv 2507.09747v2",
                       avg_top1=25.35, avg_top5=57.30, setting="joint-subject BrainFLORA-uni EEG, Table 1 (150 epochs; the released "
                       "'EEG single-modality retrieval encoder' eeg_01-06_01-46_150.pth is this setting)",
                       intra_subject_avg_top1=21.85, intra_subject_avg_top5=51.65, intra_subject_source="Table 7 (in-subject, BrainFLORA-unimodal; no released checkpoint)",
                       top1=None, top5=None),
    "ubp": dict(source="Wu et al., CVPR 2025, arXiv 2503.04207v2, Table 1 (intra-subject, UBP)",
                top1=[41.2, 51.2, 51.2, 51.1, 42.2, 57.5, 49.0, 58.6, 45.1, 61.5], top5=[70.5, 80.9, 82.0, 76.9, 72.8, 83.5, 79.9, 85.8, 76.2, 88.2],
                avg_top1=50.9, avg_top5=79.7),
}


def score(E, Y):
    """E (200, d) query embeddings, Y (200, d) candidates, true on the diagonal -> per-query rank, top-1, top-5, soft12, ls40."""
    E = torch.as_tensor(np.asarray(E), dtype=torch.float64, device=DEV) if not torch.is_tensor(E) else E.to(DEV, torch.float64)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float64, device=DEV) if not torch.is_tensor(Y) else Y.to(DEV, torch.float64)
    S = S1.nrm(E) @ S1.nrm(Y).T; d = S.diagonal(); rank = (S > d[:, None]).sum(1) + 1
    soft12 = float(S1.info200(S[None])[0][0]); ls40 = MP.soft_bits(S.cpu().numpy())
    return dict(top1=float((rank == 1).double().mean()), top5=float((rank <= 5).double().mean()), soft12=soft12, ls40=ls40,
                rank=rank.cpu().tolist())


def subject(s):
    t0 = time.time(); sub = f"sub-{s:02d}"
    X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); assert X.shape == (200, 80, 63, 250)
    x80 = np.asarray(X[:, np.arange(80)].mean(1))                     # same expression as S1 k = 80 (all repetitions, one set)
    print(sub, "k80 mean loaded", f"{time.time()-t0:.0f}s", flush=True)
    MD = S1.models(sub, DEV); res = dict(participant=sub, code_sha256=CODE_SHA256, rows={})
    ft, _, Yc = MD["nice_ga"]; En = S1.embed(ft, x80, bs=40)
    Yt = np.squeeze(np.load(NICE_DIR / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float64); assert Yt.shape == Yc.shape
    res["rows"]["nice_ga_centres"] = score(En, Yc); res["rows"]["nice_ga_testimg"] = score(En, Yt)
    fb, _, Yb = MD["brainflora"]; res["rows"]["brainflora_native"] = score(S1.embed(fb, x80, bs=40), Yb)
    res["rows"]["atm_s_native"] = score(np.asarray(R.load_torch(R.EMB / f"ATM_S_eeg_features_{sub}_test.pt"), np.float64), np.asarray(R.load_torch(R.CLIP / "ViT-H-14_features_test.pt"), np.float64))
    # G2 cross-checks against existing files (same model, same construction)
    r1 = json.loads((ROOT / f"results/method_pilots/r1_input_bottleneck_{sub}_basis-{sub}.json").read_text())
    r1v = next(r for r in r1["rows"] if r["L"] == "full" and r["k"] == 80 and r["condition"] == "eeg")
    acq = next(r for r in json.loads((ROOT / f"results/eegbits/acq_controls_nice_ga_{sub}_seed141.json").read_text())["rows"] if r["k"] == 80 and r["condition"] == "eeg")
    bf = next(r for r in json.loads((ROOT / "results/standard/summary.json").read_text())["image"]["open_weight_models"]["brainflora_native"]["per_participant"] if r["subject"] == sub)
    atm = json.loads((ROOT / "results/eegbits/native_retrieval.json").read_text())["per_subject"][sub]
    nc = res["rows"]["nice_ga_centres"]; g = dict(
        nice_top1_vs_r1=nc["top1"] - r1v["top1"], nice_top1_vs_acq=nc["top1"] - acq["top1"], nice_top5_vs_acq=nc["top5"] - acq["top5"],
        nice_ls40_vs_r1_soft_bits=nc["ls40"] - r1v["soft_bits"], nice_soft12_vs_r1_soft_bits=nc["soft12"] - r1v["soft_bits"], r1_soft_bits=r1v["soft_bits"],
        brainflora_soft12_vs_registered_k80=res["rows"]["brainflora_native"]["soft12"] - bf["k80"],
        atm_top1_vs_native=res["rows"]["atm_s_native"]["top1"] - atm["top1"], atm_top5_vs_native=res["rows"]["atm_s_native"]["top5"] - atm["top5"])
    s1f = ROOT / f"results/ext/s1_v5/{sub}.json"
    if s1f.exists(): g["nice_soft12_vs_s1_v5_yield200"] = nc["soft12"] - json.loads(s1f.read_text())["rows"]["nice_ga_k80"]["yield200"]
    res["g2_identity"] = g; res["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, {k: {q: round(v[q], 4) for q in ("top1", "top5", "soft12", "ls40")} for k, v in res["rows"].items()}, flush=True)
    print(sub, "G2 identity", {k: round(v, 5) for k, v in g.items()}, f"{res['runtime_s']}s", flush=True)


def summarize():
    per = [json.loads((OUT / f"sub-{s:02d}.json").read_text()) for s in range(1, 11)]
    assert all(p["code_sha256"] == per[0]["code_sha256"] for p in per), "mixed code versions"
    ubp_r = json.loads((ROOT / "results/baselines/ubp/ubp_summary.json").read_text())["released"]
    ubp_t1 = [100 * x for x in ubp_r["k80_top1"]["per_participant"]]
    ubp_t5 = [100 * json.load(open(UBP_D / f"ckpt/sub-{s:02d}/test_results.json"))[0]["test_top5_acc"] for s in range(1, 11)]
    ubp_t1_json = [100 * json.load(open(UBP_D / f"ckpt/sub-{s:02d}/test_results.json"))[0]["test_top1_acc"] for s in range(1, 11)]
    table = {}
    def row(name, pubkey, t1, t5, protocol, extra=None):
        P = PUB[pubkey]; m1, m5 = float(np.mean(t1)), float(np.mean(t5)); d1 = m1 - P["avg_top1"]; d5 = m5 - P["avg_top5"]
        r = dict(protocol=protocol, ours_top1=[round(x, 2) for x in t1], ours_top5=[round(x, 2) for x in t5], ours_mean_top1=m1, ours_mean_top5=m5,
                 published_top1=P.get("top1"), published_top5=P.get("top5"), published_mean_top1=P["avg_top1"], published_mean_top5=P["avg_top5"],
                 published_source=P["source"], diff_mean_top1_pp=d1, diff_mean_top5_pp=d5, pass_top1_2pp=bool(abs(d1) <= 2.0))
        if P.get("top1"): r["per_participant_diff_top1_pp"] = [round(a - b, 2) for a, b in zip(t1, P["top1"])]; r["per_participant_diff_top5_pp"] = [round(a - b, 2) for a, b in zip(t5, P["top5"])]
        if extra: r.update(extra)
        table[name] = r
    g = lambda key, q: [100 * p["rows"][key][q] for p in per]
    row("nice_ga_centres", "nice_ga", g("nice_ga_centres", "top1"), g("nice_ga_centres", "top5"), "k80 vs CLIP ViT-L/14 concept centres (center_clip.npy) = the NICE paper's protocol (Sec. 3.1, repo README) and S1's", dict(matches_paper_protocol=True))
    row("nice_ga_testimg", "nice_ga", g("nice_ga_testimg", "top1"), g("nice_ga_testimg", "top5"), "k80 vs CLIP ViT-L/14 of the 200 shown test images (clip_feature_maps_test.npy); NOT the NICE protocol", dict(matches_paper_protocol=False))
    row("atm_s", "atm_s", g("atm_s_native", "top1"), g("atm_s_native", "top5"), "released k80 test embeddings vs CLIP ViT-H/14 test-image features", dict(matches_paper_protocol=True, published_note=PUB["atm_s"]["note"]))
    row("brainflora", "brainflora", g("brainflora_native", "top1"), g("brainflora_native", "top5"), "released joint-subject EEG checkpoint, k80, vs its released ViT-H-14 test features",
        dict(matches_paper_protocol=True, published_setting=PUB["brainflora"]["setting"], intra_subject_published=dict(top1=PUB["brainflora"]["intra_subject_avg_top1"], top5=PUB["brainflora"]["intra_subject_avg_top5"], source=PUB["brainflora"]["intra_subject_source"]),
             diff_vs_intra_subject_top1_pp=float(np.mean(g("brainflora_native", "top1"))) - PUB["brainflora"]["intra_subject_avg_top1"]))
    row("ubp", "ubp", ubp_t1, ubp_t5, "released checkpoints, k80, UBP's own RN50 fovea-blur features; top-1 = our forward pass (ubp_card.py, results/baselines/ubp); top-5 = the released test_results.json",
        dict(matches_paper_protocol=True, top5_source="released test_results.json (our top-1 equals its top-1 in all 10: max |d| = %.2f pp)" % max(abs(a - b) for a, b in zip(ubp_t1, ubp_t1_json))))
    gi = {k: [p["g2_identity"].get(k) for p in per] for k in per[0]["g2_identity"]}
    diag = dict(nice_soft12_k80=[p["rows"]["nice_ga_centres"]["soft12"] for p in per], nice_ls40_k80=[p["rows"]["nice_ga_centres"]["ls40"] for p in per],
                r1_registered_soft_bits=gi["r1_soft_bits"], soft12_minus_ls40=[p["rows"]["nice_ga_centres"]["soft12"] - p["rows"]["nice_ga_centres"]["ls40"] for p in per])
    diag.update(mean_soft12=float(np.mean(diag["nice_soft12_k80"])), mean_ls40=float(np.mean(diag["nice_ls40_k80"])), mean_r1=float(np.mean(diag["r1_registered_soft_bits"])))
    res = dict(code_sha256=CODE_SHA256, per_participant_code_sha256=per[0]["code_sha256"], n=len(per), participants=[p["participant"] for p in per],
               scoring="200-way zero-shot test identification, intra-subject split, all 80 repetitions averaged (k80); top-k by rank = 1 + #candidates with strictly higher cosine",
               pass_rule="|mean top-1 ours - published| <= 2 pp", table=table, g2_identity_checks=gi, nice_ga_k80_infonce_grid_diagnosis=diag)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for k, r in table.items():
        print(f"{k:16s} ours {r['ours_mean_top1']:.2f}/{r['ours_mean_top5']:.2f}  published {r['published_mean_top1']}/{r['published_mean_top5']}  d top-1 {r['diff_mean_top1_pp']:+.2f} pp  {'PASS' if r['pass_top1_2pp'] else 'FAIL'}")
    print("NICE-GA k80 InfoNCE: soft12", round(diag["mean_soft12"], 4), "ls40", round(diag["mean_ls40"], 4), "r1", round(diag["mean_r1"], 4))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--summarize", action="store_true"); a = ap.parse_args()
    if a.summarize: return summarize()
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
        if not (OUT / f"sub-{s:02d}.json").exists(): subject(s)


if __name__ == "__main__":
    main()
