"""S1 POST-HOC (LOCK §4 noise-input floor for the exemplar / category rows; LOCK §5 k40-ceiling label). FIXER, 21 Nov 2026.
v2 (FIXER-3; REVIEWER overnight H): reads results/ext/s1_v6 (the locked soft12 grid for every construction; never edits it), not v5
(degenerate ceilings) nor v4 (biased ceilings). Noise floors unchanged in construction; the S1 module hash (s1_code_sha256_module) and the
reference file's hash (s1_code_sha256_ref) are both recorded and must agree. Reads the reference directory given by --ref (default s1_v6). Per participant x model x k: the SAME N(0,1) test input as S1's floor_noise (generator seed
1000 + k, shape 200 x 63 x 250) through the same frozen model; scored with the S1 v6 estimators (s1_averaging imported unchanged):
yield200 (G2: reproduces s1_v5 floor_noise within the S1 tolerance 1e-3; smoke sub-01 max 4e-4 from nondeterministic GPU convolutions), exemplar (LOO N-way tau, matched queries), category.
k40 label (LOCK §5): per participant and model, |new k40 ceiling200 - registered ceiling_k40| > 0.05 bits -> 'random-split ceiling'.
Output results/ext/s1_v6_posthoc/{sub-XX.json, summary.json} with code_sha256."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext"))
import s1_averaging as S1
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); REF = ROOT / "results/ext/s1_v6"


def subject(s, out):
    sub = f"sub-{s:02d}"; ref = json.loads((REF / f"{sub}.json").read_text())
    te = S1.categories(); N, pool_of, classes, cov = S1.pools_from(te); lab = np.full(200, -1)
    for c, v in enumerate(classes): lab[v] = c
    inc = np.where(lab >= 0)[0]; MD = S1.models(sub, S1.DEV); res = dict(participant=sub, code_sha256=CODE_SHA256, s1_code_sha256_ref=ref.get("code_sha256"), s1_code_sha256_module=S1.CODE_SHA256,
               s1_code_match=ref.get("code_sha256") == S1.CODE_SHA256, ref=str(REF), tau_grid=S1.taus().tolist(), rows={})
    for m, (ft, _, Yc) in MD.items():
        Yc = S1.nrm(torch.as_tensor(Yc, device=S1.DEV)); cent = S1.nrm(torch.stack([Yc[v].mean(0) for v in classes]))
        for k in S1.KS:
            gN = torch.Generator(device=S1.DEV); gN.manual_seed(1000 + k)
            En = S1.nrm(S1.embed(ft, torch.randn((200, 63, 250), generator=gN, device=S1.DEV).cpu().numpy())); S = (En @ Yc.T)[None]
            y = float(S1.info200(S)[0][0]); ex = S1.nway(S, None, pool_of, N, 50, 7 + k); ca = float(S1.catid(En[inc][None] @ cent.T, lab[inc])[0])
            r0 = ref["rows"][f"{m}_k{k}"]
            res["rows"][f"{m}_k{k}"] = dict(noise_yield200=y, g2_vs_s1_floor_noise=abs(y - r0["floor_noise"]), noise_exemplar=float(ex["matched"].mean()), noise_category=ca)
    (out / f"{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, "max G2", max(v["g2_vs_s1_floor_noise"] for v in res["rows"].values()), "max |noise ex|", round(max(abs(v["noise_exemplar"]) for v in res["rows"].values()), 4),
          "max |noise cat|", round(max(abs(v["noise_category"]) for v in res["rows"].values()), 4), flush=True)


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; S = json.loads((REF / "summary.json").read_text()) if (REF / "summary.json").exists() else None
    res = dict(code_sha256=CODE_SHA256, ref=str(REF), s1_code_sha256_ref=sorted({p["s1_code_sha256_ref"] for p in P}),
               s1_code_match_all=all(p["s1_code_match"] for p in P), ref_summary_present=S is not None, n=len(P), g2_max=max(v["g2_vs_s1_floor_noise"] for p in P for v in p["rows"].values()), noise_floors={})
    for key in P[0]["rows"]:
        res["noise_floors"][key] = {q: float(np.mean([p["rows"][key][q] for p in P])) for q in ("noise_yield200", "noise_exemplar", "noise_category")}
    res["floors_within_0.02"] = bool(all(abs(v) <= 0.02 for d in res["noise_floors"].values() for v in d.values()))
    if S is None or "ceiling_k40_vs_registered" not in S: res["k40_ceiling_label"] = f"NOT COMPUTED: {REF}/summary.json missing or has no ceiling_k40_vs_registered"
    if S and "ceiling_k40_vs_registered" in S:
        res["k40_ceiling_label"] = {m: {sub: dict(new=v["new"], registered=v["registered"], abs_diff=None if v["registered"] is None else abs(v["new"] - v["registered"]),
                                                  label=None if v["registered"] is None else ("random-split ceiling" if abs(v["new"] - v["registered"]) > 0.05 else "registered k40 ceiling"))
                                        for sub, v in d.items()} for m, d in S["ceiling_k40_vs_registered"].items()}
        res["k40_ceiling_label_counts"] = {m: sum(x["label"] == "random-split ceiling" for x in d.values()) for m, d in res["k40_ceiling_label"].items()}
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "noise_floors"}, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/s1_v6_posthoc")); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--ref", default=str(REF))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); REF = Path(a.ref)
    if a.summarize: summarize(out)
    else:
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists() and (REF / f"sub-{s:02d}.json").exists(): subject(s, out)
