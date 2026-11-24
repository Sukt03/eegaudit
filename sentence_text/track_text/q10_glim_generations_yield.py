"""Q10: yield of GLIM's RELEASED generations (Liu, Shen & Liu, arXiv 2505.17099) with their own noise-input and prompt-free arms
(EXPERIMENTS agent, 2026-11-20). Files (unmodified) in data_raw/third_party/glim_results/: wandb_export_gen_samples_{glim,noise_input_test,prompt_free_test}.csv.
Columns used: Dataset, Subject, Raw Task Key, Input Text (the read sentence = target), Generated Text (free-running), Generated Text (w/tf).
Estimator = X5 / paper: MiniLM embedding of the generation vs embeddings of the unique target sentences of the SAME task (Dataset + Raw Task Key);
top-1, 2-way (ties 1/2), Wolpaw at N = #unique sentences of the task. Controls: shuffled pairs, length-only read-out, #unique generations.
Arms paired on (Dataset, Subject, task, Input Text, occurrence index). Per participant means; EEG minus noise per participant (bootstrap over
participants) and per entry (cluster bootstrap over sentences). Output results/loop/q10_glim/q10_glim.json"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import embed_sentences, wolpaw_bits
D = Path("/path/to/eegaudit/data_raw/third_party/glim_results"); OUT = Path("/path/to/eegaudit/results/loop/q10_glim"); OUT.mkdir(parents=True, exist_ok=True)
MINILM = "/path/to/eegaudit/data_raw/zuco/minilm"; _c = {}
ARMS = {"glim": "wandb_export_gen_samples_glim.csv", "noise_input_test": "wandb_export_gen_samples_noise_input_test.csv", "prompt_free_test": "wandb_export_gen_samples_prompt_free_test.csv"}


def emb(texts):
    new = [t for t in dict.fromkeys(texts) if t not in _c]
    if new:
        E, _ = embed_sentences(new, MINILM); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12); _c.update(zip(new, E))
    return np.stack([_c[t] for t in texts])


def ident(S, true):
    t = S[np.arange(len(true)), true][:, None]; N = S.shape[1]; m = np.ones_like(S, bool); m[np.arange(len(true)), true] = False
    beat = ((S > t) & m).sum(1); tie = ((S == t) & m).sum(1); return np.where(beat == 0, 1.0 / (1 + tie), 0.0), 1 - (beat + 0.5 * tie) / (N - 1)


def boot(v, seed=0):
    r = np.random.default_rng(seed); v = np.asarray(v, float); b = [r.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cboot(diff, cl, seed=0):
    r = np.random.default_rng(seed); u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
    b = [np.mean(np.concatenate([diff[idx[c]] for c in r.choice(u, len(u))])) for _ in range(3000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


rows = {}
for arm, fn in ARMS.items():
    R = list(csv.DictReader(open(D / fn, encoding="utf-8"))); occ = defaultdict(int); keyed = {}
    for r in R:
        k0 = (r["Dataset"], r["Subject"], r["Raw Task Key"], r["Input Text"].strip()); occ[k0] += 1; keyed[k0 + (occ[k0],)] = r
    rows[arm] = keyed
headers = list(csv.DictReader(open(D / ARMS["glim"], encoding="utf-8")).fieldnames)
common = sorted(set.intersection(*[set(v) for v in rows.values()]))
tasks = sorted({(k[0], k[2]) for k in common})
rng = np.random.default_rng(0); res = dict(note=__doc__, headers_verbatim=headers, n_rows={a: len(v) for a, v in rows.items()}, n_common=len(common), tasks={}, overall={})
per_entry = {a: {"tw": [], "t1": [], "ltw": [], "cl": [], "subj": [], "task": []} for a in ARMS}
for ds, tk in tasks:
    ks = [k for k in common if k[0] == ds and k[2] == tk]; U = sorted({k[3] for k in ks}); pos = {u: i for i, u in enumerate(U)}; N = len(U)
    if N < 5: continue
    EU = emb(U); true = np.array([pos[k[3]] for k in ks]); lu = np.array([len(u.split()) for u in U], float); tres = dict(n_entries=len(ks), N=N, n_subjects=len({k[1] for k in ks}))
    for arm in ARMS:
        G = [rows[arm][k]["Generated Text"].strip() for k in ks]; S = emb(G) @ EU.T; t1, tw = ident(S, true); st1, stw = ident(S, true[rng.permutation(len(true))])
        lg = np.array([len(g.split()) for g in G], float); lt1, ltw = ident(-np.abs(lg[:, None] - lu[None, :]), true); top1 = float(t1.mean())
        tres[arm] = dict(top1=top1, two_way=float(tw.mean()), wolpaw=float(wolpaw_bits(top1, N)), shuffled_two_way=float(stw.mean()), length_only_two_way=float(ltw.mean()), n_unique_generations=len(set(G)))
        GT = [rows[arm][k].get("Generated Text (w/tf)", "").strip() for k in ks]
        if all(GT): tt1, ttw = ident(emb(GT) @ EU.T, true); tres[arm]["teacher_forced_top1"] = float(tt1.mean()); tres[arm]["teacher_forced_two_way"] = float(ttw.mean())
        pe = per_entry[arm]; pe["tw"] += list(tw); pe["t1"] += list(t1); pe["ltw"] += list(ltw); pe["cl"] += [f"{ds}|{tk}|{k[3]}" for k in ks]; pe["subj"] += [k[1] for k in ks]; pe["task"] += [f"{ds}|{tk}"] * len(ks)
    d = np.array([x for x in per_entry["glim"]["tw"][-len(ks):]]) - np.array(per_entry["noise_input_test"]["tw"][-len(ks):]); cl = np.array(per_entry["glim"]["cl"][-len(ks):])
    dl = np.array(per_entry["glim"]["tw"][-len(ks):]) - np.array(per_entry["glim"]["ltw"][-len(ks):])
    tres["eeg_minus_noise_two_way"] = dict(mean=float(d.mean()), ci95=cboot(d, cl)); tres["eeg_minus_length_only_two_way"] = dict(mean=float(dl.mean()), ci95=cboot(dl, cl))
    tres["pass"] = bool(tres["eeg_minus_noise_two_way"]["ci95"][0] > 0 and tres["eeg_minus_length_only_two_way"]["ci95"][0] > 0)
    res["tasks"][f"{ds}|{tk}"] = tres
    print(ds, tk, {a: {k: round(v, 4) if isinstance(v, float) else v for k, v in tres[a].items()} for a in ARMS}, "EEG-noise", tres["eeg_minus_noise_two_way"], "EEG-length", tres["eeg_minus_length_only_two_way"], flush=True)
# per participant (pooled over tasks), EEG minus each other arm
for other in ("noise_input_test", "prompt_free_test"):
    subj = np.array(per_entry["glim"]["subj"]); d = np.array(per_entry["glim"]["tw"]) - np.array(per_entry[other]["tw"])
    pp = [d[subj == s].mean() for s in np.unique(subj)]
    res["overall"][f"glim_minus_{other}_two_way_per_participant"] = dict(mean=float(np.mean(pp)), ci95=boot(pp), n=len(pp), n_pos=int(np.sum(np.array(pp) > 0)))
res["n_tasks_pass"] = int(sum(t["pass"] for t in res["tasks"].values()))
res["verdict"] = "STRONG (EEG > noise and > length-only in >= 2 tasks)" if res["n_tasks_pass"] >= 2 else "null: GLIM generations do not carry sentence yield beyond noise/length under this estimator"
(OUT / "q10_glim.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["overall"], indent=1), res["n_tasks_pass"], res["verdict"])
