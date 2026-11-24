"""X5: yield of the RELEASED text generations (EXPERIMENTS agent, 2026-11-20). Files fetched unmodified from the decoders' repos into
data_raw/released_generations/{neuspeech,wangji}/ (NeuSpeech/EEG-To-Text results/: T5Translator; MikeWangWZHL/EEG-To-Text results/: BrainTranslator).
Each file = blocks of 'target string:', 'predicted string with tf:' (teacher forcing, NOT a decoder output usable at test time), 'predicted string:'
(free-running generation). Estimator (same as the within-task identification of the paper): MiniLM (all-MiniLM-L6-v2, mean pooling) embedding of the
generation vs the embeddings of all UNIQUE target sentences of the same file; top-1, 2-way (ties 1/2), Wolpaw at N = #unique targets.
Controls: shuffled pairs (generations permuted against targets); length-only read-out (target chosen by |#words(gen) - #words(target)|, ties 1/2);
#unique generations. Paired arm differences (EEG minus noise arm) per entry when the two files list the same targets in the same order; CI by
bootstrap over unique target sentences (cluster bootstrap, 5000). Output: results/loop/x5_released_generations/x5_released_generations.json"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import embed_sentences, wolpaw_bits
D = Path("/path/to/eegaudit/data_raw/released_generations"); OUT = Path("/path/to/eegaudit/results/loop/x5_released_generations"); OUT.mkdir(parents=True, exist_ok=True)
MINILM = "/path/to/eegaudit/data_raw/zuco/minilm"; _cache = {}


def parse(f):
    T, TF, G = [], [], []
    txt = open(f, encoding="utf-8", errors="replace").read()
    if "Predicted:" in txt:   # Wang & Ji format: start### / Predicted: / True: / end###; teacher-forcing files carry the tf output as 'Predicted'
        for blk in txt.split("end################################################"):
            g = re.search(r"Predicted:(.*)", blk); t = re.search(r"True:(.*)", blk)
            if t and g: T.append(t.group(1).strip()); TF.append(None); G.append(g.group(1).strip())
        return T, TF, G
    for blk in open(f, encoding="utf-8", errors="replace").read().split("################################################"):
        t = re.search(r"target string:(.*)", blk); tf = re.search(r"predicted string with tf:(.*)", blk); g = re.search(r"predicted string:(.*)", blk)
        if t and g: T.append(t.group(1).strip()); TF.append(tf.group(1).strip() if tf else None); G.append(g.group(1).strip())
    return T, TF, G


def emb(texts):
    new = [t for t in dict.fromkeys(texts) if t not in _cache]
    if new:
        E, _ = embed_sentences(new, MINILM); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12); _cache.update(zip(new, E))
    return np.stack([_cache[t] for t in texts])


def ident(S, true):
    t = S[np.arange(len(true)), true][:, None]; N = S.shape[1]; mask = np.ones_like(S, bool); mask[np.arange(len(true)), true] = False
    beat = ((S > t) & mask).sum(1); tie = ((S == t) & mask).sum(1)
    return np.where(beat == 0, 1.0 / (1 + tie), 0.0), 1 - (beat + 0.5 * tie) / (N - 1)


def score(T, G, rng):
    U = list(dict.fromkeys(T)); pos = {u: i for i, u in enumerate(U)}; true = np.array([pos[t] for t in T]); N = len(U)
    EU, EG = emb(U), emb(G); S = EG @ EU.T
    t1, tw = ident(S, true); st1, stw = ident(S, true[rng.permutation(len(true))])
    lu = np.array([len(u.split()) for u in U], float); lg = np.array([len(g.split()) for g in G], float); lt1, ltw = ident(-np.abs(lg[:, None] - lu[None, :]), true)
    top1 = float(t1.mean())
    return dict(n_entries=len(T), n_unique_targets=N, n_unique_generations=len(set(G)), top1=top1, two_way=float(tw.mean()), wolpaw=float(wolpaw_bits(top1, N)),
                shuffled_top1=float(st1.mean()), shuffled_two_way=float(stw.mean()), length_only_top1=float(lt1.mean()), length_only_two_way=float(ltw.mean()),
                chance_top1=1.0 / N), dict(t1=t1, tw=tw, ltw=ltw, true=true)


def cboot(diff, clusters, seed=0):
    rng = np.random.default_rng(seed); cl = np.unique(clusters); idx = {c: np.where(clusters == c)[0] for c in cl}; b = []
    for _ in range(5000):
        pick = rng.choice(cl, len(cl)); b.append(np.mean(np.concatenate([diff[idx[c]] for c in pick])))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


rng = np.random.default_rng(0); res = dict(note=__doc__, files={}, contrasts={})
raw = {}
for f in sorted(D.glob("*/*.txt")):
    T, TF, G = parse(f); key = f"{f.parent.name}/{f.stem}"
    s, per = score(T, G, rng); res["files"][key] = dict(free_running=s); raw[key] = (T, per)
    if all(TF): s2, _ = score(T, TF, rng); res["files"][key]["teacher_forcing_not_a_decoder_output"] = s2
    print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
# paired contrasts: EEG-test arm vs noise-test arm of the same release and task set
pairs = [("neuspeech/task1_task2_taskNRv2-T5Translator-all_decoding_results", "neuspeech/task1_task2_taskNRv2-T5Translator-EEG_noise-all_decoding_results", "T5 NRv2: EEG test vs noise test (EEG-trained)"),
         ("neuspeech/task1_task2_taskNRv2-T5Translator-noise_EEG-all_decoding_results", "neuspeech/task1_task2_taskNRv2-T5Translator-noise_noise-all_decoding_results", "T5 NRv2: noise-trained, EEG test vs noise test"),
         ("neuspeech/task1_task2_taskNRv2-T5Translator-all_decoding_results", "neuspeech/task1_task2_taskNRv2-T5Translator-noise_noise-all_decoding_results", "T5 NRv2: EEG/EEG vs noise/noise"),
         ("neuspeech/task1_task2_task3-T5Translator-all_decoding_results", "neuspeech/task1_task2_task3-T5Translator-EEG_noise-all_decoding_results", "T5 task3: EEG test vs noise test (EEG-trained)"),
         ("neuspeech/task1_task2_task3-T5Translator-all_decoding_results", "neuspeech/task1_task2_task3-T5Translator-noise_noise-all_decoding_results", "T5 task3: EEG/EEG vs noise/noise"),
         ("wangji/task1_task2_taskNRv2-BrainTranslator-all_decoding_results", "wangji/task1_task2_taskNRv2-BrainTranslator-intput_noise-all_decoding_results", "BrainTranslator NRv2: EEG test vs noise test")]
for a, b, name in pairs:
    if a not in raw or b not in raw: continue
    (Ta, pa), (Tb, pb) = raw[a], raw[b]
    if Ta != Tb: res["contrasts"][name] = dict(error="target order differs; unpaired"); continue
    d2 = pa["tw"] - pb["tw"]; d1 = pa["t1"] - pb["t1"]; dl = pa["tw"] - pa["ltw"]
    res["contrasts"][name] = dict(n=len(d2), n_clusters=int(len(np.unique(pa["true"]))), two_way_diff=float(d2.mean()), two_way_diff_ci95=cboot(d2, pa["true"]),
                                  top1_diff=float(d1.mean()), top1_diff_ci95=cboot(d1, pa["true"]), eeg_minus_length_only_two_way=float(dl.mean()), eeg_minus_length_only_ci95=cboot(dl, pa["true"]))
    c = res["contrasts"][name]; c["pass"] = bool(c["two_way_diff_ci95"][0] > 0 and c["eeg_minus_length_only_ci95"][0] > 0)
    print(name, {k: v for k, v in c.items()}, flush=True)
(OUT / "x5_released_generations.json").write_text(json.dumps(res, indent=1))
