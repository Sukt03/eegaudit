"""D4.4 / c1 WORD SURPRISAL WHILE LISTENING (Alice, Brennan & Hale 2019, NEMAR nm000180, CC-BY; rungs R1/R3). EXECUTOR-3, 21 Nov 2026.
Design = EXPERIMENT_TABLES_FINAL.md D4.4 + METHOD NOVEL SOLUTIONS ROUND 1 c1 (word-onset gate FIRST).
Data: 45 BIDS participants (BrainVision, 60 EEG ch, 500 Hz); word onsets from events.tsv (reconstructed per participant by the curators).
  Participants: those marked analyzed == yes in participants.tsv (the original QC; exclusions printed).
Preprocessing (named CPU exception: file loading + MNE filtering, no model fitting on the CPU): average reference over the 60 EEG channels,
  0.1-30 Hz, decimate to 50 Hz; epochs 0-600 ms after each word onset (baseline: none, the stream is continuous; auditory epochs of
  neighbouring words overlap: stated); epochs with any channel peak-to-peak > 200 uV dropped.
WORD-ONSET GATE (--gate, run FIRST): per participant, GFP of the word-locked average at 50-250 ms vs 20 sets of random onsets (uniform over
  the recording, same count); pass = GFP above the 95th percentile of the random sets. Group gate = pass in >= 80 % of participants.
  The yield is run only if the group gate passes.
Target: GPT-2 small surprisal per word (-log2 p(word | all previous words of the chapter), summed over the word's BPE tokens, context window
  1024 tokens), CONTENT words only (is_lexical == 1), normal-scored, residualised on log frequency, word length (characters), position in
  sentence, local onset rate (log inter-onset interval) and the surprisal of the previous and next word (METHOD G1 FIX), z-scored
  (stimulus covariates only, as X7). Folds: 2-word embargo around every held-out block (outer and inner; METHOD G1 FIX).
Read-out and score: X7-style ridge on the 60 x 31 epoch (in-fold z-score, PCA-64 primary + full, LAM * n, 10 contiguous folds in time order),
  locked Gaussian log-density gain with nested calibration (d2_pe_tasks.pe_gain, batched GPU); floors: circular shifts n/4, n/2, 3n/4 and noise
  input; ceiling: ridge split-half (d2_pe_tasks.split_half_ceiling; r12 <= 0 -> "no usable ceiling"). Group: participant bootstrap (GPU),
  CHANGE 12 floor gate (d2_pe_tasks.g3_test). Listening is never pooled with reading identification (METHOD T3).
Usage: python d4_alice_surprisal.py --surprisal | --gate [--subjects ..] | [--subjects ..] --out DIR | --summarize --out DIR
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2

DEV = "cuda"; A = ROOT / "data_raw/third_party/alice_nm000180"; OUTD = ROOT / "results/ext/alice"; OUTD.mkdir(parents=True, exist_ok=True)
GPT2 = ROOT / "data_raw/third_party/gpt2_small"; LS = (64, "full"); EMB = 2   # METHOD G1 FIX c1: 2-word fold embargo


def participants():
    rows = list(csv.DictReader(open(A / "participants.tsv"), delimiter="\t"))
    return [int(r["participant_id"][4:]) for r in rows if r["analyzed"].strip() == "yes"], {r["participant_id"]: r["exclusion_reason"] for r in rows if r["analyzed"].strip() != "yes"}


def events(s):
    return list(csv.DictReader(open(A / f"sub-{s:03d}/eeg/sub-{s:03d}_task-alicelistening_events.tsv"), delimiter="\t"))


def surprisal():
    """GPT-2 small word surprisal (bits) for the chapter's word sequence (identical across participants; checked)."""
    f = OUTD / "gpt2_surprisal.npz"
    if f.exists(): return np.load(f)["s"]
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    words = [e["word"] for e in events(1)]
    tok = GPT2TokenizerFast.from_pretrained(GPT2); mdl = GPT2LMHeadModel.from_pretrained(GPT2).to(DEV).eval()
    ids, owner = [tok.bos_token_id], [-1]
    for i, w in enumerate(words):
        t = tok.encode((" " if i else "") + w); ids += t; owner += [i] * len(t)
    ids = torch.tensor(ids, device=DEV); owner = np.array(owner); s = np.zeros(len(words)); W = 1024; step = 512
    with torch.no_grad():
        done = 1
        while done < len(ids):
            lo = max(0, done - (W - step)); hi = min(len(ids), lo + W); lp = torch.log_softmax(mdl(ids[lo:hi][None]).logits[0].double(), -1)
            for p in range(max(done, lo + 1), hi):
                s[owner[p]] += float(-lp[p - 1 - lo, ids[p]] / np.log(2))
            done = hi
    np.savez(f, s=s, words=np.array(words)); return s


def load_eeg(s):
    import mne
    raw = mne.io.read_raw_brainvision(A / f"sub-{s:03d}/eeg/sub-{s:03d}_task-alicelistening_eeg.vhdr", preload=True, verbose="error")
    eeg = [c for c in raw.ch_names if c not in ("VEOG", "Aux5")]; raw.pick(eeg); raw.set_eeg_reference("average", verbose="error")
    raw.filter(0.1, 30.0, verbose="error"); raw.resample(50.0, verbose="error"); return raw.get_data() * 1e6, raw.info["sfreq"], eeg


def epochs(X, sf, onsets):
    idx = np.round(np.asarray(onsets) * sf).astype(int); ok = (idx >= 0) & (idx + 31 <= X.shape[1])
    E = np.stack([X[:, i:i + 31] for i in idx[ok]]) if ok.any() else np.zeros((0, X.shape[0], 31)); return E, ok


def gate(subjects):
    res = {}
    for s in subjects:
        X, sf, ch = load_eeg(s); ev = events(s); on = np.array([float(e["onset"]) for e in ev]); E, ok = epochs(X, sf, on)
        Et = torch.as_tensor(E, device=DEV); gfp = lambda M: float(M.mean(0)[:, 2:13].std(0).mean())     # 40-240 ms at 50 Hz (samples 2..12)
        g0 = gfp(Et); rng = np.random.default_rng(s); rnd = []
        for k in range(20):
            Er, _ = epochs(X, sf, np.sort(rng.uniform(0, X.shape[1] / sf - 0.7, len(on)))); rnd.append(gfp(torch.as_tensor(Er, device=DEV)))
        res[f"sub-{s:03d}"] = dict(gfp_word=g0, gfp_random_p95=float(np.quantile(rnd, 0.95)), passed=bool(g0 > np.quantile(rnd, 0.95)), n_words=int(ok.sum()))
        print(f"sub-{s:03d}", res[f"sub-{s:03d}"], flush=True)
    frac = float(np.mean([v["passed"] for v in res.values()])); out = dict(per=res, pass_fraction=frac, group_gate_passed=bool(frac >= 0.8))
    (OUTD / "word_onset_gate.json").write_text(json.dumps(out, indent=1)); print("GATE", frac, out["group_gate_passed"], flush=True); return out


def subject(s, out, S):
    t0 = time.time(); X, sf, ch = load_eeg(s); ev = events(s)
    assert [e["word"] for e in ev] == [str(w) for w in np.load(OUTD / "gpt2_surprisal.npz")["words"]], "word sequence differs from the surprisal file"
    on = np.array([float(e["onset"]) for e in ev]); E, ok = epochs(X, sf, on)
    lex = np.array([e["is_lexical"] == "1" for e in ev])[ok]; ptp = (E.max(2) - E.min(2)).max(1); keep = lex & (ptp < 200)
    from scipy import stats
    ns = lambda v: stats.norm.ppf((stats.rankdata(v) - 0.5) / len(v))                   # target transform (rank-based; CPU, trivial)
    ev_ok = [e for e, k in zip(ev, ok) if k]; ii = np.where(keep)[0]
    y = ns(S[ok][ii]); onk = on[ok][ii]
    widx = np.where(ok)[0][ii]                                                      # original word positions of the kept content words
    Sprev = np.r_[0.0, S][widx]; Snext = np.r_[S, 0.0][widx + 1]                   # METHOD G1 FIX c1: neighbour-word surprisal covariates (all words)
    cov = np.column_stack([[float(ev_ok[i]["log_freq"]) for i in ii], [len(ev_ok[i]["word"]) for i in ii], [float(ev_ok[i]["position_in_sentence"]) for i in ii],
                           np.log(np.r_[1.0, np.clip(np.diff(onk), 0.05, None)]), Sprev, Snext])
    y = D2.resid(y, [cov[:, j] for j in range(cov.shape[1])]); y = (y - y.mean()) / (y.std() + 1e-9)
    F = E[ii].reshape(len(ii), -1).astype(np.float64); n = len(y)
    Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)]); sc = D2.pe_gain(F, Ys, LS, embargo=EMB); nz = D2.pe_gain(D2.gnoise(F.shape, 1000 + s), Ys[:, :1], LS, embargo=EMB)
    ce = D2.split_half_ceiling(F, y)
    rows = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean()), noise=float(nz[L][:, 0].mean()),
                         floor=max(float(sc[L][:, 1:].mean()), float(nz[L][:, 0].mean()))) for L in LS}
    res = dict(participant=f"sub-{s:03d}", n_words=int(len(ev)), n_content_kept=int(n), n_dropped_ptp=int((lex & ~(ptp < 200)).sum()), channels=len(ch),
               unit="bits per content word", target="GPT-2 small surprisal (bits), normal-scored, residualised on log frequency, length, position, onset rate, neighbour surprisal", embargo_words=EMB,
               rows=rows, ceiling=ce, runtime_s=round(time.time() - t0, 1))
    (out / f"alice_sub-{s:03d}.json").write_text(json.dumps(res, indent=1))
    print(f"sub-{s:03d}", "n", n, "pca64", round(rows["64"]["gain"], 4), "floor", round(rows["64"]["floor"], 4), "full", round(rows["full"]["gain"], 4), "ceil", ce["bits"], f"{res['runtime_s']}s", flush=True)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("alice_sub-*.json"))]; res = dict(n=len(per))
    for L in ("64", "full"):
        g = np.array([p["rows"][L]["gain"] for p in per]); fl = np.array([p["rows"][L]["floor"] for p in per])
        res["L" + L] = dict(gain=float(g.mean()), gain_ci=D2.boot_ci(g), floor=float(fl.mean()), gain_minus_floor_ci=D2.boot_ci(g - fl), n_above_floor=int((g > fl).sum()),
                            g3_shift=D2.g3_test([p["rows"][L]["shift_mean"] for p in per]), g3_noise=D2.g3_test([p["rows"][L]["noise"] for p in per]))
    cu = [p["ceiling"]["bits"] for p in per if p["ceiling"]["usable"]]; res["ceiling_usable_n"] = len(cu); res["ceiling_mean_usable"] = float(np.mean(cu)) if cu else None
    (out / "summary_alice.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", default=str(OUTD)); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--surprisal", action="store_true"); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); subs, excl = participants()
    if a.subjects: subs = [int(x) for x in a.subjects.split(",")]
    if a.surprisal: s = surprisal(); print("surprisal", len(s), "mean bits", float(s.mean()), flush=True); return
    if a.gate: print("excluded by the original QC:", excl, flush=True); return gate(subs)
    if a.summarize: return summarize(out)
    g = json.loads((OUTD / "word_onset_gate.json").read_text()); assert g["group_gate_passed"], "word-onset gate not passed"
    S = surprisal()
    for s in subs:
        if not (out / f"alice_sub-{s:03d}.json").exists(): subject(s, out, S)


if __name__ == "__main__":
    main()
