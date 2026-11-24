"""ZuCo POSITIVE CONTROL cache builder (, 23 Nov 2026;).
Plants a KNOWN lexical signal into the word-level EEG features of the existing BrainTranslator cache (results/track_text/brain_translator/cache_full.pkl, 840-d GD word features):
for record r (sentence g) the i-th EEG word row gets  amp * SD_f * v(token_i)  added, where token_i is the i-th whitespace token of the sentence (lower-cased, punctuation stripped),
v(token) = fixed Gaussian 840-vector from a hash of the token (same vector for the same word in every sentence and for every participant), SD_f = per-feature SD of the TRAINING rows.
(EEG word rows are fixation-aligned and are fewer than tokens, so the i-th row is only approximately the i-th token: the planted signal is the known function 'token i of this sentence', not a claim about reading.)
The pipeline is then the unchanged train_bt.py (pretrained BART-base, seed 0) and its identification yield is compared with the unplanted EEG arm and the N(0,1) noise-input arm. amp in {0 (control: must equal the unplanted cache), 0.3, 1.0}.
Gate G1: amp=0 cache identical to the source cache (max abs diff 0). Usage: --amp 1.0 --out results/gpu_fill_20261123/zuco_plant/cache_amp1.0.pkl"""
import argparse, hashlib, pickle, re, sys, zlib
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--amp', type=float, required=True); ap.add_argument('--out', required=True)
ap.add_argument('--cache', default='/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl'); a = ap.parse_args()
D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']
Xtr = np.concatenate([r['words'] for r in recs if r['split'] == 'train']); sd = Xtr.std(0) + 1e-6; del Xtr
def vec(tok): return np.random.default_rng(zlib.crc32(tok.encode())).standard_normal(840).astype(np.float32)
norm = lambda w: re.sub(r"[^a-z0-9']", '', w.lower()) or '_'
cache = {}; out = []; maxd = 0.0; npl = 0
for r in recs:
    toks = [norm(w) for w in texts[r['g']].split()]; W = r['words'].copy(); n = min(len(W), len(toks))
    for i in range(n):
        v = cache.setdefault(toks[i], vec(toks[i])); W[i] += a.amp * sd * v
    maxd = max(maxd, float(np.abs(W - r['words']).max())); npl += n; out.append({**r, 'words': W})
if a.amp == 0: assert maxd == 0.0
pickle.dump({**D, 'recs': out, 'planted': dict(amp=a.amp, source=a.cache, n_rows=npl, script_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest())}, open(a.out, 'wb'))
print('planted amp', a.amp, 'rows', npl, 'max |delta|', maxd, 'wrote', a.out)
