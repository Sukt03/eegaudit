"""Probe the structure of a Wang & Ji (AAAI 2022) EEG-To-Text ZuCo pickle. Run with python -I."""
import pickle, sys, numpy as np
p = sys.argv[1]
d = pickle.load(open(p, 'rb'))
print(type(d), len(d), list(d.keys())[:20])
subj = list(d.keys())[0]
sents = d[subj]
print('subject', subj, 'n sentences', len(sents), 'None count', sum(s is None for s in sents))
s = next(x for x in sents if x is not None)
print('sentence keys', list(s.keys()))
print('content:', s['content'])
print('sentence_level_EEG keys', list(s['sentence_level_EEG'].keys())[:10])
w = s['word'][0]
print('word keys', list(w.keys()))
print('word content', w['content'], 'nFix', w.get('nFixations'))
wl = w['word_level_EEG']
print('word_level_EEG keys', list(wl.keys()))
for k in wl:
    print(' ', k, {kk: np.asarray(v).shape for kk, v in wl[k].items()})
print('n words', len(s['word']), 'tokens_has_fixation', s.get('word_tokens_has_fixation'))
# per subject counts
for sj in d:
    ss = d[sj]; nn = sum(x is None for x in ss)
    nw = [len(x['word']) for x in ss if x is not None]
    print(sj, 'sent', len(ss), 'None', nn, 'mean words with EEG', np.mean(nw) if nw else None)
