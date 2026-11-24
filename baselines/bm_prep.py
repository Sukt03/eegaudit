"""brainmagick-style speech segment retrieval on Brennan & Hale 2019 (Alice, EEG) — OUR RE-IMPLEMENTATION of Défossez et al. 2023 (NMI;
arXiv 2208.12266; CC BY-NC 4.0 code not run). IMPLEMENTER, 21 Nov 2026. Step 1: preprocessing (paper §2.4).
EEG: BIDS nm000180 (500 Hz, 60 EEG ch; EOG dropped), participants with analyzed == yes; resampled to 120 Hz on the GPU (torchaudio);
robust scaler per recording and channel ((x - median) / (IQR / 2), so [q25, q75] -> [-1, 1]); clamp at +-20. Baseline correction is per
window at training time. Sensor 2D positions: azimuthal-equidistant projection of the CapTrak 3D positions, normalised to [0.1, 0.9]
(the paper uses MNE find_layout; margin 0.1 as in the paper). Word table: EEG word onset (events.tsv), wav segment, onset inside the wav
(sourcedata/AliceChapterOne-EEG.csv by Order = word_index), sentence id.
Speech: wav2vec2-large-xlsr-53 (external/brainmagick_w2v, safetensors converted with weights_only), average of the LAST FOUR transformer
layers, standard-normalised per dimension over the whole story, resampled 50 -> 120 Hz (linear). Output: results/baselines/brainmagick/prep.pt"""
import csv, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import mne, soundfile as sf, torchaudio
from transformers import Wav2Vec2Model

dev = torch.device('cuda'); t0 = time.time()
D = '/path/to/eegaudit/data_raw/third_party/alice_nm000180'
W2V = '/path/to/eegaudit/external/brainmagick_w2v/wav2vec2_large_xlsr53'
OUT = '/path/to/eegaudit/results/baselines/brainmagick'; os.makedirs(OUT, exist_ok=True)
SR = 120
subs = [r['participant_id'] for r in csv.DictReader(open(f'{D}/participants.tsv'), delimiter='\t') if r['analyzed'] == 'yes']
print(len(subs), 'participants (analyzed == yes)', flush=True)
audio_on = {}
for r in csv.DictReader(open(f'{D}/sourcedata/AliceChapterOne-EEG.csv')):
    audio_on[int(r['Order'])] = (int(r['Segment']), float(r['onset']))

out = {'note': __doc__, 'subjects': {}, 'sr': SR}
pos2d = None
for s in subs:
    e = f'{D}/{s}/eeg/{s}_task-alicelistening'
    raw = mne.io.read_raw_brainvision(e + '_eeg.vhdr', preload=True, verbose='ERROR')
    chs = [r for r in csv.DictReader(open(e + '_channels.tsv', encoding='utf-8-sig'), delimiter='\t')]
    eeg_names = [r['name'] for r in chs if r['type'].upper() == 'EEG']
    X = torch.from_numpy(raw.get_data(picks=eeg_names)).float().to(dev)                  # (60, N) volts at 500 Hz
    X = torchaudio.functional.resample(X, int(raw.info['sfreq']), SR)
    med = X.median(1, keepdim=True).values; q = torch.quantile(X, torch.tensor([0.25, 0.75], device=dev), dim=1).T
    X = ((X - med) / ((q[:, 1:] - q[:, :1]) / 2 + 1e-12)).clamp(-20, 20)
    if pos2d is None:
        el = {r['name']: (float(r['x']), float(r['y']), float(r['z'])) for r in csv.DictReader(open(f'{D}/{s}/eeg/{s}_space-CapTrak_electrodes.tsv', encoding='utf-8-sig'), delimiter='\t') if r['x'] != 'n/a'}
        P = np.array([el[n] for n in eeg_names]); P = P - P.mean(0) * np.array([1, 1, 0]); P /= np.linalg.norm(P, axis=1, keepdims=True)
        theta = np.arccos(np.clip(P[:, 2], -1, 1)); phi = np.arctan2(P[:, 1], P[:, 0])   # azimuthal equidistant projection
        xy = np.stack([theta * np.cos(phi), theta * np.sin(phi)], 1); xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        pos2d = 0.1 + 0.8 * xy
    ev = [r for r in csv.DictReader(open(e + '_events.tsv'), delimiter='\t') if r['trial_type'] == 'word']
    words = [{'eeg_onset': float(r['onset']), 'word_index': int(r['word_index']), 'sentence_id': int(r['sentence_id']), 'word': r['word'],
              'wav': audio_on[int(r['word_index'])][0], 'audio_onset': audio_on[int(r['word_index'])][1]} for r in ev]
    out['subjects'][s] = {'eeg': X.half().cpu(), 'words': words, 'channels': eeg_names}
    print(s, tuple(X.shape), len(words), 'words', f'{time.time()-t0:.0f}s', flush=True)
out['pos2d'] = torch.tensor(pos2d, dtype=torch.float32)

# ---------------- wav2vec 2.0 features per wav
m = Wav2Vec2Model.from_pretrained(W2V).to(dev).eval()
feats = {}
for k in range(1, 13):
    a, sr = sf.read(f'{D}/stimuli/DownTheRabbitHoleFinal_SoundFile{k}.wav', dtype='float32')
    a = torch.from_numpy(a if a.ndim == 1 else a.mean(1)).to(dev)
    a = torchaudio.functional.resample(a, sr, 16000); a = (a - a.mean()) / (a.std() + 1e-7)
    with torch.no_grad():
        hs = m(a[None], output_hidden_states=True).hidden_states
    h = torch.stack(hs[-4:]).mean(0)[0]                                                    # (T50, 1024)
    feats[k] = h; print('wav', k, tuple(h.shape), f'{a.numel()/16000:.1f}s', flush=True)
allh = torch.cat(list(feats.values())); mu, sd = allh.mean(0), allh.std(0) + 1e-6
for k, h in feats.items():
    h = (h - mu) / sd; n120 = int(round(h.shape[0] * SR / 50))
    feats[k] = F.interpolate(h.T[None], size=n120, mode='linear', align_corners=False)[0].half().cpu()   # (1024, T120)
out['w2v'] = feats
torch.save(out, f'{OUT}/prep.pt'); print('wrote', f'{OUT}/prep.pt', f'{time.time()-t0:.0f}s', flush=True)
