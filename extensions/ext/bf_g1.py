"""J-3 BrainFLORA trained-path G1 (, 23 Nov 2026). Released BrainFLORA EEG encoder checkpoint scored with the soft12 scorer
(eegbits run.soft_bits), direct cosine to ViT-H-14 test features, compared with saved u1_native_v2 brainflora k=80 InfoNCE. Reports differences; no assert."""
import json, sys, numpy as np, torch
from pathlib import Path
ROOT = Path('/path/to/eegaudit'); O = ROOT / 'outputs/AGENT_LOOP_20261120'
sys.path[:0] = [str(O / 'models/brainflora/source'), str(ROOT / 'experiments/eegbits')]
from run import soft_bits
from model.medformer_encoders import eeg_encoder
dev = 'cuda'; enc = eeg_encoder().to(dev).eval()
ck = torch.load(O / 'models/brainflora/eeg_01-06_01-46_150.pth', map_location=dev, weights_only=False)
sd = ck.get('state_dict', ck.get('model', ck)) if isinstance(ck, dict) else ck
print('load', enc.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False))
Y = torch.load('/path/to/eegaudit/outputs/AGENT_LOOP_20261120/models/brainflora/ViT-H-14_features_test.pt', map_location='cpu', weights_only=True); Y = (Y['img_features'] if isinstance(Y, dict) else Y).float(); Y = Y.reshape(len(Y), -1)
nrm = lambda v: v / (v.norm(dim=1, keepdim=True) + 1e-12); out = {}
for i in range(1, 11):
    sub = f'sub-{i:02d}'; x = torch.from_numpy(np.load(ROOT / f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy')).float()
    with torch.inference_mode(): z = enc(x.mean(1).to(dev)).float().cpu()
    b = float(soft_bits((nrm(z) @ nrm(Y).T).numpy().astype(np.float64))); ref = json.loads((O / f'results/u1_native_v2/{sub}.json').read_text())
    sv = [r['eeg']['InfoNCE'] for r in ref['rows'] if r['model'] == 'brainflora' and r['k'] == 80][0]
    out[sub] = dict(this=b, saved=sv, abs_diff=abs(b - sv)); print(sub, out[sub], flush=True)
(ROOT / 'results/ext/bf_g1.json').write_text(json.dumps(dict(NOT_CITABLE='until review', subs=out, max_abs_diff=max(v['abs_diff'] for v in out.values())), indent=1))
