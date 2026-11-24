"""DTHA-style teacher features (our re-implementation of arXiv 2611.12798, IMPLEMENTER 21 Nov 2026).
DINO teacher: DINOv2 ViT-L/14 (facebook/dinov2-large, transformers), CLS token after blocks 0..14 (hidden_states[1..15]), l2-normalised,
standard DINOv2 preprocessing (resize 256, centre crop 224, ImageNet normalisation). CLIP teacher: UBP's released RN50 features
(DirectT = sharp, FoveaBlur low / medium / high), re-keyed to the same image order. Output (fp16):
results/baselines/dtha/teacher_features.pt = {'train_imgs', 'test_imgs', 'dino_train' (N,15,1024), 'dino_test', 'clip_train' (N,4,1024), 'clip_test'}."""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load
from transformers import Dinov2Model, AutoImageProcessor

dev = torch.device('cuda'); t0 = time.time()
U = '/path/to/eegaudit/data_raw/third_party/ubp/things-eeg'
IMG = '/path/to/eegaudit/data_raw/things_eeg2_images/images_set'
OUT = '/path/to/eegaudit/results/baselines/dtha'; os.makedirs(OUT, exist_ok=True)
clip = {}
for mode in ('train', 'test'):
    fb = safe_load(f'{U}/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_{mode}.pt')['img_features']
    dt = safe_load(f'{U}/Image_feature/DirectT/eeg_intra-subject_baseline_EEGProjectLayer_RN50_{mode}.pt')['img_features']
    keys = sorted(fb['medium'].keys())
    clip[mode] = (keys, torch.stack([torch.stack([F.normalize(d[k].float(), dim=0) for d in (dt, fb['low'], fb['medium'], fb['high'])]) for k in keys]).half())
    print(mode, len(keys), keys[:2], f'{time.time()-t0:.0f}s', flush=True)
proc = AutoImageProcessor.from_pretrained('/path/to/eegaudit/data_raw/third_party/dinov2_large')
mdl = Dinov2Model.from_pretrained('/path/to/eegaudit/data_raw/third_party/dinov2_large', torch_dtype=torch.float16).to(dev).eval()
from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(int(os.environ.get('OMP_NUM_THREADS', '4')))
def prep(ks):  # decode + preprocess 16 images in a worker thread (PIL releases the GIL while decoding)
    ims = [Image.open(os.path.join(IMG, k.replace('train_images/', 'training_images/', 1))).convert('RGB') for k in ks]
    return proc(images=ims, return_tensors='pt')['pixel_values']
out = {'note': __doc__}
for mode in ('train', 'test'):
    keys = clip[mode][0]; feats = []
    for i in range(0, len(keys), 128):
        x = torch.cat(list(POOL.map(prep, [keys[j:j + 16] for j in range(i, min(i + 128, len(keys)), 16)]))).to(dev).half()
        with torch.no_grad():
            hs = mdl(pixel_values=x, output_hidden_states=True).hidden_states   # 25 = embeddings + 24 blocks
        feats.append(F.normalize(torch.stack([hs[l + 1][:, 0].float() for l in range(15)], 1), dim=-1).half().cpu())
        if i % 2560 == 0:
            print(mode, i, f'{time.time()-t0:.0f}s', flush=True)
    out[f'{mode}_imgs'] = keys; out[f'dino_{mode}'] = torch.cat(feats); out[f'clip_{mode}'] = clip[mode][1]
torch.save(out, f'{OUT}/teacher_features.pt')
print('wrote', {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, f'{time.time()-t0:.0f}s', flush=True)
