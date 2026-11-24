import torch, numpy as np, sys
D='/path/to/eegaudit/data_raw/third_party/ubp'
sys.path.insert(0,'/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load
def tl(p):
    return safe_load(p), 'weights_only=True (+numpy globals only)'
c, how = tl(f'{D}/ckpt/sub-01/last.ckpt'); print('ckpt', how, list(c.keys()))
print({k: tuple(v.shape) for k, v in c['state_dict'].items()})
f, how = tl(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt'); print('feat', how, f.keys(), type(f['img_features']), list(f['img_features'].keys()))
k0 = list(f['img_features']['medium'].keys())[:3]; print(k0, f['img_features']['medium'][k0[0]].shape)
t, how = tl(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt'); print('test', how, {k: (getattr(v, 'shape', None), getattr(v, 'dtype', None)) for k, v in t.items()})
print(t['img'][:2, :2], t['session'][:2, :] if t['session'].ndim > 1 else t['session'][:10], t['times'][:5], t['times'][-3:])
print('img order sorted?', list(t['img'][:, 0]) == sorted(t['img'][:, 0]), 'sessions per row', np.unique(t['session'][0], return_counts=True) if t['session'].ndim > 1 else '')
