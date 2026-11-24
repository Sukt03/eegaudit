"""Convert a downloaded pytorch_model.bin to model.safetensors with torch.load(weights_only=True) (tensors only; no code runs)."""
import sys, torch
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location='cpu', weights_only=True)
sd = {k: v.contiguous() for k, v in sd.items()}
# BART ties shared/encoder/decoder embeddings and lm_head; keep one copy of shared memory
seen = {}; out = {}
for k, v in sd.items():
    key = (v.data_ptr(), v.shape)
    out[k] = v.clone() if key in seen else v; seen[key] = k
save_file(out, dst, metadata={'format': 'pt'}); print('wrote', dst, len(out), 'tensors')
