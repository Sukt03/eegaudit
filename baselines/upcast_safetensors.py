"""Upcast every floating tensor of a safetensors file to float32 in place (BART-large release is fp16; train_bt.py feeds float32)."""
import sys, torch
from safetensors.torch import load_file, save_file
p = sys.argv[1]; sd = load_file(p)
save_file({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, p, metadata={'format': 'pt'}); print('upcast', len(sd), 'tensors')
