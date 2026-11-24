"""I/O helper (EXECUTOR-2): write THINGS-EEG2 preprocessed epochs as float32 .npy (train (16540, 4, 63, 250), test (200, 80, 63, 250)) so that
GPU jobs on the 30-GB cluster node can memory-map them instead of unpickling 8.3 GB of float64. Values = np.float32 of the original (as run.load_raw).
Usage: python things_f32_cache.py 1 2 3"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
import run as R
OUT = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/cache_f32")
for s in sys.argv[1:]:
    sub = f"sub-{int(s):02d}"
    if (OUT / f"{sub}_train.npy").exists(): continue
    Xtr, Xte, meta = R.load_raw(sub)
    np.save(OUT / f"{sub}_test.npy", Xte); np.save(OUT / f"{sub}_train.tmp.npy", Xtr); (OUT / f"{sub}_train.tmp.npy").rename(OUT / f"{sub}_train.npy")
    print(sub, Xtr.shape, Xte.shape, meta["ch_names"][:3], flush=True)
