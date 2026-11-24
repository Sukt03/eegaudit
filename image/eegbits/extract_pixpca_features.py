"""Pixel-space target for the target-swap test (ideation card A1): 64x64 RGB pixels of each THINGS-EEG2 image, PCA to 768 dimensions
(basis fitted on the training images only), L2-normalised; same ordering and file names as extract_dinov2_features.py so the NICE
training script can use --dnn pixpca.  Usage: python extract_pixpca_features.py <image_root> <out_dir>"""
import sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); t0 = time.time()


def partition(name):
    hits = [p for p in root.rglob(name) if p.is_dir()]; assert len(hits) == 1, (name, hits); return hits[0]


def pixels(d):
    paths = sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.BICUBIC), dtype=np.float32).ravel() / 255.0 for p in paths]), len(paths)


Xtr, ntr = pixels(partition("training_images")); Xte, nte = pixels(partition("test_images")); assert (ntr, nte) == (16540, 200), (ntr, nte)
dev = torch.device("cuda"); A = torch.from_numpy(Xtr).to(dev); mu = A.mean(0, keepdim=True)
_, _, Vh = torch.linalg.svd(A - mu, full_matrices=False); V = Vh[:768].T
def proj(X):
    Z = ((torch.from_numpy(X).to(dev) - mu) @ V); Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12); return Z.cpu().numpy().astype(np.float32)
Ftr, Fte = proj(Xtr), proj(Xte)
np.save(out / "pixpca_feature_maps_training.npy", Ftr); np.save(out / "pixpca_feature_maps_test.npy", Fte); np.save(out / "center_pixpca.npy", Fte)
print("pixpca", Ftr.shape, Fte.shape, f"({time.time() - t0:.0f}s)")
