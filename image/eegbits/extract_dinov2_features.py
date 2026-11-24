"""DINOv2 ViT-B/14 image features for THINGS-EEG2 (target swap for the read-out-bounded test): CLS embedding (768-d, L2-normalised,
same preprocessing as the benchmark's extract_image_features.py), for the training images, test images and centre images, in the
order NICE uses (all image paths under a partition, sorted). Writes the three files the NICE training script expects for --dnn dinov2:
  <out>/dinov2_feature_maps_training.npy (16540, 768), <out>/dinov2_feature_maps_test.npy (200, 768), <out>/center_dinov2.npy (200, 768)
Sanity check (printed): ridge from DINOv2 to CLIP on the training images must identify the 200 test images from their DINOv2 features at
well above chance; an ordering mismatch between the image files and the released CLIP features would make it fail.
Usage: python extract_dinov2_features.py <image_root> <out_dir>"""
import glob, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision import transforms
root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
HUB = "/path/to/image_benchmark/models/facebookresearch_dinov2_main"; CKPT = "/path/to/image_benchmark/models/checkpoints/dinov2_vitb14_pretrain.pth"
NICE = Path("/path/to/image_benchmark/models/nice/external/NICE-EEG/dnn_feature")
dev = torch.device("cuda"); t0 = time.time()
model = torch.hub.load(HUB, "dinov2_vitb14", source="local", pretrained=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu")); model = model.to(dev).eval()
prep = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(),
                           transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def find_partition(name):
    hits = [p for p in root.rglob(name) if p.is_dir()]
    assert len(hits) == 1, (name, hits); return hits[0]


@torch.no_grad()
def feats(paths, bs=128):
    out_ = []
    for i in range(0, len(paths), bs):
        x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        f = model(x).float(); f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12); out_.append(f.cpu().numpy())
    return np.concatenate(out_).astype(np.float32)


res = {}
for part, fname, n_expect in [("training_images", "dinov2_feature_maps_training.npy", 16540), ("test_images", "dinov2_feature_maps_test.npy", 200), ("center_images", "center_dinov2.npy", 200)]:
    try:
        d = find_partition(part)
    except AssertionError as e:
        print("partition missing:", e); continue
    paths = sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert len(paths) == n_expect, (part, len(paths))
    F = feats(paths); np.save(out / fname, F); res[part] = F
    print(part, F.shape, f"({time.time() - t0:.0f}s)", flush=True)
if "center_images" not in res and "test_images" in res:
    np.save(out / "center_dinov2.npy", res["test_images"]); print("no centre images in the archive: center_dinov2.npy = test-image features")
# ---- sanity: ordering against the released CLIP features
Ctr = np.squeeze(np.load(NICE / "clip_feature_maps_training.npy", allow_pickle=True)).astype(np.float32); Cte = np.squeeze(np.load(NICE / "clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32)
A = torch.from_numpy(res["training_images"]).to(dev); Y = torch.from_numpy(Ctr).to(dev)
W = torch.linalg.solve(A.T @ A + 1e-2 * torch.eye(A.shape[1], device=dev), A.T @ Y)
P = (torch.from_numpy(res["test_images"]).to(dev) @ W).cpu().numpy(); P /= np.linalg.norm(P, axis=1, keepdims=True); Cn = Cte / np.linalg.norm(Cte, axis=1, keepdims=True)
top1 = float((np.argmax(P @ Cn.T, axis=1) == np.arange(200)).mean())
print(f"SANITY ridge DINOv2->CLIP: test top-1 {100*top1:.1f}% (chance 0.5%; ordering mismatch would give ~chance)", flush=True)
assert top1 > 0.3, "ordering check failed"
