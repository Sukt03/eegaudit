"""Frozen EEG foundation models as a rung of the visual ladder (same rung family as the decision ladder's CBraMod / LaBraM /
EEGPT pooled embeddings), on THINGS-EEG2 with the official preprocessing and split.

Convention (field-standard for each model, recorded in the output):
  data      : official THINGS-EEG2 preprocessed epochs (Gifford et al. 2022 pipeline as released with the ATM repo), 63 channels,
              250 Hz, 0-1.0 s after onset, whitened; training images = mean of their 4 repetitions, test = mean of k repetitions.
  cbramod   : 200 Hz, one 1-s patch per channel, inputs scaled to unit variance per channel (the uV/100 convention on whitened
              data), tokens = model output (B, C, 1, 200) with proj_out removed; features = the output tokens, flattened (unpooled; channel-pooled summaries were at chance on a pilot).
  labram    : 200 Hz, channels passed by 10-20 name, features = all output tokens, flattened (braindecode port, fm_zoo.load).
  eegpt     : 256 Hz, the checkpoint's 62-channel layout filled by name (unmatched channels zero), features = all output tokens, flattened.
Decoder   : ridge from standardized features to CLIP ViT-H/14 image embeddings (penalty picked on a held-out tenth of the
            training images by MRR, test never used), 200-way identification on the official test images, metrics of run.py.
Controls  : noise input of the same shape through the same model and ridge; shuffled pairs (predictions vs permuted targets).
Output    : results/eegbits/fm_probe_<model>_<subject>.json
Usage     : python run_fm_probe.py --model cbramod --subject sub-01 [--slice 3000]
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np, torch
from scipy.signal import resample
ROOT = Path("/path/to/eegaudit")
sys.path.insert(0, str(ROOT / "experiments/eegbits")); sys.path.insert(0, str(ROOT))
from run import CLIP, load_torch, load_raw, metrics
OUT = ROOT / "outputs/AGENT_LOOP_20261120/results/q7"   # loop R6 copy of run_fm_probe.py: same pipeline, new output dir (existing fm_probe files untouched)
SF = {"cbramod": 200, "labram": 200, "eegpt": 256, "reve": 200}


def build(model_name, chans, dev):
    if model_name == "reve":
        from transformers import AutoModel
        positions = AutoModel.from_pretrained(str(ROOT / "external/reve/positions"), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(dev).eval()
        model = AutoModel.from_pretrained(str(ROOT / "external/reve/base"), trust_remote_code=True, local_files_only=True, use_safetensors=True).to(dev).eval()
        names = list(chans)
        assert all(c in positions.mapping for c in names), "REVE channel position unavailable; no silent padding or dropping"
        pos = positions(names).to(dev)
        assert pos.shape == (len(chans), 3)
        def feats(x):
            tokens = model(x, pos.expand(len(x), -1, -1))
            assert tokens.shape[1] == len(chans) and tokens.shape[2] == 1
            return tokens.reshape(len(x), -1)
        return feats, None
    if model_name == "cbramod":
        sys.path.insert(0, str(ROOT / "external/CBraMod"))
        from models.cbramod import CBraMod
        m = CBraMod().to(dev)
        m.load_state_dict(torch.load(ROOT / "external/CBraMod/pretrained_weights/pretrained_weights.pth", map_location=dev, weights_only=True))
        m.proj_out = torch.nn.Identity(); m.eval()

        def feats(x):                                            # x (B, C, 200) -> tokens (B, C, 200), flattened (unpooled)
            f = m(x[:, :, None, :])[:, :, 0, :]
            return f.reshape(f.shape[0], -1)
        return feats, None
    import os
    os.chdir(ROOT)                                               # fm_zoo reads external/<model>/config.json relative to the project root
    import fm_zoo
    m, _ = fm_zoo.load(model_name, chans, SF[model_name], dev)
    names62 = None
    if model_name == "eegpt":
        names62 = [c["ch_name"].upper() for c in json.load(open(ROOT / "external/eegpt/config.json"))["chs_info"]]

    def feats(x):
        if model_name == "labram":                               # patch tokens (B, n_patches = channels x 1-s patches, 200), not the pooled head
            t = m(x, return_patch_tokens=True, ch_names=[c.upper() for c in chans])
        else:
            t = fm_zoo.tokens(m, x, model_name, chans)
        if isinstance(t, (tuple, list)):
            t = t[0]
        if isinstance(t, dict):
            t = t.get("features", next(v for v in t.values() if hasattr(v, "dim")))
        assert t.dim() >= 3 and t.shape[1] > 1, f"{model_name}: expected unpooled tokens, got {tuple(t.shape)}"
        return t.reshape(t.shape[0], -1)                         # all output tokens, flattened (unpooled)
    return feats, names62


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SF)); ap.add_argument("--subject", required=True)
    ap.add_argument("--slice", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); dev = torch.device("cuda"); rng = np.random.default_rng(a.seed); torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    Xtr4, Xte80, meta = load_raw(a.subject); chans = meta["ch_names"]
    assert Xtr4.shape[2] == 63 and Xtr4.shape[3] == 250 and len(chans) == 63, meta
    Ytr, Yte = load_torch(CLIP / "ViT-H-14_features_train.pt"), load_torch(CLIP / "ViT-H-14_features_test.pt")
    Xtr = Xtr4.mean(1)
    if a.slice:
        sel = rng.choice(len(Xtr), a.slice, replace=False); Xtr, Ytr = Xtr[sel], Ytr[sel]
    sf = SF[a.model]; n_t = int(round(250 * sf / 250)) if sf == 250 else sf      # 1.0 s at the model's rate
    def prep(X):                                                  # (n, 63, 250) -> (n, 63, n_t), model rate, unit variance per channel
        return resample(X, n_t, axis=2).astype(np.float32)
    Ztr = prep(Xtr); mu, sd = Ztr.mean(axis=(0, 2), keepdims=True), Ztr.std(axis=(0, 2), keepdims=True) + 1e-6
    std = lambda Z: (Z - mu) / sd
    feats, names62 = build(a.model, chans, dev)
    up = [c.upper() for c in chans]
    def to_model(Z):
        if names62 is None:
            return Z
        Z62 = np.zeros((len(Z), len(names62), Z.shape[2]), np.float32); hit = 0
        for k, nm in enumerate(names62):
            if nm in up:
                Z62[:, k] = Z[:, up.index(nm)]; hit += 1
        to_model.hit = hit
        return Z62

    @torch.no_grad()
    def embed(Z, bs=64):
        Z = to_model(std(Z)); out = []
        for b in range(0, len(Z), bs):
            out.append(feats(torch.from_numpy(np.ascontiguousarray(Z[b:b + bs])).to(dev)).float().cpu().numpy())
        F = np.concatenate(out); assert np.isfinite(F).all(), "non-finite features"
        return F

    Ftr = embed(Ztr)
    if names62 is not None:
        print(f"EEGPT channels matched by name: {to_model.hit} of {len(names62)}", flush=True)
    Fte80 = embed(prep(Xte80.mean(1)))
    Fte1 = [embed(prep(Xte80[:, rng.choice(80, 1, replace=False)].mean(1))) for _ in range(5)]
    Fnoise_tr = embed(rng.standard_normal(Ztr.shape).astype(np.float32) * sd + mu)   # noise at the data's scale, same pipeline
    Fnoise_te = embed(rng.standard_normal((200,) + Ztr.shape[1:]).astype(np.float32) * sd + mu)

    def ridge_fit_predict(Ftr, Ytr, Ftes):
        """Dual (kernel) ridge on the GPU for D >> n or D ~ n; alpha chosen on a held-out tenth of the training rows by MRR;
        refit on all rows. Features standardized with training statistics."""
        fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.from_numpy((Ftr - fm) / fs).to(dev); Ym = Ytr.mean(0); Y = torch.from_numpy(Ytr - Ym).to(dev)
        n = A.shape[0]; idx = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed)).to(dev); cut = int(0.9 * n)
        tr, va = idx[:cut], idx[cut:]
        K = A @ A.T; D = A.shape[1]
        best, best_a = -1, None
        for alpha in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            Ktr = K[tr][:, tr]; coef = torch.linalg.solve(Ktr + alpha * D * torch.eye(len(tr), device=dev), Y[tr])
            s = metrics((K[va][:, tr] @ coef).cpu().numpy() + Ym, Ytr[va.cpu().numpy()])["mrr"]
            if s > best: best, best_a = s, alpha
        coef = torch.linalg.solve(K + best_a * D * torch.eye(n, device=dev), Y)
        preds = [((torch.from_numpy((F - fm) / fs).to(dev) @ A.T) @ coef).cpu().numpy() + Ym for F in Ftes]
        del K; torch.cuda.empty_cache()
        return best_a, preds

    rows = []
    alpha, (p80, *p1s) = ridge_fit_predict(Ftr, Ytr, [Fte80] + Fte1)
    rows.append(dict(condition="eeg", k=80, draw=0, **metrics(p80, Yte)))
    for dr, p1 in enumerate(p1s):
        rows.append(dict(condition="eeg", k=1, draw=dr, **metrics(p1, Yte)))
    rows.append(dict(condition="shuffled_pair", k=80, draw=0, **metrics(p80, Yte[rng.permutation(200)])))
    alpha_s, (ps,) = ridge_fit_predict(Ftr[rng.permutation(len(Ftr))], Ytr, [Fte80])          # trial-shuffled: EEG rows permuted against targets at training
    rows.append(dict(condition="trial_shuffled", k=80, draw=0, **metrics(ps, Yte)))
    alpha_n, (pn,) = ridge_fit_predict(Fnoise_tr, Ytr, [Fnoise_te])
    rows.append(dict(condition="noise", k=80, draw=0, **metrics(pn, Yte)))
    assert all(np.isfinite(r['soft_bits']) for r in rows), 'nonfinite score'
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'external/reve/base/model.safetensors', ROOT/'external/reve/base/config.json', ROOT/'external/reve/base/modeling_reve.py', ROOT/'external/reve/positions/model.safetensors', ROOT/'external/reve/positions/config.json', ROOT/'external/reve/positions/position_bank.py']} if a.model == 'reve' else {}
    out = dict(gpt_legacy_completion=True, seed=a.seed, hashes=hashes, software=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0)), target_space="CLIP ViT-H/14, identical 200 official test candidates", original_script="experiments/loop/fm_probe_r6.py", reve_convention="200Hz Fourier-resampled 1s epochs; train mean/std per-channel; all63 named positions required; flattened ch x1patch x512 tokens; local inspected code and safetensors", limitations="Internal-dev feature standardization uses all official training rows as original r6; external test never used for tuning. k1 controls absent; descriptive k1 only. NICE native ViTL results are separate target space.", model=a.model, subject=a.subject, n_train=int(len(Ftr)), slice=a.slice, feature_dim=int(Ftr.shape[1]), sample_rate=sf,
               alpha=alpha, alpha_noise=alpha_n, channels_matched=getattr(to_model, "hit", 63),
               convention=__doc__.split("Convention")[1].split("Decoder")[0].strip(), split="THINGS-EEG2 official train/test; test never used for selection",
               rows=rows, runtime_s=round(time.time() - t0, 1))
    tag = f"fm_probe_{a.model}_{a.subject}" + (f"_slice{a.slice}" if a.slice else "")
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1))
    e, n_, s_ = rows[0], rows[-1], rows[-2]
    print(f"{a.model} {a.subject} n={len(Ftr)} D={Ftr.shape[1]} alpha={alpha:g}: k=80 2-way {100*e['two_way']:.1f}% top1 {100*e['top1']:.1f}% bits {e['bits']:.2f} | k=1 2-way {100*np.mean([r['two_way'] for r in rows if r['condition']=='eeg' and r['k']==1]):.1f}% | noise 2-way {100*n_['two_way']:.1f}% | shuffled-pair 2-way {100*s_['two_way']:.1f}%  ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
