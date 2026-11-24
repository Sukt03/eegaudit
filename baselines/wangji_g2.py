"""G2 under the PAPERS' OWN protocol for the ZuCo word-feature decoders (IMPLEMENTER, 21 Nov 2026; coordinator "reimplement again").
Purpose: check that our re-implementations are FAITHFUL; our leak-free split stays the evaluation split of the card. No test tuning.
Protocol = MikeWangWZHL/EEG-To-Text (read: model_decoding.py, train_decoding.py, data.py, eval_decoding.py, metrics.py):
- data: SR + NR + NR2 pickles, reference split = first 80 / 10 / 10 % of sentence indices per task, duplicates KEPT
  (results/track_text/brain_translator/cache_full_reference_split.pkl via prep_cache.py --keep_dups; their TSR task is not on disk);
  each word's 840-d GD vector z-scored on its own (data.normalize_1d); max 56 words; targets tokenised to max_length 56.
- BrainTranslator: 6-layer TransformerEncoder(d 840, 8 heads, ff 2048, batch_first) -> Linear(840, 1024) -> ReLU -> BART-large.
  DeWave-style: the same pre-encoder + VQ codex (K 2048, d 512) before the projection (our DeWave-style, as train_dewave.py).
  EEG2Text-style: the 10-view encoder of e2t_style.py (+ 10 epochs re-masked pre-training) before the projection.
- training (all three; their 2-step recipe): seed 312; step 1 = SGD momentum 0.9, lr 5e-5, StepLR(20, 0.1), 20 epochs, trainable =
  everything except BART, plus BART 'shared', 'embed_positions', 'encoder.layers.0'; step 2 = all parameters, SGD lr 5e-7, StepLR(30, 0.1),
  30 epochs; batch 32; best model on dev loss (per-sample mean, as their bs-1 dev loader), step 2 starts from the step-1 best.
- metrics on the TEST split: teacher-forced (argmax of logits given the gold prefix) AND free-running (5 beams, repetition_penalty 5,
  max 100, as eval_decoding.py); (a) PAPER-2022 metric: corpus BLEU-1..4 over BART BPE tokens (convert_ids_to_tokens, truncated at the
  first </s>, nltk corpus_bleu semantics, no smoothing); (b) the current repo metric (metrics.compute_metrics: torchmetrics BLEUScore on
  decoded strings, ROUGE-1 F x 100). G2 targets: BrainTranslator BLEU-1 40.1, DeWave 41.3, EEG2Text 45.2 (teacher-forced)."""
import argparse, json, math, os, pickle, sys, time, copy
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['bt', 'dewave', 'e2t'], required=True)
ap.add_argument('--cache', default='/path/to/eegaudit/results/track_text/brain_translator/cache_full_reference_split.pkl')
ap.add_argument('--bart', default='/path/to/eegaudit/data_raw/third_party/bart_large')
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/text/g2_paper_protocol')
ap.add_argument('--ep1', type=int, default=20); ap.add_argument('--ep2', type=int, default=30); ap.add_argument('--pre_epochs', type=int, default=10)
ap.add_argument('--max_train', type=int, default=None); ap.add_argument('--max_eval', type=int, default=None); ap.add_argument('--tag', default='')
a = ap.parse_args()
torch.manual_seed(312); np.random.seed(312); torch.cuda.manual_seed_all(312)
dev = torch.device('cuda'); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
TAG = f'g2_{a.model}{a.tag}'; logf = open(os.path.join(a.out, TAG + '.log'), 'w')
def P(*x):
    m = ' '.join(str(v) for v in x); print(m, flush=True); logf.write(m + '\n'); logf.flush()

D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']; del D
split = {s: [r for r in recs if r['split'] == s] for s in ('train', 'dev', 'test')}
if a.max_train:
    split['train'] = split['train'][:a.max_train]; split['dev'] = split['dev'][: a.max_train // 8]
if a.max_eval:
    split['test'] = split['test'][:a.max_eval]
from transformers import BartTokenizer, BartForConditionalGeneration
tok = BartTokenizer.from_pretrained(a.bart)


def collate(batch):
    n = 56; X = torch.zeros(len(batch), n, 840); M = torch.zeros(len(batch), n, dtype=torch.long)
    for i, r in enumerate(batch):
        w = torch.from_numpy(r['words'][:n]).double()
        w = (w - w.mean(1, keepdim=True)) / w.std(1, keepdim=True)          # data.normalize_1d per word (torch.std = unbiased)
        X[i, :len(w)] = w.float(); M[i, :len(w)] = 1
    enc = tok([texts[r['g']] for r in batch], padding='max_length', max_length=56, truncation=True, return_tensors='pt')
    return X, M, enc['input_ids'], [r['g'] for r in batch], [r['subject'] for r in batch]


loader = lambda rs, sh, bs: torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=sh, collate_fn=collate, num_workers=0)


class VQ(nn.Module):
    def __init__(self, K=2048, d=512, beta=0.25):
        super().__init__(); self.emb = nn.Embedding(K, d); nn.init.uniform_(self.emb.weight, -1 / K, 1 / K); self.beta = beta
    def forward(self, z, M):
        flat = z.reshape(-1, z.shape[-1]); d2 = (flat ** 2).sum(1, keepdim=True) - 2 * flat @ self.emb.weight.t() + (self.emb.weight ** 2).sum(1)[None]
        q = self.emb(d2.argmin(1)).view_as(z); m = M.float().unsqueeze(-1)
        loss = ((((q - z.detach()) ** 2) * m).sum() + self.beta * (((z - q.detach()) ** 2) * m).sum()) / m.sum().clamp(min=1) / z.shape[-1]
        return z + (q - z).detach(), loss


VIEWS = [[6, 12, 5, 11, 16, 15, 20, 118, 24, 124, 26, 2, 27, 123, 3, 4, 23, 19, 22, 9, 10, 18, 28, 33, 117, 122],
         [129, 7, 106, 105, 104, 115, 114, 120, 110, 116, 121, 111, 112, 109, 13, 30], [29, 36, 35, 34], [40, 38, 39, 43, 44, 46, 57, 58, 64],
         [31, 80, 55, 37, 87, 93, 103, 102, 108], [54, 79, 61, 78, 62, 53, 86, 92, 98, 100, 101], [67, 77, 71, 72, 76, 66, 84, 60, 85],
         [59, 91, 97, 51], [41, 42, 52, 47, 45, 50], [65, 69, 70, 74, 75, 82, 83, 89, 90, 95, 96]]
ORDER = sorted(e for v in VIEWS for e in v); VIEW_IDX = [torch.tensor([b * 105 + ORDER.index(e) for b in range(8) for e in v]) for v in VIEWS]


class Decoder(nn.Module):
    def __init__(self, bart, kind):
        super().__init__(); self.kind = kind; self.pretrained = bart
        if kind == 'e2t':
            self.inp = nn.ModuleList([nn.Linear(len(ix), 128) for ix in VIEW_IDX])
            self.views = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 8, 512, batch_first=True), 1) for _ in VIEWS])
            self.mix = nn.Linear(128 * len(VIEWS), 840); self.rec = nn.Linear(840, 840)
        self.additional_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=840, nhead=8, dim_feedforward=2048, batch_first=True), num_layers=6)
        if kind == 'dewave':
            self.to_code = nn.Linear(840, 512); self.vq = VQ(); self.fc1 = nn.Linear(512, bart.config.d_model)
        else:
            self.fc1 = nn.Linear(840, bart.config.d_model)
    def encode(self, X, M):
        pad = (M == 0)
        if self.kind == 'e2t':
            X = self.mix(torch.cat([v(f(X[..., ix.to(X.device)]), src_key_padding_mask=pad) for f, v, ix in zip(self.inp, self.views, VIEW_IDX)], -1))
        return self.additional_encoder(X, src_key_padding_mask=pad)
    def embed(self, X, M):
        h = self.encode(X, M); vql = h.new_zeros(())
        if self.kind == 'dewave':
            h, vql = self.vq(self.to_code(h), M)
        return F.relu(self.fc1(h)), vql
    def forward(self, X, M, labels):
        e, vql = self.embed(X, M); out = self.pretrained(inputs_embeds=e, attention_mask=M, labels=labels); return out, vql


model = Decoder(BartForConditionalGeneration.from_pretrained(a.bart), a.model).to(dev)
P(f'{TAG}: train/dev/test records {len(split["train"])}/{len(split["dev"])}/{len(split["test"])}; params {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

if a.model == 'e2t' and a.pre_epochs:   # EEG2Text re-masked token pre-training of the encoder (before step 1)
    encp = [p for n, p in model.named_parameters() if not n.startswith(('pretrained.', 'fc1.'))]
    mtok = nn.Parameter(torch.zeros(840, device=dev)); popt = torch.optim.AdamW(encp + [mtok], lr=1e-4, weight_decay=0.01)
    for ep in range(a.pre_epochs):
        tot = 0; nb = 0
        for X, M, L, *_ in loader(split['train'], True, 32):
            X, M = X.to(dev), M.to(dev); mk = (torch.rand(M.shape, device=dev) < 0.15) & (M == 1)
            rec = model.rec(model.encode(torch.where(mk[..., None], mtok.expand_as(X), X), M))
            loss = ((rec - X) ** 2)[mk].mean() if mk.any() else rec.sum() * 0
            popt.zero_grad(); loss.backward(); popt.step(); tot += loss.item(); nb += 1
        P(f'pretrain {ep+1}/{a.pre_epochs} mse {tot/nb:.4f} ({time.time()-t0:.0f}s)')


def dev_loss():
    model.eval(); tot = 0; n = 0
    with torch.no_grad():
        for X, M, L, *_ in loader(split['dev'], False, 32):
            X, M, L = X.to(dev), M.to(dev), L.to(dev); L = L.masked_fill(L == tok.pad_token_id, -100)
            out, vql = model(X, M, L)
            lt = F.cross_entropy(out.logits.transpose(1, 2), L, reduction='none', ignore_index=-100)   # per-sample mean = their bs-1 loss
            ps = lt.sum(1) / (L != -100).sum(1).clamp(min=1); tot += (ps + vql).sum().item(); n += len(ps)
    model.train(); return tot / n


def run_step(epochs, lr, step_size, trainable):
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(trainable(n_))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.1)
    best = (float('inf'), copy.deepcopy(model.state_dict()), 0)
    for ep in range(epochs):
        model.train(); run = 0; nb = 0
        for X, M, L, *_ in loader(split['train'], True, 32):
            X, M, L = X.to(dev), M.to(dev), L.to(dev); L = L.masked_fill(L == tok.pad_token_id, -100)
            out, vql = model(X, M, L); loss = out.loss + vql
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item(); nb += 1
        sch.step(); dl = dev_loss()
        if dl < best[0]:
            best = (dl, copy.deepcopy(model.state_dict()), ep + 1)
        P(f'  epoch {ep+1}/{epochs} train {run/nb:.4f} dev {dl:.4f} best ep {best[2]} ({time.time()-t0:.0f}s)')
    model.load_state_dict(best[1]); return best[0], best[2]


keep1 = lambda n: (not n.startswith('pretrained.')) or ('shared' in n) or ('embed_positions' in n) or ('encoder.layers.0.' in n)
P('=== step 1'); b1 = run_step(a.ep1, 5e-5, 20, keep1)
P('=== step 2'); b2 = run_step(a.ep2, 5e-7, 30, lambda n: True)


# ---------------- metrics
def corpus_bleu_tokens(hyps, refs, n_max):
    """nltk corpus_bleu with one reference, weights uniform over 1..n_max, no smoothing."""
    num = [0] * n_max; den = [0] * n_max; hl = rl = 0
    for h, r in zip(hyps, refs):
        hl += len(h); rl += len(r)
        for n in range(1, n_max + 1):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1)); rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items()); den[n - 1] += max(0, len(h) - n + 1)
    if min(num) == 0 or hl == 0:
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return bp * math.exp(sum(math.log(num[i] / den[i]) for i in range(n_max)) / n_max)


from torchmetrics.text import BLEUScore
from torchmetrics.functional.text.rouge import rouge_score


def all_metrics(pred_ids, tgt_ids):
    def toks_of(ids):
        out = []
        for t in ids:
            if t == tok.eos_token_id:
                break
            out.append(t)
        return tok.convert_ids_to_tokens(out, skip_special_tokens=True)
    ph = [toks_of(p) for p in pred_ids]; rf = [tok.convert_ids_to_tokens([t for t in r if t != tok.pad_token_id], skip_special_tokens=True) for r in tgt_ids]
    res = {f'paper2022_token_bleu{n}': corpus_bleu_tokens(ph, rf, n) for n in (1, 2, 3, 4)}
    ps = tok.batch_decode(pred_ids, skip_special_tokens=True); ts = tok.batch_decode(tgt_ids, skip_special_tokens=True)
    for n in (1, 2, 3, 4):
        res[f'repo_string_bleu{n}'] = float(BLEUScore(n_gram=n)(ps, [[t] for t in ts]))
    rf1 = [float(rouge_score(p, t, rouge_keys='rouge1')['rouge1_fmeasure']) for p, t in zip(ps, ts)]
    res['repo_rouge1_f_x100'] = 100 * float(np.mean(rf1)); return res, ps, ts


model.eval(); tf_ids, fr_ids, tg_ids, rows = [], [], [], []
with torch.no_grad():
    for X, M, L, G, S in loader(split['test'], False, 32):
        X, M, L = X.to(dev), M.to(dev), L.to(dev); Lm = L.masked_fill(L == tok.pad_token_id, -100)
        out, _ = model(X, M, Lm); tf_ids += out.logits.argmax(-1).tolist(); tg_ids += L.tolist()
        e, _ = model.embed(X, M)
        fr_ids += [x for x in model.pretrained.generate(inputs_embeds=e, attention_mask=M, max_length=100, num_beams=5, do_sample=False, repetition_penalty=5.0).tolist()]
        rows += [{'g': g, 'subject': s} for g, s in zip(G, S)]
m_tf, ps_tf, ts = all_metrics(tf_ids, tg_ids); m_fr, ps_fr, _ = all_metrics(fr_ids, tg_ids)
target = {'bt': 0.401, 'dewave': 0.413, 'e2t': 0.452}[a.model]
res = {'note': __doc__, 'cfg': vars(a), 'tag': TAG, 'n_test_records': len(ts), 'best_dev_loss_step1': b1[0], 'best_epoch_step1': b1[1],
       'best_dev_loss_step2': b2[0], 'best_epoch_step2': b2[1], 'teacher_forced': m_tf, 'free_running': m_fr, 'published_tf_bleu1': target,
       'G2_paper2022_tf_bleu1_rel_diff': (m_tf['paper2022_token_bleu1'] - target) / target, 'runtime_s': time.time() - t0}
res['G2_pass_10pct'] = abs(res['G2_paper2022_tf_bleu1_rel_diff']) <= 0.10
res['code_sha256'] = CODE_SHA256; json.dump(res, open(os.path.join(a.out, TAG + '.json'), 'w'), indent=1)
with open(os.path.join(a.out, TAG + '_generations.jsonl'), 'w') as f:
    for r, p1, p2, t in zip(rows, ps_tf, ps_fr, ts):
        f.write(json.dumps(dict(r, target=t, teacher_forced=p1, beam=p2)) + '\n')
try:
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, TAG + '_weights_fp16.pt'))
except Exception as e:
    P('weight save failed', repr(e))
P(f'{TAG}: TF paper-2022 token BLEU-1 {m_tf["paper2022_token_bleu1"]:.4f} (published {target}), repo string BLEU-1 {m_tf["repo_string_bleu1"]:.4f}, '
  f'ROUGE-1 F {m_tf["repo_rouge1_f_x100"]:.2f} | free-running token BLEU-1 {m_fr["paper2022_token_bleu1"]:.4f} | G2 (10 %) {res["G2_pass_10pct"]} ({time.time()-t0:.0f}s)')
