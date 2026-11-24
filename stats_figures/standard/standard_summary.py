"""Standard test summary (2026-11-20). Collects per-target yield / floor / ceiling / fraction / L90 under the common log-score, and the
two open-weight image models (task 4). Image log-score = InfoNCE soft_bits over the 200 THINGS-EEG2 test candidates (bits per query).
BrainFLORA native: outputs/AGENT_LOOP_20261120/results/u1_native_v2/sub-XX.json (released checkpoint, native ViT-H candidates; k=1 mean of
5 repetition draws; floors = shuffled-pair mean of 100 candidate permutations and Gaussian-noise input; ceiling = split-half k40 vs k40
through the same model). REVE Q7: outputs/AGENT_LOOP_20261120/results/q7/fm_probe_reve_sub-XX.json (frozen REVE + ridge, ViT-H target;
k=1 has no control in Q7, floors at k=80 only; no split-half ceiling in Q7). NICE-GA (ours) k40 and its ceiling: results/loop/s3_fraction.
Image L90: Q8 PCA bottleneck of NICE-GA input at k=80 (widths 25..2000 + full). Sentence L90: X6 PCA curve (eeg nce). Paired participant
bootstrap, 10000 resamples, seed 20261120."""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path('/path/to/eegaudit'); A = ROOT / 'outputs/AGENT_LOOP_20261120/results'; S = ROOT / 'results/standard'
rng = np.random.default_rng(20261120); B = 10000
rd = lambda p: json.loads(Path(p).read_text())


def boot(v):
    v = np.asarray(v, float); ix = rng.integers(0, len(v), size=(B, len(v))); return np.quantile(v[ix].mean(1), [.025, .975]).tolist()


def stat(v, floor=None):
    v = np.asarray(v, float); o = dict(n=len(v), mean=float(v.mean()), ci=boot(v), n_positive=int((v > 0).sum()))
    if floor is not None:
        f = np.asarray(floor, float); d = v - f; o.update(floor=float(f.mean()), floor_ci=boot(f), minus_floor=float(d.mean()), minus_floor_ci=boot(d), n_above_floor=int((d > 0).sum()))
    return o


def frac(y, c):
    y, c = np.asarray(y, float), np.asarray(c, float); ix = rng.integers(0, len(y), size=(B, len(y)))
    return dict(value=float(y.mean() / c.mean()), ci=np.quantile(y[ix].mean(1) / c[ix].mean(1), [.025, .975]).tolist(), mean_of_ratios=float((y / c).mean()))


def L90(G, widths):
    G = np.asarray(G, float); m = G.mean(0); L = int(widths[int((m >= .9 * m.max()).argmax())]); ix = rng.integers(0, len(G), size=(B, len(G))); M = G[ix].mean(1)
    bi = np.array(widths)[(M >= .9 * M.max(1, keepdims=True)).argmax(1)]; return dict(value=L, ci=[int(np.quantile(bi, .025)), int(np.quantile(bi, .975))], plateau=float(m.max()))


def image_models():
    out = {}
    rows = []
    for f in sorted(glob.glob(str(A / 'u1_native_v2/sub-[0-9][0-9].json'))):
        d = rd(f); g = lambda k, key, c='eeg': np.mean([r[c][key] for r in d['rows'] if r['model'] == 'brainflora' and r['k'] == k])
        rows.append(dict(subject=d['subject'], k1=g(1, 'InfoNCE'), k1_shuffled=g(1, 'shuffled_InfoNCE_mean'), k1_noise=g(1, 'InfoNCE', 'noise'), k40=g(40, 'InfoNCE'),
                         k80=g(80, 'InfoNCE'), k80_shuffled=g(80, 'shuffled_InfoNCE_mean'), k80_noise=g(80, 'InfoNCE', 'noise'), ceiling_k40=d['references']['brainflora']['InfoNCE']))
    r = lambda k: [x[k] for x in rows]
    out['brainflora_native'] = dict(n=len(rows), participants=r('subject'), per_participant=rows, k1=stat(r('k1'), r('k1_shuffled')), k1_noise=stat(r('k1_noise')),
                                    k80=stat(r('k80'), r('k80_shuffled')), k80_noise=stat(r('k80_noise')), k40=stat(r('k40')), ceiling_k40=stat(r('ceiling_k40')),
                                    fraction_k40=frac(r('k40'), r('ceiling_k40')), fraction_k80_of_k40_ceiling=frac(r('k80'), r('ceiling_k40')))
    rows = []
    for f in sorted(glob.glob(str(A / 'q7/fm_probe_reve_sub-[0-9][0-9].json'))):
        d = rd(f); g = lambda c, k: np.mean([x['soft_bits'] for x in d['rows'] if x['condition'] == c and x['k'] == k])
        rows.append(dict(subject=d['subject'], n_train=d['n_train'], k1=g('eeg', 1), k80=g('eeg', 80), k80_shuffled=g('shuffled_pair', 80), k80_noise=g('noise', 80), k80_trial_shuffled=g('trial_shuffled', 80)))
    out['reve_q7'] = dict(n=len(rows), participants=r('subject'), skipped='sub-04: no official training file (preprocessed_eeg_training.npy absent)', per_participant=rows,
                          k1=stat(r('k1')), k1_floor='not available (Q7 ran controls at k=80 only)', k80=stat(r('k80'), r('k80_shuffled')), k80_noise=stat(r('k80_noise')),
                          k80_trial_shuffled=stat(r('k80_trial_shuffled')), ceiling='not available (no split-half through REVE in Q7)')
    return out


def main():
    res = dict(note=__doc__, unit='bits per trial/query: image = per test query (mean of k repetitions); sentence = per reading; PE = per feedback trial')
    # image (NICE-GA ours, k40)
    s3 = rd(ROOT / 'results/loop/s3_fraction/s3_fraction_k40.json')['rows']; y = [x['yield_k40']['I_NCE'] for x in s3]; fl = [x['floor']['I_NCE'] for x in s3]; c = [x['ceiling_k40']['I_NCE'] for x in s3]
    q8 = rd(A / 'q8/summary.json'); W = [int(w) for w in q8['widths']]; subs = [p['subject'] for p in q8['widths'][str(W[0])]['participants']]
    G = [[next(p['pca'] for p in q8['widths'][str(w)]['participants'] if p['subject'] == s) for w in W] + [rd(A / f'q8/q8_{s}.json')['full_width']['soft_bits']] for s in subs]
    im = dict(decoder='NICE-GA (ours), k=40', yield_per_query=stat(y, fl), ceiling=stat(c), fraction=frac(y, c), L90=dict(L90(G, W + [15750]), widths=W + ['full=15750'], source='Q8 k=80 PCA', n=len(G)))
    im['open_weight_models'] = image_models(); (S / 'image_models').mkdir(parents=True, exist_ok=True)
    (S / 'image_models/summary.json').write_text(json.dumps(im['open_weight_models'], indent=1)); res['image'] = im
    # sentence
    for task in ('SR', 'NR2'):
        t = rd(S / f'text_ceiling/{task}.json'); x6 = rd(ROOT / f'results/loop/x6_text_bottleneck/x6_{task}.json'); pcs = [cv for cv in x6['curve'] if cv['basis'] == 'pca']
        Ws = [int(cv['L']) for cv in pcs]; ids = list(pcs[0]['per_participant'].keys()); Gt = [[cv['per_participant'][i]['eeg']['nce'] for cv in pcs] for i in ids]
        yy = t['yield']; pp = yy['per_participant']
        res[f'sentence_{task}'] = dict(candidates=t['N'], yield_per_reading=stat([v['eeg'] for v in pp.values()], [v['noise'] for v in pp.values()]),
                                       shuffled_floor=stat([v['shuffled_pair'] for v in pp.values()]),
                                       ceiling_half=dict(mean=t['ceiling']['mhalf']['eeg'], ci_splits=t['ceiling']['mhalf']['eeg_ci_splits'], shuffled=t['ceiling']['mhalf']['shuffled_pair'], length_only=t['ceiling']['mhalf']['length_fixated_split'], n_splits=t['ceiling']['mhalf']['n_splits']),
                                       ceiling_single_reader=dict(mean=t['ceiling']['m1']['eeg'], ci_splits=t['ceiling']['m1']['eeg_ci_splits'], shuffled=t['ceiling']['m1']['shuffled_pair'], length_only=t['ceiling']['m1']['length_fixated_split']),
                                       length_words_control=t['controls']['length_words'], fraction_of_half_ceiling=t['fraction_mhalf'], fraction_of_single_reader_ceiling=t['fraction_m1'],
                                       fraction_interpretable=bool(t['ceiling']['mhalf']['eeg_ci_splits'][0] > t['ceiling']['mhalf']['shuffled_pair'] and yy['minus_noise_ci'][0] > 0),
                                       fraction_note='a fraction needs a yield above its floor and a ceiling above its floor; flagged false otherwise',
                                       checks=t['checks'],
                                       L90=dict(L90(Gt, Ws), defined=False, reason='X6 curve at floor at every width (yield not above noise twin); L90 of a null curve is not interpretable', source='X6 PCA'))
    # prediction error
    pc = rd(S / 'pe_ceiling/summary.json'); pb = rd(S / 'pe_bottleneck/summary.json'); rep = rd(A / 'log_yield_replication.json')['datasets']
    pe = dict(decoder='EEGNet LOPO read-out, Gaussian density gain vs intercept-only')
    for ds in ('ds003458', 'ds004595'):
        rr = rep[ds]; pe[f'yield_{ds}'] = stat([x['gain'] for x in rr], [np.mean(x['shifted']) for x in rr])
    pe['ceiling_ds003458'] = dict(n=pc['n_ceiling'], mean=pc['ceiling']['mean'], ci=pc['ceiling']['ci'], source=pc['ceiling_source'])
    pe['fraction_ds003458'] = dict(n=pc['n_ceiling'], value=pc['fraction_primary']['value'], ci=pc['fraction_primary']['ci'], definition='(gain - floor)/(ceiling - floor), ceiling at the mean reliability (METHOD CHANGE 10)',
                                  sensitivity_ceiling10=pc['fraction_sensitivity_ceiling10'], sensitivity_per_participant_ceiling=pc['fraction_sensitivity_per_participant_ceiling'])
    pe['raw_erp_bottleneck_ds003458'] = dict(n=pb['n'], full_width_gain=pb['full_width_gain'], full_width_gain_ci=pb['curve']['full']['pca_gain_ci'], shift_floor=pb['full_width_shift_floor'],
                                             full_minus_shift_ci=pb['full_width_gain_minus_shift_ci'], n_above_shift=pb['full_width_n_above_shift'], plateau=pb['plateau_gain'], plateau_L=pb['plateau_L'],
                                             L90=pb['L90_num'], L90_ci=pb['L90_ci'], random_projection_L90width_gain=pb['curve'][str(pb['L90'])]['random_gain'])
    res['prediction_error'] = pe
    (S / 'summary.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=None)[:6000])


if __name__ == '__main__':
    main()
