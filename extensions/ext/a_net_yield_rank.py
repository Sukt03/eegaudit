"""Alternatives option A, cheapest first experiment, image target only.
Net yield = trained k80 bits - max(shuffled floor, noise floor, untrained-encoder yield), per participant, three decoders on THINGS-EEG2.
Participant bootstrap (10000, seed 0): rank per draw on mean net yield, rank intervals, P(i beats j).
PE and MI have one trained decoder each with untrained cells, so a decoder ranking is undefined there (not run).
"""
import json, numpy as np
R='/path/to/eegaudit/results/'
rng=np.random.default_rng(0); subs=range(1,11)
net={'ubp':[],'nice_ga':[],'brainflora':[]}; raw={k:[] for k in net}; fl={k:[] for k in net}
for s in subs:
    S=json.load(open(R+f'baselines/ubp/ubp_card_soft12_sub{s}.json'))['subjects'][str(s)]; u=S['released']
    t=u['k80_bits']; f=max(u['k80_floor_bits'],u['noise_input_floor_bits'],S['random_init_ridge']['k80_bits'])
    net['ubp'].append(t-f);raw['ubp'].append(t);fl['ubp'].append(f)
    d=json.load(open(R+f'gpu_fill_20261123/image_untrained/sub-{s:02d}.json'))
    for m in ('nice_ga','brainflora'):
        t=d['trained_saved'][m]['k80']; x=d['models'][m]
        f=max(x['k80'],x['noise_k80'],x['shuffled_k80'])
        net[m].append(t-f);raw[m].append(t);fl[m].append(f)
names=list(net); A=np.array([net[k] for k in names])  # 3 x 10
import torch
G=torch.Generator(device='cuda'); G.manual_seed(0)   # GPU bootstrap
idx=torch.randint(0,10,(10000,10),device='cuda',generator=G); M=torch.tensor(A,device='cuda',dtype=torch.float64)[:,idx].mean(2)  # 3 x 10000
rank=((-M).argsort(0).argsort(0)+1).cpu().numpy()
out={'note':' Image target only (THINGS-EEG2, n=10, k80, soft12 bits).',
 'decoders':names,'mean_net_yield':{k:float(np.mean(net[k])) for k in names},'mean_raw_yield':{k:float(np.mean(raw[k])) for k in names},
 'mean_max_floor_subtracted':{k:float(np.mean(fl[k])) for k in names},
 'rank_ci95':{k:[int(np.percentile(rank[i],2.5)),int(np.percentile(rank[i],97.5))] for i,k in enumerate(names)},
 'rank_prob':{k:[float((rank[i]==r).astype(float).mean()) for r in (1,2,3)] for i,k in enumerate(names)},
 'p_beats':{f'{a}>{b}':float((M[i]>M[j]).double().mean()) for i,a in enumerate(names) for j,b in enumerate(names) if i!=j},
 'ranking_by_raw_equals_net':sorted(names,key=lambda k:-np.mean(raw[k]))==sorted(names,key=lambda k:-np.mean(net[k])),
 'per_participant_net':{k:net[k] for k in names}}
top=sorted(names,key=lambda k:-np.mean(net[k]))
out['top1_vs_top2_nonoverlapping_rank_intervals']=bool(out['rank_ci95'][top[0]][1]<out['rank_ci95'][top[1]][0] or out['rank_ci95'][top[0]][1]<=1 and out['rank_ci95'][top[1]][0]>=2)
json.dump(out,open(R+'ext/a_net_yield_rank/summary.json','w'),indent=1)
print({k:v for k,v in out.items() if k!='per_participant_net'})
