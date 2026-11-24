"""D9 participant-half replicate (aggregation only).
Halves: odd (1,3,5,7,9) vs even (2,4,6,8,10) THINGS-EEG2 participants.
Yield = UBP released-checkpoint soft12 bits minus shuffled-pair floor (k80 and k40).
Blank = noise-input floor bits (random-noise EEG through the same checkpoint) and the noise-arm retrain bits.
Three decoders (UBP, NICE-GA, BrainFLORA) from km3_decoders_v3 'none' eraser, k=80, soft12 minus floor.
"""
import json, itertools, numpy as np
R='/path/to/eegaudit/results/'
import torch
dev='cuda'; G=torch.Generator(device=dev); G.manual_seed(0)   # GPU bootstrap (torch)
def tq(m): return [float(m.quantile(0.025)),float(m.quantile(0.975))]
subs=range(1,11)
ubp={s:json.load(open(R+f'baselines/ubp/ubp_card_soft12_sub{s}.json')) for s in subs}
rel={s:ubp[s]['subjects'][str(s)]['released'] for s in subs}
ret={s:ubp[s]['retrain'] for s in subs}
km={s:json.load(open(R+f'ext/km3_decoders_v3/sub-{s:02d}.json'))['decoders'] for s in subs}
def noise_arm(s,key):
    v=[r[key] for n,r in ret[s].items() if '/noise/' in n]; return float(np.mean(v))
M={
 'ubp_k80':{s:rel[s]['k80_bits']-rel[s]['k80_floor_bits'] for s in subs},
 'ubp_k40':{s:rel[s]['k40_bits']-rel[s]['k40_floor_bits'] for s in subs},
 'nice_ga_k80':{s:km[s]['nice_ga']['soft12'][0]-km[s]['nice_ga']['floor_soft12'][0] for s in subs},
 'brainflora_k80':{s:km[s]['brainflora']['soft12'][0]-km[s]['brainflora']['floor_soft12'][0] for s in subs},
}
B={'noise_input_k80':{s:rel[s]['noise_input_floor_bits'] for s in subs},
   'noise_arm_retrain_k80':{s:noise_arm(s,'k80_bits') for s in subs},
   'noise_arm_retrain_k40':{s:noise_arm(s,'k40_bits') for s in subs}}
halves={'odd':[1,3,5,7,9],'even':[2,4,6,8,10]}
def boot(x,n=10000):
    x=torch.tensor(np.array(x,dtype=np.float64),device=dev); i=torch.randint(0,len(x),(n,len(x)),device=dev,generator=G); m=x[i].mean(1)
    return float(x.mean()),tq(m)
out={'note':' Participant bootstrap (10000), seed 0, n=5 per half. Blank = noise-input / noise-arm bits; "excludes blank" = yield CI lower bound > blank CI upper bound AND all 5 participants above their own shuffled-pair floor.','halves':{}}
for h,ids in halves.items():
    o={}
    for k,d in M.items():
        mean,ci=boot([d[s] for s in ids])
        bk={'ubp_k40':'noise_arm_retrain_k40'}.get(k,'noise_arm_retrain_k80')
        bi=B['noise_input_k80'] if k!='ubp_k40' else B['noise_arm_retrain_k40']
        bm,bci=boot([B[bk][s] for s in ids])
        o[k]={'per_participant':{str(s):d[s] for s in ids},'mean':mean,'ci95':ci,
              'n_above_floor':int(sum(d[s]>0 for s in ids)),
              'blank_noise_arm_mean':bm,'blank_noise_arm_ci95':bci,
              'blank_noise_input_mean':boot([B['noise_input_k80'][s] for s in ids])[0],
              'ci_excludes_blank':bool(ci[0]>bci[1] and all(d[s]>0 for s in ids))}
    out['halves'][h]=o
# half difference and CI overlap
out['replicate']={}
for k,d in M.items():
    a=[d[s] for s in halves['odd']];b=[d[s] for s in halves['even']]
    ia=torch.randint(0,5,(10000,5),device=dev,generator=G);ib=torch.randint(0,5,(10000,5),device=dev,generator=G)
    diff=torch.tensor(a,device=dev,dtype=torch.float64)[ia].mean(1)-torch.tensor(b,device=dev,dtype=torch.float64)[ib].mean(1)
    ca=out['halves']['odd'][k]['ci95'];cb=out['halves']['even'][k]['ci95']
    out['replicate'][k]={'mean_diff_odd_minus_even':float(np.mean(a)-np.mean(b)),'diff_ci95':tq(diff),
        'cis_overlap':bool(ca[0]<=cb[1] and cb[0]<=ca[1]),
        'both_exclude_blank':bool(out['halves']['odd'][k]['ci_excludes_blank'] and out['halves']['even'][k]['ci_excludes_blank'])}
# decoder ranking agreement across halves (k80 mean yield)
dec=['ubp_k80','nice_ga_k80','brainflora_k80']
rk={h:sorted(dec,key=lambda k:-out['halves'][h][k]['mean']) for h in halves}
pp={h:{k:out['halves'][h][k]['mean'] for k in dec} for h in halves}
out['decoder_ranking']={'odd':rk['odd'],'even':rk['even'],'same_order':rk['odd']==rk['even'],'half_means':pp,
  'per_participant_order_counts':{}}
cnt={}
for s in subs:
    o=tuple(sorted(dec,key=lambda k:-M[k][s]));cnt[' > '.join(o)]=cnt.get(' > '.join(o),0)+1
out['decoder_ranking']['per_participant_order_counts']=cnt
json.dump(out,open(R+'ext/d9_participant_halves/summary.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='halves'},indent=1))
for h in halves:
    for k,v in out['halves'][h].items(): print(h,k,round(v['mean'],3),[round(x,3) for x in v['ci95']],v['n_above_floor'],v['ci_excludes_blank'])
