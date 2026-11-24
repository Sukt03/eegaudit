# Raw released-UBP bits minus noise-input twin (N(0,1) test input, same model/read-out/soft12 scorer), from the saved soft12 cards. Participant bootstrap on GPU.
import json, hashlib, sys, torch
R='/path/to/eegaudit/results/baselines/ubp'; O='/path/to/eegaudit/results/baselines/ubp_noise_input'
dev='cuda'; per={}
for s in range(1,11):
    r=json.load(open(f'{R}/ubp_card_soft12_sub{s}.json'))['subjects'][str(s)]['released']
    nz=r['noise_input_floor_bits']; assert abs(nz)<=0.02,(s,nz)
    per[s]={'noise_input_bits':nz,'noise_input_top1':r['noise_input_floor_top1'],'k80_bits':r['k80_bits'],'k40_bits':r['k40_bits'],
            'k80_minus_noise':r['k80_bits']-nz,'k40_minus_noise':r['k40_bits']-nz}
    json.dump({'participant':s,**per[s],'source':f'ubp_card_soft12_sub{s}.json released row'},open(f'{O}/sub{s}.json','x'))
g=torch.Generator(device=dev); g.manual_seed(0); idx=torch.randint(0,10,(10000,10),device=dev,generator=g); out={}
for k in ('k80_minus_noise','k40_minus_noise','noise_input_bits'):
    v=torch.tensor([per[s][k] for s in range(1,11)],device=dev,dtype=torch.float64); b=v[idx].mean(1)
    out[k]={'mean':v.mean().item(),'ci95':[b.quantile(.025).item(),b.quantile(.975).item()],'n':10}
sha=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump({'status':'','script_sha256':sha,'bootstrap':'participant, 10000 resamples, seed 0, GPU torch',
 'note':'No new model run: released-model noise-input floor (N(0,1) test input, shape of the 80-rep average, seed 7) was already saved in the soft12 cards; one noise pass serves as twin for both k=80 and k=40 (no separate k=40 noise pass exists).',
 'summary':out,'per_participant':per},open(f'{O}/summary.json','x'),indent=1)
print(json.dumps(out,indent=1),sha)
