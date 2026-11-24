import json,glob,random,statistics as st,sys
R='/path/to/eegaudit/results/'
P=json.load(open(R+'ext/ueyc/profile.json'))
prof=P['profile']
def row(t,d): return [r for r in prof if r['target'].startswith(t) and r['decoder'].startswith(d)][0]
L=lambda f: json.load(open(R+f))
def mi(arm): return [json.load(open(f))[arm]['eeg']['bits'] for f in sorted(glob.glob(R+'ext/mi_card_v3/primary_lpgo/sub-*.json'))]
def ubp(k): 
    return [json.load(open(R+'baselines/ubp/ubp_summary.json'))['released'][k]['per_participant']][0]
ck={}
ck['MI raw_minus_eog_proxy mean']=(row('motor','raw')['specific_yield']['raw_minus_eog_proxy']['value'], st.mean(a-b for a,b in zip(mi('raw'),mi('eog_proxy'))))
ck['MI raw_eogreg kept ratio']=(row('motor','raw')['specific_yield']['raw_eogreg_kept_share']['value'], st.mean(mi('raw_eogreg'))/st.mean(mi('raw')))
ck['UBP own fraction k40/ceil']=(row('image','UBP')['own_reference_fraction']['recomputed_gpu']['ratio_of_means'], st.mean(ubp('k40_bits'))/st.mean(ubp('ceiling_k40_bits')))
u=L('baselines/ubp/ubp_summary.json')
ck['UBP baseline share k80']=(row('image','UBP')['untrained_encoder_baseline']['share_recomputed_gpu']['ratio_of_means'], st.mean(u['random_init_ridge']['k80_bits']['per_participant'])/st.mean(u['released']['k80_bits']['per_participant']))
m=L('ext/m7/summary_pe.json')['arms']
ck['PE REVE baseline share']=(row('prediction','REVE')['untrained_encoder_baseline']['share_of_raw'], m['reve_twin']['gain']['mean']/m['reve']['gain']['mean'])
t=L('baselines/text/text_yield_card.json')['models']['glim_released']['yield']['free_running']['pooled']
ck['GLIM noise floor']=(row('text','GLIM')['noise_input_floor']['value'], t['arms']['noise']['yield_bits'] if 'noise' in t['arms'] else None)
e=L('ext/eu_bac/full.json')['A_ubp_budget']
ck['UBP net k80']=(P['scaling']['Y_k']['UBP']['80']['net_of_untrained_encoder']['value'], st.mean(a-b for a,b in zip(u['released']['k80_bits']['per_participant'],u['random_init_ridge']['k80_bits']['per_participant'])))
ck['k80_minus_noise']=(P['scaling']['Y_k']['UBP_raw_minus_noise_input']['80']['value'], st.mean(v['k80_minus_noise'] if 'k80_minus_noise' in v else float('nan') for v in L('baselines/ubp_noise_input/summary.json')['per_participant'].values()))
ck['MI FM dividend']=(None,None)
random.seed(int(sys.argv[1]) if len(sys.argv)>1 else 3)
for k in random.sample([k for k in ck if ck[k][1] is not None and ck[k][0] is not None],5):
    a,b=ck[k]; print(k, a, b, 'OK' if abs(a-b)<1e-6 else 'DIFF')
