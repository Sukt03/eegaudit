import sys, torch, collections
p=sys.argv[1]
sd=torch.load(p, map_location="cpu", weights_only=True)
print(type(sd), len(sd) if hasattr(sd,'__len__') else '')
if isinstance(sd,dict) and not all(torch.is_tensor(v) for v in sd.values()):
    print("top-level keys:", list(sd.keys())[:20])
    for k,v in sd.items():
        if isinstance(v,dict): print(k,"-> dict of",len(v)); 
        elif torch.is_tensor(v): print(k,"-> tensor",tuple(v.shape))
        else: print(k,"->",repr(v)[:200])
    for k in sd:
        if isinstance(sd[k],dict) and all(torch.is_tensor(x) for x in sd[k].values()): sd=sd[k]; print("using sub-dict",k); break
tot=0; prefixes=collections.Counter()
for k,v in sd.items():
    tot+=v.numel(); prefixes[".".join(k.split(".")[:2])]+=v.numel()
print("n tensors",len(sd),"params %.1fM"%(tot/1e6))
for pfx,n in prefixes.most_common(40): print(f"{n/1e6:8.2f}M  {pfx}")
print("--- first 60 keys")
for k in list(sd.keys())[:60]: print(k, tuple(sd[k].shape), sd[k].dtype)
print("--- non-pretrained keys")
for k in sd:
    if not k.startswith("pretrained"): print(k, tuple(sd[k].shape))
