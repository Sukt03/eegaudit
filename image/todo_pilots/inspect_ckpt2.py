import sys, torch, collections
sd=torch.load(sys.argv[1], map_location="cpu", weights_only=True)
c=collections.Counter()
for k,v in sd.items(): c[".".join(k.split(".")[:3])]+=v.numel()
for p,n in sorted(c.items()): print(f"{n/1e6:8.2f}M {p}")
print("--- keys outside encoder.encoders.*.transformer_encoder and bart.model")
for k,v in sd.items():
    if ("transformer_encoder" not in k) and not k.startswith("bart.model"): print(k, tuple(v.shape))
print("--- bart.model sample"); 
for k in [k for k in sd if k.startswith("bart.model")][:8]: print(k, tuple(sd[k].shape))
print("n enc layers:", len({k.split(".")[4] for k in sd if "transformer_encoder.layers" in k and k.startswith("encoder.encoders.C")}))
print("bart enc layers:", len({k.split(".")[4] for k in sd if k.startswith("bart.model.encoder.layers")}), "dec layers:", len({k.split(".")[4] for k in sd if k.startswith("bart.model.decoder.layers")}))
