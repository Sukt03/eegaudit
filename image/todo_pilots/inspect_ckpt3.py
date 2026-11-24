import sys, torch
sd=torch.load(sys.argv[1], map_location="cpu", weights_only=True)
regs={}
for k,v in sd.items():
    if k.endswith("cnn_encoder.conv1.weight"): regs[k.split(".")[2]]=v.shape[1]
print("region -> n input channels:", regs, "sum", sum(regs.values()))
for k,v in sd.items():
    if k.startswith("eeg_to_bart") or k.startswith("encoder.global_transformer.layers.0") or k.startswith("bart.model.encoder.embed") or k.startswith("bart.model.decoder.embed") or "final_logits" in k: print(k, tuple(v.shape))
print("global layers", len({k.split(".")[3] for k in sd if k.startswith("encoder.global_transformer.layers")}))
print("bart enc layers", len({k.split(".")[4] for k in sd if k.startswith("bart.model.encoder.layers")}), "dec", len({k.split(".")[4] for k in sd if k.startswith("bart.model.decoder.layers")}))
print("any non-tensor / config keys:", [k for k in sd if not torch.is_tensor(sd[k])])
