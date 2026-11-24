"""Re-register EVERY cell of results/stats/paper_stats.json (run after each regeneration; rows are keyed by index).
Names St<Mod3><TwoLetterRowTag><Field> (coordinator's scheme), keys <modality>.<i>.<field>."""
import json, subprocess
E = "/path/to/eegaudit/experiments/stats"; F = "/path/to/eegaudit/results/stats/paper_stats.json"
PY = "python"; LED = "/path/to/tools"
S = json.load(open(F)); MOD = {"image": "Ima", "text": "Tex", "prediction_error": "Pre"}
FIELDS = [("Diff", "diff", ".3f"), ("Dz", "dz", ".2f"), ("PHolm", "p_holm", ".3f"), ("PHolmMant", "p_holm_mant", ".1f"), ("PTost", "p_tost", ".3f"),
          ("PTostMant", "p_tost_mant", ".1f"), ("MeanA", "mean_a", ".3f"), ("MeanB", "mean_b", ".3f")]
n = 0
for mod, rows in S.items():
    for i, r in enumerate(rows):
        tag = chr(65 + i // 26) + chr(65 + i % 26)
        todo = [(f, k, fmt) for f, k, fmt in FIELDS if k in r and r[k] == r[k]]
        if "diff_ci" in r and r["diff_ci"]: todo += [("Cilo", "diff_ci.0", ".3f"), ("Cihi", "diff_ci.1", ".3f")]
        for f, k, fmt in todo:
            subprocess.run([PY, LED, "number", E, f"St{MOD[mod]}{tag}{f}", "--file", F, "--key", f"{mod}.{i}.{k}", "--fmt", fmt], check=True, capture_output=True); n += 1
print("registered", n, "cells")
