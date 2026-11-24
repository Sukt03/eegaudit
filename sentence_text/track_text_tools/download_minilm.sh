#!/bin/bash
M=/path/to/eegaudit/data_raw/zuco/minilm; mkdir -p $M
for f in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors; do
  curl -sS -L --retry 3 -o "$M/$f" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f" || echo "FAILED minilm $f"
done
ls -l $M; echo "$(date) minilm done"
