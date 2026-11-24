#!/bin/bash
# Downloads the Wang & Ji (AAAI 2022) EEG-To-Text preprocessed ZuCo pickles from the HF mirror NiallMcGuire12/ZuCo
set -u
D=/path/to/eegaudit/data_raw/zuco/eeg2text_pickles
B=https://huggingface.co/datasets/NiallMcGuire12/ZuCo/resolve/main/ZuCo
for f in task1-SR/pickle/task1-SR-dataset.pickle task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle task2-NR/pickle/task2-NR-dataset.pickle; do
  n=$(basename $f)
  echo "$(date) start $n"
  curl -sS -L -C - --retry 5 -o "$D/$n" "$B/$f" || echo "FAILED $n"
  ls -l "$D/$n"
done
# small sentence encoder for the target space (all-MiniLM-L6-v2, ~90 MB)
M=/path/to/eegaudit/data_raw/zuco/minilm
mkdir -p $M
for f in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors; do
  curl -sS -L --retry 3 -o "$M/$f" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f" || echo "FAILED minilm $f"
done
ls -l $M
echo "$(date) done"
