#!/bin/bash
# Official ZuCo 1.0 Matlab files (same files as OSF q3zws; HF mirror NiallMcGuire12/ZuCo is faster) for reading-time extraction
D=/path/to/eegaudit/data_raw/zuco/osf_mat
B=https://huggingface.co/datasets/NiallMcGuire12/ZuCo/resolve/main/ZuCo
echo "$(date) start"; curl -sS -L -C - --retry 5 -o $D/resultsZDN_SR.mat $B/task1-SR/Matlab_files/resultsZDN_SR.mat ; ls -l $D/resultsZDN_SR.mat
curl -sS -L -C - --retry 5 -o $D/resultsZJS_NR.mat $B/task2-NR/Matlab_files/resultsZJS_NR.mat ; ls -l $D/resultsZJS_NR.mat; echo "$(date) done"
