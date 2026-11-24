#!/usr/bin/env bash
# Download PhysioNet EEG Motor Movement/Imagery DB (eegmmidb 1.0.0, ODC-By) imagery runs only: R04, R06, R08, R10, R12, R14 (.edf + .edf.event)
# for S001-S109 into data_raw/third_party/eegmmidb/SXXX/.
# v2 (EXECUTOR-3, 21 Nov): v1 called curl without -L, PhysioNet answers 302, and the empty redirect body was saved as success (all files 0 B).
# Now: curl -L; a file counts only if size > 0 and, for .edf, its first 8 bytes are the EDF header "0" + 7 spaces; else FAILED (removed).
# Existing VALID files are skipped; empty / invalid files are re-downloaded (overwritten). Final line: valid file count and total size.
set -uo pipefail
OUT=/path/to/eegaudit/data_raw/third_party/eegmmidb; cd "$OUT"; B=https://physionet.org/files/eegmmidb/1.0.0
valid() { local f=$1; [ -s "$f" ] || return 1; case "$f" in *.edf) [ "$(head -c 8 "$f")" = "0       " ] || return 1;; esac; return 0; }
for i in $(seq 1 109); do S=$(printf "S%03d" $i); mkdir -p $S
  for r in 04 06 08 10 12 14; do for ext in edf edf.event; do f=$S/${S}R$r.$ext; valid $f && continue
    curl -s -f -L --retry 3 --retry-delay 5 --connect-timeout 60 --max-time 600 -o $f.part $B/$f && mv $f.part $f || rm -f $f.part
    valid $f || { echo "FAILED $f"; rm -f $f; }; done; done; done
n=0; for f in S*/S*R*.edf S*/S*R*.edf.event; do valid $f && n=$((n+1)); done
echo "EEGMMIDB_DONE $(date +%H:%M) valid files $n (expected 1308) size $(du -sh . | cut -f1)"
