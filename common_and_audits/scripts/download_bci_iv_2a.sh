#!/usr/bin/env bash
# Download BCI Competition IV 2a (BNCI Horizon 001-2014; 9 participants x sessions T/E, .mat, < 1 GB) into data_raw/third_party/bci_iv_2a.
# Official BNCI Horizon host (the URL MOABB uses). Retries each file over http and https; logs FAILED per file.
set -uo pipefail
OUT=/path/to/eegaudit/data_raw/third_party/bci_iv_2a; cd "$OUT"
for s in 1 2 3 4 5 6 7 8 9; do for t in T E; do f=A0${s}${t}.mat; [ -s $f ] && continue
  ok=0; for try in 1 2 3; do for base in http://bnci-horizon-2020.eu/database/data-sets/001-2014 https://bnci-horizon-2020.eu/database/data-sets/001-2014; do
    curl -s -f --connect-timeout 60 --max-time 1200 -o $f.part $base/$f && { mv $f.part $f; ok=1; break 2; }; done; sleep 30; done
  [ $ok = 1 ] && echo "OK $f $(stat -c %s $f)" || echo "FAILED $f"; done; done
echo "BCI2A_DONE $(date +%H:%M) $(ls *.mat 2>/dev/null | wc -l) files"
