#!/bin/bash
# Converts the text of every figure PDF in paper/v2/figs to outlines, in place (AAAI: no CID / Identity-H fonts inside graphics).
# Keeps <name>.pdf.bak_fonts. Safe to rerun after a figure is regenerated (matplotlib writes CID TrueType again with pdf.fonttype 42).
#   srun -n1 -t 10 bash paper/v2/figs/src/outline_fonts.sh
F=/path/to/eegaudit/paper/v2/figs; T=conda
mkdir -p $F/preview/fonts; cd $F
printf "%-46s %-22s %-22s %9s %9s %s\n" file fonts_before fonts_after bytes_b bytes_a pagebox
for f in *.pdf; do
  nb=$($T/pdffonts $f 2>/dev/null | tail -n +3 | wc -l); tb=$($T/pdffonts $f 2>/dev/null | tail -n +3 | awk '{print $2,$3}' | sort -u | head -1)
  [ "$nb" = "0" ] && { printf "%-46s %-22s %-22s\n" $f "none" "(already outlined)"; continue; }
  [ -e $f.bak_fonts ] || cp -p $f $f.bak_fonts
  gs -q -o preview/fonts/$f -sDEVICE=pdfwrite -dNoOutputFonts -dCompatibilityLevel=1.5 $f || { echo "$f: gs failed"; continue; }
  na=$($T/pdffonts preview/fonts/$f 2>/dev/null | tail -n +3 | wc -l)
  pb=$($T/pdfinfo $f | grep "Page size" | awk '{print $3,$5}'); pa=$($T/pdfinfo preview/fonts/$f | grep "Page size" | awk '{print $3,$5}')
  ok=$(echo "$pb $pa" | awk '{d1=$1-$3; d2=$2-$4; if (d1<0) d1=-d1; if (d2<0) d2=-d2; if (d1<=0.01 && d2<=0.01) print "same"; else print "CHANGED"}')   # gs rounds the page box to 0.01 pt
  sb=$(stat -c %s $f); sa=$(stat -c %s preview/fonts/$f)
  if [ "$na" = "0" ] && [ "$ok" = "same" ]; then
    $T/pdftoppm -png -r 100 -singlefile $f.bak_fonts preview/fonts/${f%.pdf}_before; $T/pdftoppm -png -r 100 -singlefile preview/fonts/$f preview/fonts/${f%.pdf}_after
    cp preview/fonts/$f $f
  else ok="$ok NOT REPLACED"; fi
  printf "%-46s %-22s %-22s %9s %9s %s\n" $f "$nb x $tb" "$na" $sb $sa "$ok"
done
