#!/bin/bash
# Tiles four existing figure PDFs into fig_controls_composite.pdf. Run through the compute node:
#   srun -n1 -t 5 bash paper/v2/figs/src/make_fig_controls_composite.sh
set -e
F=/path/to/eegaudit/paper/v2/figs; B=$F/preview/composite
mkdir -p $B; cp $F/src/fig_controls_composite.tex $B/
cp $F/fig_glim.pdf $F/fig_mi_derivations.pdf $F/fig_planted_calibration.pdf $F/fig_category_erasure.pdf $B/
cd $B && conda -X compile fig_controls_composite.tex > build.log 2>&1 || { tail -20 build.log; exit 1; }
gs -q -o $F/fig_controls_composite.pdf -sDEVICE=pdfwrite -dNoOutputFonts -dCompatibilityLevel=1.5 $B/fig_controls_composite.pdf   # outline the panel letters (AAAI: no CID fonts in graphics)
T=conda
$T/pdfinfo $F/fig_controls_composite.pdf | grep "Page size"; $T/pdffonts $F/fig_controls_composite.pdf | tail -n +3 | wc -l
$T/pdftoppm -png -r 150 -singlefile $F/fig_controls_composite.pdf $F/preview/pdf_fig_controls_composite
