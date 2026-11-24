#!/bin/bash
# Registers every yield-protocol number quoted in fig_yield / fig_usable captions and the text (read from results/stats/yield_stats.json).
PY=python; L="$PY /path/to/tools number /path/to/eegaudit/experiments/stats"
F=/path/to/eegaudit/results/stats/yield_stats.json
reg() { $L "$1" --file $F --key "$2" --fmt "$3" ${4:+--scale $4}; }
tri() { reg "$1" "$2.mean" "$3" $4; reg "$1Lo" "$2.ci95.0" "$3" $4; reg "$1Hi" "$2.ci95.1" "$3" $4; }
tri YieldImgFull image.input_nicega_InfoNCE.full_yield .2f
reg YieldImgFullN image.input_nicega_InfoNCE.n .0f
reg YieldImgLninety image.input_nicega_InfoNCE.L90.median .0f
reg YieldImgLninetyLo image.input_nicega_InfoNCE.L90.median_ci95.0 .0f
reg YieldImgLninetyHi image.input_nicega_InfoNCE.L90.median_ci95.1 .0f
reg YieldImgLninetyMin image.input_nicega_InfoNCE.L90.min .0f
reg YieldImgLninetyMax image.input_nicega_InfoNCE.L90.max .0f
reg YieldImgFullDim image.input_nicega_InfoNCE.full_dim .0f
reg YieldImgReprLninety image.representation_nicega_Wolpaw.L90_median .0f
reg YieldImgLratio image.L90_input_over_representation.median .0f
reg YieldImgLratioLo image.L90_input_over_representation.median_ci95.0 .0f
reg YieldImgLratioHi image.L90_input_over_representation.median_ci95.1 .0f
reg YieldImgAtmLninety image.input_atms_InfoNCE.L90.median .0f
tri UsableImgNCE image.usable_fraction.nicega_k40_I_NCE.fraction .1f 100
tri UsableImgWolpaw image.usable_fraction.nicega_k40_I_F.fraction .1f 100
reg CeilImgNCE image.usable_fraction.nicega_k40_I_NCE.ceiling.mean .2f
reg CeilImgWolpaw image.usable_fraction.nicega_k40_I_F.ceiling.mean .2f
tri YieldPeEnsFull pe.input_ens_rawr.full_yield .3f
tri YieldPeRawFull pe.input_raw_rawr.full_yield .3f
reg YieldPeEnsLninety pe.input_ens_rawr.L90.median .0f
reg YieldPeEnsLninetyLo pe.input_ens_rawr.L90.median_ci95.0 .0f
reg YieldPeEnsLninetyHi pe.input_ens_rawr.L90.median_ci95.1 .0f
reg YieldPeRawLninety pe.input_raw_rawr.L90.median .0f
reg YieldPeN pe.input_ens_rawr.n .0f
reg UsablePeEEGNet pe.usable_fraction.eegnet.fraction.mean .1f 100
reg UsablePeEEGNetLo pe.usable_fraction.eegnet.fraction.ci95.0 .1f 100
reg UsablePeEEGNetHi pe.usable_fraction.eegnet.fraction.ci95.1 .1f 100
reg UsablePeEns pe.usable_fraction.ens.fraction.mean .1f 100
reg UsablePeEnsLo pe.usable_fraction.ens.fraction.ci95.0 .1f 100
reg UsablePeEnsHi pe.usable_fraction.ens.fraction.ci95.1 .1f 100
reg CeilPeEEGNet pe.usable_fraction.eegnet.ceiling .2f
reg CeilPeEns pe.usable_fraction.ens.ceiling .2f
reg CeilPeEEGNetN pe.usable_fraction.eegnet.n_ceiling .0f
tri YieldTxtEeg text.full_width_ridge.eeg .3f
tri YieldTxtNoise text.full_width_ridge.noise .3f
tri YieldTxtEegTwoWay text.full_width_ridge.eeg_two_way .3f
reg YieldTxtN text.full_width_ridge.n_participants .0f
# protocol-schema sweeps (results/yield/*): PE raw-ERP ridge, Gaussianised null-corrected; ZuCo ridge InfoNCE
G=pe.protocol_ds003458_raw_erp_ridge_bits_g
tri YieldPeGaussFull $G.full_yield .3f
reg YieldPeGaussLninety $G.L90.median .0f
reg YieldPeGaussLninetyLo $G.L90.median_ci95.0 .0f
reg YieldPeGaussLninetyHi $G.L90.median_ci95.1 .0f
reg YieldPeGaussLsixtyfour $G.curves.eeg.64.mean .3f
reg YieldPeGaussFloor $G.floor_full_max .3f
reg YieldPeGaussFullDim $G.full_dim .0f
reg YieldPeEEGNetFull pe.usable_fraction.eegnet.yield_mean .3f
reg YieldPeEEGNetFullLo pe.usable_fraction.eegnet.yield_ci95.0 .3f
reg YieldPeEEGNetFullHi pe.usable_fraction.eegnet.yield_ci95.1 .3f
tri YieldTxtSRNCEFull text.protocol_zuco_SR_ridge_mean_wordsall_bits_NCE.full_yield .3f
tri YieldTxtNRtwoNCEFull text.protocol_zuco_NR2_ridge_mean_wordsall_bits_NCE.full_yield .3f
reg YieldTxtNRtwoN text.protocol_zuco_NR2_ridge_mean_wordsall_bits_NCE.n .0f
$PY /path/to/tools export /path/to/eegaudit/experiments/stats
