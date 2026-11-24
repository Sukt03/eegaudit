# EEG Gauge

Code for *EEG Gauge: A Framework for Diagnosing Signal Yield in EEG Decoders* (anonymous submission).

EEG Gauge scores a decoder with and without the EEG and reports the difference, the **signal yield**, in bits. Each yield is then checked against blank, replicate, nuisance, planted-signal and exposure controls, and ten diagnostics (D0 to D9) are graded separately for each target. They are never summed.

## Layout

| Folder | Contents |
|---|---|
| `core/` | Shared EEG extraction, decoders, and the bottleneck and yield read-outs |
| `cogmodels/` | Learning models (Rescorla-Wagner and others) used to build the prediction-error target |
| `image/` | Image identity on THINGS-EEG2: UBP, BrainFLORA and NICE-GA scoring, blanks, replicate yield, category erasure |
| `sentence_text/` | Sentence reading on ZuCo 1.0 and 2.0: re-implemented text decoders, noise-trained twins, task controls |
| `baselines/` | Re-implemented and released baselines and their noise controls |
| `prediction_error/` | Reward prediction error on ds003458, ds004595 and ds007647 |
| `motor_imagery/` | Motor imagery on EEGMMIDB: raw, EEGNet, tangent-space and foundation-model read-outs, proxy regression |
| `all_targets/` | Analyses run on every target (yield replication, stress tests) |
| `extensions/` | Planted-signal calibration, untrained-encoder arms, phase surrogates, exposure checks |
| `common_and_audits/` | Estimator and control audits, plotting style |
| `stats_figures/` | Participant bootstraps and summary tables |
| `figures/` | Scripts that draw the paper's figures and tables from saved result files |

## Setup

```
conda create -n eeggauge python=3.11
conda activate eeggauge
pip install -r requirements.txt
```

The scripts expect the project root at `/path/to/eegaudit`. Replace that prefix with your own path, or create a link to it. Results go under `results/` and figures under `paper/`.

## Data

All data are public. Download them from their original sources:

- THINGS-EEG2 (Gifford et al. 2022)
- ZuCo 1.0 and ZuCo 2.0 (Hollenstein et al. 2018, 2020)
- OpenNeuro ds003458, ds004595 and ds007647
- EEGMMIDB on PhysioNet (Schalk et al. 2004)

Released checkpoints for UBP, BrainFLORA, GLIM and the EEG foundation models (CBraMod, REVE, LaBraM, EEGPT) come from their authors' repositories. Some image scripts import the ATM code (`ATMS_retrieval`, `model`) from its public repository.

## Running

Each script takes `--help`. Most write a JSON summary with the code hash, participant bootstrap intervals (10,000 resamples) and the seed. GPU scripts use PyTorch and were run on one A100 40 GB.

## License

MIT
