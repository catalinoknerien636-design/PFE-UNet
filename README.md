# PFE-UNet

[![DOI](https://zenodo.org/badge/1305596882.svg)](https://doi.org/10.5281/zenodo.21679107)

Companion code and frozen source data for **PFE-UNet: DoLP-Conditioned Feature Modulation for RGB-DoLP Camouflaged Object Segmentation**.

## Overview

The repository implements the Polarization Feature Enhancement U-Net (PFE-UNet) and evaluates bounded DoLP-conditioned feature modulation at configurable encoder levels for RGB camouflaged object segmentation. It contains the five matched-seed configurations for direct RGB+DoLP concatenation, PFE-DoLP, PFE-S0, PFE-RGB-gray, and learned 1 x 1 fusion. The code preserves validation-IoU checkpoint selection and image-averaged MAE.

## Repository Structure

`configs/` contains 25 frozen core configurations; `splits/` contains the fixed PCOD manifests; `models/` and `datasets/` contain the network and loaders; `metrics/` contains COD metric code; `analysis/` contains smoke and figure scripts; `results/` and the source-data folders contain frozen CSVs; `external_review_extension/` contains the gate-placement and synthetic DoLP-sensitivity extension; `checkpoints/checkpoint_manifest.csv` records checkpoint identifiers and checksums.

## Requirements and Setup

Use Python 3.11. On Windows, run `scripts/setup_environment.ps1`. CPU and CUDA execution are supported by the cleaned scripts. DirectML can be installed separately when required by the local hardware.

## Dataset Access

Download PCOD_1200 from Mendeley Data (version 1, DOI: 10.17632/kyrc4ccym7.1) and retain the official directory names. MCOD is used only for the cross-modal stress test and must be obtained from its official release under its license. Third-party datasets are not redistributed.

Example:

```text
data/PCOD_1200/full-dataset/
  train/train-rgb, train/train-dop, train/train-gt, train/train-Stokes/
  test/test-rgb, test/test-dop, test/test-gt, test/test-Stokes/
```

## Fixed Manifests

All five-seed PCOD runs reuse `splits/train_split.csv` (3104 records), `splits/val_split.csv` (776 records), and the official 230-record test subset listed in `splits/test_split.csv`. Checkpoints are selected exclusively by validation IoU. The test subset is used only after checkpoint selection.

## Training and Evaluation

```powershell
python train.py --config configs/pfe_dolp_seed42.yaml --device cpu
python evaluate.py --config configs/pfe_dolp_seed42.yaml --checkpoint checkpoints/pfe_dolp_seed42/best.pt --dataset-root data/PCOD_1200/full-dataset
```

Run the matched seeds with `scripts/train_core_models.ps1`, then evaluate them with `scripts/evaluate_core_models.ps1`.

## Reproducing Tables and Figures

```powershell
python reproduce_tables.py
python analysis/export_smoke_figure.py
```

`scripts/reproduce_supplementary.ps1` copies the frozen statistical tables into a release folder. Publication qualitative figures require the third-party PCOD files and the archived checkpoints; their sample IDs and source descriptions are in `figure_source_data/`.

## Extended Experiments

`external_review_extension/` preserves the completed gate-placement ablation, the matched M6 subset calculation, and the synthetic DoLP translation, rotation, blur, noise, and quantization tests. Its source tables and plotting scripts reproduce the corresponding manuscript and supplementary summaries without changing the frozen scientific values.

The extension uses repository-relative paths. When rerunning analyses against separately stored project artifacts, set `PFE_SOURCE_PROJECT_ROOT` and `PFE_EVIDENCE_ROOT` to the applicable local directories. The public source release excludes trained model files, third-party datasets, prediction maps, and machine-specific absolute paths.

The included run manifests retain short run identifiers, configurations, selected epochs, and checksums. Experiments that require trained weights must be reproduced by training from the supplied configurations or by obtaining the separately archived weights when they become available.

## Checkpoint Selection

The latest checkpoint is saved each epoch. `best.pt` is updated only when validation IoU increases. No test metric is used for checkpoint, hyperparameter, augmentation, or model selection.

## Known Limitations

The principal evidence comes from PCOD_1200. Image-level paired analyses are conditional on five trained realizations and 230 released test images. The DoLP-S0 contrast is suggestive rather than definitive. MCOD is a negative cross-modal stress test, not polarization validation. The method is not claimed to be universal, state of the art, or lightweight.

## License

The source code in this repository is licensed under the MIT License; see
`LICENSE`. This license does not apply to third-party datasets. PCOD_1200,
MCOD, and all software dependencies remain subject to their respective
licenses and terms.

## Citation

Citation metadata for this software release are provided in `CITATION.cff`.
For the archived `v1.0.0` release, cite the version-specific DOI:
https://doi.org/10.5281/zenodo.21679108.
