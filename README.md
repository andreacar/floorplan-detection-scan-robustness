# floorplan-detection-scan-robustness

Code + scripts for the floorplan detection robustness paper experiments (RT-DETR focus).

This repository is organized as a **paper pipeline**: dataset loaders, evaluation/analysis scripts, and figure/table generation.

## Install

```bash
conda env create -f environment.yml
conda activate floorplan-detection-scan-robustness
python tools/fetch_dataset.py --check-remote
python tools/fetch_dataset.py
python tools/fetch_release_assets.py --check-remote
python tools/fetch_release_assets.py
python -c "import config as c; print(c.BASE_DIR)"
python diagnostic/render_4_variants.py --help
python tools/quickcheck_submission.py --strict
```

If you already have a compatible local environment, replace the environment name accordingly.

## Dataset Release

The dataset companion release for this repository is published on Zenodo:

- Dataset: `CubiCasa5k-ScanShift`
- Version-specific DOI: `10.5281/zenodo.18890484`
- Concept DOI (all versions): `10.5281/zenodo.18890483`
- Record URL: `https://doi.org/10.5281/zenodo.18890484`

Recommended dataset citation:

```text
Carrara, A., Nousias, S., & Borrmann, A. (2026). CubiCasa5k-ScanShift (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18890484
```

## Stable Checkpoints

The public GitHub release can also carry the stable RT-DETR runs as release assets. The recommended assets are the
full experiment-root archives, not isolated weight files, because the code expects Hugging Face checkpoint directories
and the experiment roots also preserve run metadata.

Recommended release assets:

- `exp1_clean_stable_run.tar.gz`
- `exp2_scanned_stable_run.tar.gz`
- `ab_on_cd_results.tar.gz`

After downloading and unpacking them:

```bash
mkdir -p stable_runs/20260115_134958
tar -xzf exp1_clean_stable_run.tar.gz -C stable_runs/20260115_134958
tar -xzf exp2_scanned_stable_run.tar.gz -C stable_runs/20260115_134958
```

This restores the checkpoint directories expected by the public evaluation commands:

- `stable_runs/20260115_134958/exp1_clean/checkpoints/best`
- `stable_runs/20260115_134958/exp2_scanned/checkpoints/best`

To restore all public quickcheck assets directly from GitHub Releases:

```bash
python tools/fetch_release_assets.py --check-remote
python tools/fetch_release_assets.py
```

Note: the *training* backend used during development lived in a separate RT-DETR checkout. That training code is not required
to reproduce results from the pinned checkpoints under `stable_runs/`, and training-dependent suites are conditionally handled by the runner.

## Capabilities at a glance

- Reproduce paper metrics from pinned checkpoints (`stable_runs/20260115_134958/...`).
- Evaluate cross-domain robustness (A/B models on C/D variants).
- Run targeted analyses: subset-softmax, error decomposition, size success, factorized degradation.
- Generate publication assets (figures + tables) under `artifacts/`.
- Train/fine-tune new runs with preset-driven configs.

## If you want results quickly

The commands below assume you also have the local pinned checkpoint assets used during paper preparation.
Those large artifacts are not tracked in the public GitHub repository, but they can be distributed as GitHub release assets.

### 1) Full internal submission validation run

```bash
bash tools/test_submission_full.sh gdino
```

This runs:
- strict environment + file + checkpoint checks,
- dry-run of the full paper pipeline,
- variant-generation validation run,
- checkpoint inference validation run on C/D,
- compact metric summary.

### 2) Reproduce pinned paper suite

```bash
conda run -n gdino python tools/run_paper.py paper_all
```

### 3) Core A/B-on-C/D robustness result only

```bash
conda run -n gdino python -m evaluation.test_AB_on_CD \
  --model-a-dir stable_runs/20260115_134958/exp1_clean/checkpoints/best \
  --model-b-dir stable_runs/20260115_134958/exp2_scanned/checkpoints/best \
  --out-dir AB_on_CD_results
```

## How this code is useful for others

- Benchmarking: use your own checkpoints and compare against pinned baselines.
- Dataset QA: verify split integrity, variant coverage, and config consistency before long runs.
- Ablations: run mechanism-specific probes without rewriting training/eval code.
- Reproducibility: one-command checks and dry-runs for pre-submission validation.

## Quick start

1) Create an environment (see `environment.yml`).

2) Get the dataset:

- Explicit fetch: `python tools/fetch_dataset.py`
- After download, the repo automatically reuses the default cache at
  `~/.cache/floorplan-detection-scan-robustness/datasets/` unless you override it.
- Remote availability check without downloading the archive: `python tools/fetch_dataset.py --check-remote`

3) Point the code at your dataset root if you want to override the default cache:

- Recommended: copy `configs/config_local_example.py` → `configs/config_local.py` and set `DEFAULT_BASE_DIR`.
- Or set `DATASET_BASE_DIR=/path/to/dataset` in your shell.
- Or set `FLOORPLAN_DATASET_CACHE_DIR=/path/to/cache` to keep the automatic download in a different cache location.
- After `python tools/fetch_dataset.py`, the script prints the resolved extracted dataset root and an `export DATASET_BASE_DIR=...` line you can reuse.

4) Run a training preset (examples):

```bash
# CAD-only / clean regime (A→A)
python training/train.py --preset clean --run-name E1_clean_seed0 --set IMAGE_SIZE=1024

# Scanned-only regime (B→B)
python training/train.py --preset scan  --run-name E2_scan_seed0  --set IMAGE_SIZE=1024
```

What happens under the hood:
- `training/train.py` → `training/main.py` reads a “preset” from `utils/paper_pipeline_configs.py`,
  applies environment-variable overrides, then runs the training backend (not needed for pinned reproduction).
- Outputs are written under `runs/<RUN_NAME>/...` (see “Outputs” below).

## Dataset assumptions (CubiCasaVec-style layout)

At runtime, `configs/config.py` constructs:
- `TRAIN_TXT`, `VAL_TXT`, `TEST_TXT` as `${BASE_DIR}/train.txt|val.txt|test.txt` by default.
- Each split `.txt` is expected to list drawing folders (one per line).
- Each drawing folder is expected to contain images (e.g. `model_baked.png`, `F1_scaled.png`) and `graph.json`.

Many experiments switch which “render variant” to use via `config.IMAGE_FILENAME`. Typical choices:
- A (CAD): `model_baked.png`
- B (scan): `F1_scaled.png`
- C/D variants (used for causal isolation): `four_final_variants/03_scan_inside_boxes.png`,
  `four_final_variants/04_svg_clean_plus_scan_outside.png`

To generate `four_final_variants/*` inside each drawing folder, use `diagnostic/render_4_variants.py`
with `--root-data /path/to/dataset/<split_dir>` (and optional `--config <yaml>` defaults).

The published dataset release used by default is:
- Dataset: `CubiCasa5k-ScanShift`
- Version-specific DOI: `10.5281/zenodo.18890484`
- Concept DOI (all versions): `10.5281/zenodo.18890483`

## Configuration model (how scripts share settings)

- `config.py` (repo root) is a shim that re-exports everything from `configs/config.py`.
- `configs/config.py` is the single source of truth for hyperparameters and paths.
  It is designed to be driven primarily by environment variables (e.g. `DATASET_BASE_DIR`, `RUN_NAME`,
  `IMAGE_SIZE`, `EPOCHS`, augmentation toggles).

Important config fields you’ll see referenced across scripts:
- Paths: `BASE_DIR`, `RUNS_DIR`, `RUN_DIR`, `CKPT_DIR`, `MET_DIR`
- Data: `TRAIN_TXT`, `VAL_TXT`, `TEST_TXT`, `IMAGE_FILENAME`
- RT-DETR: `BACKBONE` (default `PekingU/rtdetr_r50vd`)
- Resize: `IMAGE_SIZE` (defaults to `1024`) and `RESIZE_*` flags
- COCO eval: `COCO_MAX_DETS` (defaults to `[1, 10, 100]`)

## Class taxonomy (`hierarchy_config.py`)

`hierarchy_config.py` defines the **final detection label set** and the raw→final mapping.

Note: in this repo snapshot, the final label list is currently set to **6 classes**:
`WALL, COLUMN, STAIR, RAILING, DOOR, WINDOW`.

If you change the taxonomy, the mapping logic is in:
- `hierarchy_config.py` (constants + `map_raw_to_l2`)

## Training pipeline (`training/`)

Entry point:
- `training/train.py` is a thin wrapper that calls `training/main.py`.

Runner:
- `training/main.py`:
  - reads a preset from `utils/paper_pipeline_configs.py`,
  - applies env overrides (`--set KEY=VALUE`),
  - calls the training backend (optional; not required for pinned runs).

Presets:
- `utils/paper_pipeline_configs.py` defines named presets like `clean`, `scan`, `clean_scan`, `roi`, etc.
  Most presets simply set `RUN_EXPERIMENTS` so the RT-DETR core decides which experiments to run.

## Models (`models/`)

This repo provides detector wrappers for a few backends:
- `models/rtdetr_detector.py`: HuggingFace `RTDetrForObjectDetection` wrapper with `save_pretrained` support.
- `models/faster_rcnn_detector.py`, `models/retinanet_detector.py`: torchvision-based alternatives.
- `models/detector_utils.py`: common prediction + “export predictions to COCO json” helpers.

Most paper experiments use RT-DETR; the others exist for comparisons/ablations.

## Evaluation (`evaluation/`)

COCO metric core:
- `evaluation/coco_eval.py` wraps `pycocotools` and respects `config.COCO_MAX_DETS`.

Common evaluation scripts you’ll use:
- `evaluation/eval_best_test_plus.py`: “best checkpoint” evaluation with extra diagnostics (size buckets, etc.).
- `evaluation/eval_m0_m1_variants.py`: compares two checkpoints across multiple image variants.
- `evaluation/test_AB_on_CD.py`: evaluates two models (A trained on CAD, B trained on scans) on test variants C/D,
  writes per-class metrics and summary JSONs (either via `--config` YAML or direct `--model-a-dir/--model-b-dir` args).

## Subset evaluation (no retraining)

`paper_experiments/subset_softmax_eval.py` evaluates a trained checkpoint on a *subset* of classes on the
scanned test set, in a “subset-softmax” way:
- It renormalizes logits over `subset_classes + background`,
  then runs COCO evaluation restricted to those classes.
- It also prints per-class AP/AP50/AP75 for the subset.

This is intended for analyses like `{WINDOW, DOOR}` or `{WALL, WINDOW, DOOR}` without retraining.

## Mechanism-driven mitigation experiments (causal probes; short controlled fine-tuning runs)

These runs are analytical interventions rather than a standalone method contribution. They are short fine-tuning runs intended to probe the
failure mechanisms described in the paper section “Mechanism-Driven Mitigation Experiments”.

Canonical reference implementation (upstream in your CubiCasaVec checkout):
- `<your CubiCasaVec checkout>/RT_DETR_final/experiment_recipes/mechanism_mitigation/run_mechanism_mitigation.sh`

This repo includes a wrapper with the **same four arms** but wired to this repo’s training entrypoint:
- `mitigation/run_mechanism_mitigation.sh`

Arms (paper → code):
- Arm A (Boundary Instability): `AUGMENT_BOX_JITTER_*` + `AUGMENT_BOX_EXPAND_RATIO`
- Arm B (Thin Structure Survival): `AUGMENT_STROKE_*` + `AUGMENT_LINE_DROPOUT_*`
- Arm C (Depiction Deformation): `AUGMENT_DEPICTION_*` + `AUGMENT_DEPICTION_THRESHOLD`
- Arm D (Combined): enables A + B + C together

Usage:
1) Train the scanned-only baseline (this creates `exp2_scanned/checkpoints/best`):
   `python training/train.py --preset scan --run-name E2_scan_baseline`
2) Run the four fine-tunes starting from that baseline:
   `bash mitigation/run_mechanism_mitigation.sh runs/E2_scan_baseline`

## Diagnostics / visualization (`diagnostic/`)

These scripts are intended to analyze dataset and model behavior:
- `render_4_variants.py`: builds the A/B/C/D image variants (and the polygon-clipped variant) per drawing folder.
- `inspect_one_image.py`: inspects ranked predictions for one image, draws top-k overlays, and computes
  preliminary “recall within top-K” checks vs COCO-groundtruth.
- `render_boxes_union_polygon.py`: utility for producing the “union-of-boxes polygon” mask used in some variants.
- `size_distribution_per_class.py`, `per_class_coco.py`: dataset/metric summaries.

## Outputs (what gets written where)

Most runs write under:
- `runs/<RUN_NAME>/...`

You will commonly see:
- `runs/<RUN_NAME>/config.json`: resolved config snapshot for reproducibility
- `runs/<RUN_NAME>/checkpoints/best/`: best checkpoint directory (HuggingFace format)
- `runs/<RUN_NAME>/metrics/`: COCO annotations/predictions + JSON summaries
- `runs/<RUN_NAME>/visualizations/`: qualitative overlays

## Troubleshooting

- “`BASE_DIR is not set`”: set `DATASET_BASE_DIR`, create `configs/config_local.py`, or leave
  the downloaded dataset in the default cache after running `python tools/fetch_dataset.py`.
- “Pinned checkpoint commands fail”: `stable_runs/` and other paper-preparation assets are not included in the public GitHub repo.
  Public users can use the dataset tooling and code pipeline, but the exact pinned internal reproduction commands require separate checkpoint assets.
- Import errors like `ModuleNotFoundError: data`: ensure the RT-DETR project checkout that provides `data/`,
  `eval/`, and `RT_DETR_final.main_3_experiments` is on `PYTHONPATH` (or set `RT_DETR_FINAL_DIR` accordingly).
- CPU-only: many scripts auto-fall back to CPU, but runtime will be slow; pass `--device cpu` where supported.

## Where to look next

- Paper plotting/analysis scripts live in `paper_experiments/` (see `paper_experiments/README.md`).
- Mechanism-driven mitigation experiments are documented in `mitigation/README.md`.
