# Dataset Setup

Do not commit raw datasets to Git.

## 1) Obtain the data
- CubiCasa-style floorplan data with `graph.json` annotations.
- Paired scanned raster counterpart per layout.
- Ensure you have rights to use/distribute the data for your venue.

## 2) Expected directory layout

Set a root directory (called `<DATASET_BASE_DIR>` below) with:

```text
<DATASET_BASE_DIR>/
  train.txt
  val.txt
  test.txt
  colorful/
  high_quality/
  high_quality_architectural/
```

Each split file should list layout folders, one per line, e.g.
`/high_quality_architectural/1191/`.

Each listed layout folder is expected to contain at least:
- `graph.json`
- `model_baked.png` (clean CAD raster)
- `F1_scaled.png` (scan raster)

For causal ROI experiments (C/D), each layout should also contain:
- `four_final_variants/03_scan_inside_boxes.png`
- `four_final_variants/04_svg_clean_plus_scan_outside.png`

## 3) Configure the repo to point to the dataset

Either:
- `export DATASET_BASE_DIR=/path/to/dataset`

Or create `configs/config_local.py` from `configs/config_local_example.py` and set `DEFAULT_BASE_DIR`.

## 4) Generate A/B/C/D variants (if missing)

The variant generator in this repo is:
- `diagnostic/render_4_variants.py`

Example:

```bash
conda run -n <env_name> python diagnostic/render_4_variants.py \
  --root-data /path/to/dataset/high_quality \
  --save-mode dataset \
  --jobs 8
```

Optional: if you keep defaults in a YAML, pass `--config <yaml>` with keys `root_data` and (optionally) `hier_json`.

## 5) Verification before running paper suites

Split verification:

```bash
head -n 5 "$DATASET_BASE_DIR/test.txt"
```

C/D variant + annotation verification (small validation run):

```bash
head -n 5 "$DATASET_BASE_DIR/test.txt" > /tmp/test_small.txt
TEST_TXT=/tmp/test_small.txt conda run -n <env_name> python -m evaluation.test_AB_on_CD \
  --model-a-dir stable_runs/20260115_134958/exp1_clean/checkpoints/best \
  --model-b-dir stable_runs/20260115_134958/exp2_scanned/checkpoints/best \
  --out-dir /tmp/ab_on_cd_smoke
```

Qualitative panel verification:

```bash
TEST_TXT=/tmp/test_small.txt conda run -n <env_name> python evaluation/qualitative_abcd.py \
  --model-a stable_runs/20260115_134958/exp1_clean/checkpoints/best \
  --model-b stable_runs/20260115_134958/exp2_scanned/checkpoints/best \
  --max-layouts 2 \
  --out-dir /tmp/qual_abcd_smoke
```
