# B0 – Baseline re-run

Purpose
- Re-run the existing clean/scanned pipeline so you can compare every later experiment against the same checkpoints and metric output (clean CAD validation vs. scanned validation).

Setup
- Datasets: `TRAIN_TXT` = `/media/andrea/CubiCasaVec_data/train.txt`; `VAL_TXT` = `/media/andrea/CubiCasaVec_data/val.txt` (scanned layouts); clean validation comes from the `exp1_clean` run because that same `val.txt` is evaluated with `IMAGE_FILENAME=model_baked.png`.
- Seed: `42` (default).
- Default training recipe: `EPOCHS=100`, `LR=1e-4`, `AUGMENT_SCAN_MIX_ENABLE=0`, `AUGMENT_STROKE_ENABLE=0`, `SUBSET_* = 0`.

Command (run from `RT_DETR_final`):
```
RUN_NAME=b0_baseline_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=clean+scanned \
SEED=42 \
python train.py
```
This will create a master run under `runs/$RUN_NAME` with `exp1_clean` (clean CAD baseline) and `exp2_scanned` (scanned baseline).

Metrics & output
- Scrape `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` for Val-scanned Recall@0.85, AP@0.75, AP@0.85 and `runs/$RUN_NAME/exp1_clean/experiment_summary.json` for Val-clean.
- Save the JSON line that the tracker asked for (the schema is in the request):
```
{
  "run_id": "b0-baseline-<timestamp>",
  "seed": 42,
  "val_scanned": {
    "recall85": <from exp2_scanned>,
    "ap85": <from exp2_scanned>,
    "ap75": <from exp2_scanned>
  },
  "val_clean": {
    "recall85": <from exp1_clean>,
    "ap85": <from exp1_clean>,
    "ap75": <from exp1_clean>
  }
}
```
- The `experiment_summary.json` files already contain `best_ap75`, `best_ap85`, `best_recall`, and `best_recall85` so you can copy those numbers directly.

Notes
- Do not modify augmentations or subsets for this run; it must match the previous recipe so the +Δ runs are comparable.
