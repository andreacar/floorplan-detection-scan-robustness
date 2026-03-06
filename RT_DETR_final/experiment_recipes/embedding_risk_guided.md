# Embedding experiment: risk-guided sampling

Purpose
- Use your `dist_roi` risk signal as a sample weight so training spends more iterations on the layouts that suffer the largest embedding drift. This is the most practical embedding-driven experiment, so it should run immediately after T1–T4.

Preparation
1. Export a CSV with one row per layout, e.g.: `high_quality_architectural/123,0.72`. The value is a normalized (`0..1`) version of your ROI distance (e.g., percentile or min-max). Keep the format `layout_path,score` where `layout_path` matches the relative path in `train.txt`.
2. Place the CSV somewhere in the repo, e.g. `RT_DETR_final/experiment_data/dist_scores.csv`.

Command
```
cd RT_DETR_final
RUN_NAME=emb_risk_alpha1_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/20260115_134958/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=1 \
LR=1e-5 \
DISTANCE_SCORE_FILE=experiment_data/dist_scores.csv \
DISTANCE_SCORE_ALPHA=1.0 \
python train.py
```
Repeat with `DISTANCE_SCORE_ALPHA=2.0` (or higher) to explore how strongly focusing on high-distance layouts changes Recall@0.85.

Notes
- The trainer will multiply the default sample weight (`image_weights`) by `(1 + alpha * dist_score)`. Scores below 0.05 become nearly uniform, so you get a slowly increasing emphasis on risky layouts. If you want to mix background boosts (`CLASS_BOOST`) and this signal, they already compose naturally.
- Record the resulting `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` metrics and append them to your comparison spreadsheet (see `experiment_recipes/embedding_suite_summary.csv`).
