# Embedding experiment: distance-sensitive augmentation curriculum

Purpose
- Apply stroke/blur at graduated strengths per layout so only the risky layouts see the strongest degradation. This follows the low/mid/high buckets you distilled from the dist_roi signal and keeps clean layouts intact.

Preparation
1. Reuse the same `dist_scores.csv` from the risk experiment (layout_path, normalized score).
2. Copy `train.txt` to a new file `experiment_data/train_distance_control.txt`. Use the same list of layouts but keep the additional bucket column if you want to track it—only the relative path matters for training.
3. You can optionally pre-compute percentiles to verify thresholds of 0.5/0.85; the script below uses those by default.

Command
```
cd RT_DETR_final
RUN_NAME=emb_distance_control_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
TRAIN_TXT=/home/andrea/PycharmProjects/CubiCasaVec/RT_DETR_final/experiment_data/train_distance_control.txt \
SCANNED_INIT_WEIGHTS_DIR=/home/andrea/PycharmProjects/CubiCasaVec/RT_DETR_final/runs/20260115_134958/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=1 \
LR=1e-5 \
DISTANCE_SCORE_FILE=/home/andrea/PycharmProjects/CubiCasaVec/RT_DETR_final/experiment_data/dist_scores.csv \
DISTANCE_CURRICULUM_ENABLE=1 \
DISTANCE_CURRICULUM_THRESHOLDS=0.5,0.85 \
DISTANCE_CURRICULUM_LOW_SIGMA=0.0,0.8 \
DISTANCE_CURRICULUM_LOW_KERNELS=0,1 \
DISTANCE_CURRICULUM_MID_SIGMA=0.4,1.6 \
DISTANCE_CURRICULUM_MID_KERNELS=1,2,3 \
DISTANCE_CURRICULUM_HIGH_SIGMA=1.0,2.2 \
DISTANCE_CURRICULUM_HIGH_KERNELS=2,3,4 \
python train.py
```

Notes
- When `DISTANCE_CURRICULUM_ENABLE=1` the stroke/blur strength is picked per sample according to the value from `dist_scores.csv` (low/mid/high buckets). Only scanned inputs are distorted; clean layouts remain untouched.
- Keep `EPOCHS=1` (maybe 2 if you want a second pass) and `LR=1e-5`. Capture metrics from `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` and add to the comparison file.
