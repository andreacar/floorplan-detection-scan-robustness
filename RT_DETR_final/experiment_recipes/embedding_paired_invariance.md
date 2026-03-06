# Embedding experiment: paired invariance fine-tuning

Purpose
- Regularize the backbone so clean/scanned representations remain close without sacrificing detection loss. This is your strongest embedding-driven experiment; it explicitly uses paired data to constrain drift while keeping the detector heads untouched.

Data/layout prep
1. Identify 200–400 paired scanned layouts that correspond to clean CAD (the same layouts that are already in your scanned `train.txt`).
2. Save the list to `experiment_data/paired_layouts.txt` (same format as `train.txt`).

Command (after the paired loader is wired in—see the next section)
```
cd RT_DETR_final
RUN_NAME=emb_paired_inv_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
TRAIN_TXT=/home/andrea/PycharmProjects/CubiCasaVec/RT_DETR_final/experiment_data/paired_layouts.txt \
SCANNED_INIT_WEIGHTS_DIR=/home/andrea/PycharmProjects/CubiCasaVec/RT_DETR_final/runs/20260115_134958/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=1 \
LR=1e-5 \
INVARIANCE_LAMBDA=0.05 \
python train.py
```

Notes
- The paired loader needs to return `(clean, scanned)` for each layout and trigger the invariance loss `L_inv = mean(||f_clean - f_scanned||^2)`. The new `INVARIANCE_LAMBDA` env var controls how hard that regularizer is applied; keep it ≤0.1.
- The script above assumes you will either modify `main_3_experiments.py` to support paired batches or wrap the dataset; if you don’t have support yet, run the risk + distance experiments first.
- Save the metrics from `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` and append to the comparison file.
