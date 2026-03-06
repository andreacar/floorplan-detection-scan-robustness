# T3 – ROI-weighted fine-tune

Purpose
- Encourage training to focus on the layout interior where `dist_roi` is high by sampling/weighting examples with more ROI coverage instead of relying on background patterns.

Setup
- Start from the B0 scanned checkpoint: `SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0>/exp2_scanned/checkpoints/best`
- Scale the dataset down to 300–500 layouts using `SUBSET_TRAIN=400`
- Keep the augmentation plan permissive (stroke + blur) so the model still sees depiction noise—re-use the same settings as T1 if you like, or keep them at the default CAD-to-scan mix.

ROI weighting knobs (supported by the new sampler):
- `ROI_WEIGHT_ENABLE=1` to switch on the weighted sampler
- `ROI_WEIGHT_SCALE=2.0` to double the weight of layouts with high ROI coverage
- `ROI_WEIGHT_BIAS=1.0` keeps the base weight at 1

Training schedule
- `RUN_EXPERIMENTS=scanned`
- `EPOCHS=2` (extend to 5 only if the weight sampling needs more cycles)
- `LR=1e-5`

Command example:
```
RUN_NAME=t3_roi_weighting_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0>/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=2 \
LR=1e-5 \
ROI_WEIGHT_ENABLE=1 \
ROI_WEIGHT_SCALE=2.0 \
ROI_WEIGHT_BIAS=1.0 \
python train.py
```

Evaluation & logging
- Collect Val-scanned metrics from `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` and copy Recall@0.85 / AP@0.75 / AP@0.85.
- The JSON line should include this run ID and the usual `val_scanned`/`val_clean` block (clean metrics are still available by re-evaluating `exp1_clean` from the checkpoint if necessary).
