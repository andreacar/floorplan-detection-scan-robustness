# T4 – Stroke+Blur curriculum + ROI weighting

Purpose
- This is the “power combo”: T1’s depiction curriculum + T3’s ROI-focused sampling. You fine-tune from the scanned baseline with aggressive stroke/blur augmentations while giving more samples priority if they cover large ROIs.

Setup
- `SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0>/exp2_scanned/checkpoints/best`
- `SUBSET_TRAIN=400`
- `EPOCHS=2` (bump to 4–5 if you want more iterations on the combined signal)
- `LR=1e-5`

Augmentations (same as T1):
- `AUGMENT_STROKE_ENABLE=1`
- `AUGMENT_STROKE_PROB=0.8`
- `AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4`
- `AUGMENT_STROKE_KERNEL_PROBS=0.25,0.25,0.25,0.25`
- `AUGMENT_STROKE_SIGMA_RANGE=0.4,2.2`
- `AUGMENT_STROKE_CONTRAST_RANGE=0.85,1.15`
- `AUGMENT_STROKE_DILATE_PROB=0.7`

ROI weighting:
- `ROI_WEIGHT_ENABLE=1`
- `ROI_WEIGHT_SCALE=2.0`
- `ROI_WEIGHT_BIAS=1.0`

Command example:
```
RUN_NAME=t4_stroke_roi_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0>/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=2 \
LR=1e-5 \
AUGMENT_STROKE_ENABLE=1 \
AUGMENT_STROKE_PROB=0.8 \
AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4 \
AUGMENT_STROKE_KERNEL_PROBS=0.25,0.25,0.25,0.25 \
AUGMENT_STROKE_SIGMA_RANGE=0.4,2.2 \
AUGMENT_STROKE_CONTRAST_RANGE=0.85,1.15 \
AUGMENT_STROKE_DILATE_PROB=0.7 \
ROI_WEIGHT_ENABLE=1 \
ROI_WEIGHT_SCALE=2.0 \
ROI_WEIGHT_BIAS=1.0 \
python train.py
```

Evaluation
- Same logging expectations: store the JSON line with `val_scanned`/`val_clean` metrics and include the absolute deltas vs. B0 so reviewers see the gain.
