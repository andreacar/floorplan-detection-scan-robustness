# T1 – Stroke+Blur curriculum fine-tune

Purpose
- Fine-tune the scanned (`exp2_scanned`) branch so the detector has explicit exposure to the geometric distortions that were degrading Recall@0.85. This run starts from the B0 scanned checkpoint and only applies stroke/blur/contrast jitter, so it keeps the representation stable while improving robustness to depiction shift.

Setup
- Base init: point `SCANNED_INIT_WEIGHTS_DIR` to `runs/<B0_RUN_NAME>/exp2_scanned/checkpoints/best/` once the baseline run completes.
- Use a small subset so the fine-tuning run remains computationally efficient: `SUBSET_TRAIN=400` (you can try 300-500 layouts).
- Training schedule: `EPOCHS=2` (extend to 5 if you still see movement), `LR=1e-5` (0.1× baseline).
- Freeze backbone for the first 20% of steps if you can (manually zero-out gradients on `model.model.base_model` on the first epoch, then re-enable); otherwise rely on the low LR.

Augmentations
- Disable scan mix: `AUGMENT_SCAN_MIX_ENABLE=0`.
- Enable stroke/blur only:
  - `AUGMENT_STROKE_ENABLE=1`
  - `AUGMENT_STROKE_PROB=0.8` (so ~80% of CAD inputs get altered)
  - `AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4`
  - `AUGMENT_STROKE_KERNEL_PROBS=0.25,0.25,0.25,0.25`
  - `AUGMENT_STROKE_DILATE_PROB=0.7`
  - `AUGMENT_STROKE_SIGMA_RANGE=0.4,2.2`
  - `AUGMENT_STROKE_CONTRAST_RANGE=0.85,1.15`
  - Keep `AUGMENT_STROKE_BRIGHTNESS_RANGE` at the default `(-5,5)` so contrast jitter stays mild.

Command (run from `RT_DETR_final`):
```
RUN_NAME=t1_stroke_blur_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0_RUN_NAME>/exp2_scanned/checkpoints/best \
SUBSET_TRAIN=400 \
EPOCHS=2 \
LR=1e-5 \
AUGMENT_STROKE_ENABLE=1 \
AUGMENT_STROKE_PROB=0.8 \
AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4 \
AUGMENT_STROKE_KERNEL_PROBS=0.25,0.25,0.25,0.25 \
AUGMENT_STROKE_SIGMA_RANGE=0.4,2.2 \
AUGMENT_STROKE_CONTRAST_RANGE=0.85,1.15 \
python train.py
```

Evaluation
- Gather `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` for val-scanned metrics (Recall@0.85 / AP@0.75 / AP@0.85). The file now includes `best_ap85` / `best_recall85` because of the utility in `main_3_experiments`.
- To check clean performance, replay the checkpoint on the clean layout(s) by swapping `RUN_EXPERIMENTS=clean` and pointing `CLEAN_INIT_WEIGHTS_DIR` to `runs/$RUN_NAME/exp2_scanned/checkpoints/best`; the same training/eval script will then evaluate the scanned checkpoint on the CAD validation without modifying weights (set `EPOCHS=0` or `SUBSET_TRAIN=0` to skip training, but the easiest route is to reuse the existing clean pipeline and treat the run as a very short eval-only pass).

Logging
- Append the JSON line (same schema as B0) but use the T1 run ID. Include the absolute delta vs. B0 for scanned Recall@0.85 and clean Recall@0.85 so you can cite the +/− numbers.
