# T2 – Within-layout mixed-LM variants

Purpose
- Force the model to learn depiction invariance by repeating a small set of scanned layouts with controlled stroke/blur intensities so that geometry stays constant while the depiction changes.

Data construction
1. Pick 80 layouts from your scanned training split (use the same IDs you evaluate on or the ones that dominate Val-scanned) and write them into `runs/t2_variant_list.txt` (one absolute path per line, same format as `train.txt`).
2. Expand this list into 8 variants per layout by repeating each line eight times in a new file (`runs/train_mixedlm.txt`). You can annotate the repeat order with a suffix for bookkeeping if you like (e.g., add `|variant=stroke-only` after the path and strip it in `utils/paths` if necessary). The repetition guarantees the network sees the same geometry under multiple augmentation draws.
3. Because you now have 640 entries (= 80 × 8), keep `SUBSET_TRAIN=640` (or leave it at 0 once the file is fixed).

Augmentation plan
- Use the same stroke/blur combinations as in T1 but vary the randomness so that each pass is likely to fall into a different category. Set:
  - `AUGMENT_STROKE_ENABLE=1`
  - `AUGMENT_STROKE_PROB=0.9`
  - `AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4`
  - `AUGMENT_STROKE_KERNEL_PROBS=0.20,0.20,0.30,0.30`
  - `AUGMENT_STROKE_SIGMA_RANGE=0.0,2.2`
  - `AUGMENT_STROKE_CONTRAST_RANGE=0.9,1.1`
  - `AUGMENT_STROKE_DILATE_PROB=0.7`
  - Keep `AUGMENT_STROKE_BRIGHTNESS_RANGE` tight (`-3.0,3.0`).

Training schedule
- `RUN_EXPERIMENTS=scanned`
- Start from the B0 scanned checkpoint:
  `SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0_RUN_NAME>/exp2_scanned/checkpoints/best`
- `EPOCHS=3` (or 2500–5000 iterations on the 640-sample file; set `EPOCHS=3` and `BATCH_SIZE=8` to cover about 2400 samples per epoch).
- `LR=1e-5`
- `SUBSET_TRAIN=0` (since `train_mixedlm.txt` already limits the set).

Command example:
```
RUN_NAME=t2_mixedlm_$(date +%Y%m%d_%H%M%S) \
RUN_EXPERIMENTS=scanned \
TRAIN_TXT=/media/andrea/CubiCasaVec_data/train_mixedlm.txt \
SCANNED_INIT_WEIGHTS_DIR=/path/to/runs/<B0>/exp2_scanned/checkpoints/best \
EPOCHS=3 \
LR=1e-5 \
AUGMENT_STROKE_ENABLE=1 \
AUGMENT_STROKE_PROB=0.9 \
AUGMENT_STROKE_KERNEL_SIZES=1,2,3,4 \
AUGMENT_STROKE_KERNEL_PROBS=0.2,0.2,0.3,0.3 \
AUGMENT_STROKE_SIGMA_RANGE=0.0,2.2 \
AUGMENT_STROKE_CONTRAST_RANGE=0.9,1.1 \
python train.py
```
(If you want the variants to correspond to stroke-only vs. blur-only, regenerate `train_mixedlm.txt` with grouped repeats or run this command multiple times with different `AUGMENT_STROKE_SIGMA_RANGE` / `AUGMENT_STROKE_PROB` values and concatenate the results.)

Evaluation
- Same as T1: scraper uses Val-scanned metrics from `exp2_scanned` and clean metrics either from a dedicated clean evaluation run or by retraining/evaluating `exp1_clean` with the final checkpoint as init.
- Record the JSON line for this run after copying the relevant metrics from `runs/$RUN_NAME/exp2_scanned/experiment_summary.json` (AP@0.85 is now available under `best_ap85`).
