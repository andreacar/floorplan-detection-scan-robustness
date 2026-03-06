# Paper Experiments (RT-DETR)

This folder contains three standalone scripts for the paper experiments:

1) Embedding distance vs error + PCA visualization  
2) Error decomposition (missed / loose / tight)  
3) Success vs object size (logistic fit)
4) Factorized degradation study (synthetic factors)
5) HQ pair scoring (alignment + edge F1)
6) HQ pair scoring (all splits)
7) Manual review (mark incomplete pairs)
8) HQ pair step-by-step analysis (single example)

All scripts use the existing RT-DETR model checkpoints and the dataset split in `config.TEST_TXT`.
Run the commands from this repo root (so `config.py`, `data/`, and `paper_experiments/` are importable).

## 1) Embedding distance vs error + PCA

Example (clean vs scanned pairs, same model):

```bash
python paper_experiments/embedding_distance_pca.py \
  --ckpt runs/20260113_215642/exp1_clean/checkpoints/best \
  --image-a model_baked.png \
  --image-b F1_scaled.png \
  --out-dir paper_experiments/out/embedding_clean_model
```

Outputs:
- `pairs.json` (per-image metrics + distances)
- `summary.json` (correlations)
- `distance_vs_delta.png`
- `distance_deciles.png`
- `pca_pairs.png`

## 2) Error decomposition (missed / loose / tight)

Single run:

```bash
python paper_experiments/error_decomposition.py \
  --ckpt runs/20260113_215642/exp2_scanned/checkpoints/best \
  --image-name F1_scaled.png \
  --out-dir paper_experiments/out/error_scanned
```

Compare regimes in one plot:

```bash
python paper_experiments/error_decomposition.py \
  --run name=clean,ckpt=runs/20260113_215642/exp1_clean/checkpoints/best,image=model_baked.png \
  --run name=scanned,ckpt=runs/20260113_215642/exp2_scanned/checkpoints/best,image=F1_scaled.png \
  --out-dir paper_experiments/out/error_compare
```

Outputs:
- `<run>_decomp.json` per run
- `summary.json`
- `error_decomposition.png`

## 3) Success vs object size (logistic fit)

Compare regimes in one plot:

```bash
python paper_experiments/size_success.py \
  --run name=clean,ckpt=runs/20260113_215642/exp1_clean/checkpoints/best,image=model_baked.png \
  --run name=scanned,ckpt=runs/20260113_215642/exp2_scanned/checkpoints/best,image=F1_scaled.png \
  --out-dir paper_experiments/out/size_compare
```

Outputs:
- `<run>_size_success.json` per run
- `summary.json`
- `size_success.png`

## 4) Factorized degradation study

```bash
python paper_experiments/factorized_degradation.py \
  --ckpt runs/20260113_215642/exp1_clean/checkpoints/best \
  --image-name model_baked.png \
  --out-dir paper_experiments/out/factorized_test
```

Optional:
- `--factors blur,thicken,texture,clutter`
- `--levels mild,medium,strong`
- `--with-embedding` (adds embedding-distance correlations)

Output:
- `summary.json`

## 5) HQ pair scoring (alignment + edge F1)

```bash
python paper_experiments/hq_pairs_scoring.py \
  --image-clean model_baked.png \
  --image-scan F1_scaled.png \
  --out-dir paper_experiments/out/hq_pairs
```

Outputs:
- `pairs.json` (per-pair alignment + cov/prec/F1)
- `summary.json` (distribution stats)
- `pair_score_hist.png`
- `cov_prec_scatter.png`

## 6) HQ pair scoring (all splits)

```bash
python paper_experiments/hq_pairs_all_splits.py \
  --out-dir paper_experiments/out/hq_pairs_all
```

Optional: pass any `hq_pairs_scoring.py` args after the command (e.g. `--rotations 0`).

## 7) Manual review (mark incomplete pairs)

Use `pairs.json` from a run and press space to toggle marking the current pair.

```bash
python paper_experiments/hq_pairs_review.py \
  --pairs-json paper_experiments/out/hq_pairs/pairs.json \
  --marked-file paper_experiments/out/hq_pairs/incomplete_ids.txt
```

Keys:
- `n` / right arrow: next
- `p` / left arrow: previous
- space: toggle mark
- `q`: quit

## 8) HQ pair step-by-step analysis (single example)

Visualize preprocessing, edges, ROI, and alignment for one example.

```bash
python paper_experiments/hq_pairs_debug.py \
  --pairs-json paper_experiments/out/hq_pairs/pairs.json \
  --index 0
```

Alternative:

```bash
python paper_experiments/hq_pairs_debug.py \
  --folder /media/andrea/CubiCasaVec_data/high_quality/8731
```

## Notes
- Use `--score-thresh` or `--use-per-class-thresh` to match your evaluation policy.
- Use `--limit` for small validation runs.
- All scripts default to `config.TEST_TXT` unless `--test-txt` is set.
- To build consolidated CSV tables for submission from generated JSON summaries:
  `python -m paper_experiments.make_paper_tables`
