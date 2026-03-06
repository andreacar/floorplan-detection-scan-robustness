# Reproducibility

## Environment
- `conda env create -f environment.yml`
- `conda activate <env>`
- If you use a pre-existing env, verify imports first:
  `python -c "import torch, torchvision, transformers, pycocotools, PIL, numpy, matplotlib"`

## Dataset placement
See `docs/DATASET.md`.

## Stable checkpoint assets
The public repository does not track the large pinned checkpoints in git history.
If you publish them through GitHub Releases, unpack them under:

- `stable_runs/20260115_134958/exp1_clean/`
- `stable_runs/20260115_134958/exp2_scanned/`

The expected model directories are:

- `stable_runs/20260115_134958/exp1_clean/checkpoints/best`
- `stable_runs/20260115_134958/exp2_scanned/checkpoints/best`

## What to reproduce
- Table 1: CAD-only vs Scanned-only (RT-DETR)
- ROI isolation table: geometry vs background
- Error decomposition + classwise stats
- Size success curves
- Factorized degradations
- Mechanism mitigation
- Cross-architecture

## Where outputs go
- Tables: `artifacts/tables/`
- Figures: `artifacts/figures/`

## One-command pipeline
These commands assume the local pinned checkpoint assets used during paper preparation are present.
Those artifacts are not included in the public GitHub repository snapshot.

- Dry run (validate command graph):
  `python tools/run_paper.py paper_all --dry`
- Execute pinned pipeline:
  `python tools/run_paper.py paper_all --mode pinned`

## Pre-submission validation
- `python tools/quickcheck_submission.py --strict`
