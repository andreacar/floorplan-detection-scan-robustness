# Reproducibility

## Environment
- `conda env create -f environment.yml`
- `conda activate <env>`
- If you use a pre-existing env, verify imports first:
  `python -c "import torch, torchvision, transformers, pycocotools, PIL, numpy, matplotlib"`

## Dataset placement
See `docs/DATASET.md`.

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
- Dry run (validate command graph):
  `python tools/run_paper.py paper_all --dry`
- Execute pinned pipeline:
  `python tools/run_paper.py paper_all --mode pinned`

## Pre-submission validation
- `python tools/quickcheck_submission.py --strict`
