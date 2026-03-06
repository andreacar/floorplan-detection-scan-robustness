# Paper → Repo Mapping

This repository is structured to mirror the paper narrative.

## Data (A/B/C/D) + Verification
- Goal: build paired variants with identical boxes and verify alignment.
- Code: data generation + ROI variant code (see `data_generation/`, dataset utilities, ROI scripts)
- Outputs: A/B/C/D images + verification reports.

## Core Training (RT-DETR)
- CAD-only: train/test on CAD-derived (A)
- Scanned-only: train/test on scanned (B)

## Causal ROI Isolation (Geometry vs Background)
- Evaluate Model A and Model B on variants (C) and (D)

## Mechanism Diagnostics
- Error decomposition (miss/loose/tight)
- Classwise recall/IoU
- Size success logistic fits

## Factorized Degradations
- Blur / Thicken / Texture / Clutter sweeps + evaluation

## Mechanism Mitigation Fine-tunes
- Arms A/B/C/D starting from scanned baseline checkpoint

## Cross-Architecture Replication
- Faster R-CNN and RetinaNet on clean vs scanned

## Outputs convention
All generated tables/figures should go to `artifacts/tables` and `artifacts/figures`.
