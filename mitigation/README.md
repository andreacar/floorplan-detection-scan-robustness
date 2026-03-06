# Mechanism-Driven Mitigation Experiments

Purpose
- Test whether interventions aligned with the diagnosed failure mechanisms selectively improve the corresponding error modes.
- This is not a new detector proposal. It is a falsifiable mechanism test.

Key hypothesis (from the paper)
- Boundary ambiguity drives AP75 collapse (tight -> loose migration).
- Thin/small primitives fail first (size threshold shifts).
- Depiction deformation (stroke width + blur + boundary ambiguity) is the dominant driver.

Experimental design
- Baseline: scanned-only RT-DETR (exp2_scanned).
- Fine-tuning only (5–10 epochs) to isolate mechanism effects.
- Same data, same splits, no CAD mixing, lower LR for stability.
- Post-hoc robustness calibration on the scanned distribution.

Arms (match the paper)
A) Boundary Instability Arm
- Intervention: GT box jitter + expansion (tolerant regression target).
- Expected effect: AP75 improves more than AP50; tight matches increase; loose matches decrease.

B) Thin Structure Survival Arm
- Intervention: stroke erosion + partial line dropout.
- Expected effect: COLUMN/RAILING AP increases; Recall improves more than AP75; area@50% shifts down.

C) Depiction Deformation Arm
- Intervention: simulated depiction shift (dilate -> blur -> threshold).
- Expected effect: strict IoU improves; tight->loose migration reduces; mimics scan robustness.

D) Combined Arm (optional)
- Intervention: A + B + C together.
- Expected effect: gains larger than any single arm; still far from clean baseline.

Metrics to report
- AP, AP50, AP75
- Missed / Loose / Tight (IoU-based decomposition)
- Class-wise AP (COLUMN, RAILING emphasis)
- area@50% shift (size-robustness diagnostic)

Where results land
- Each run writes `runs/<RUN_NAME>/exp2_scanned/experiment_summary.json` and `augment_config.json`.
- Use existing diagnostics (`eval/diagnostics.py`) for tight/loose and class-wise recall.

Files in this folder
- run_mechanism_mitigation.sh: ready-to-run command wrapper for all arms.
- README.md (this file): experiment intent + interpretation guide.

Notes
- Set AUGMENT_APPLY_ALL=1 so augmentations apply to scanned images.
- Keep AUGMENT_SCAN_MIX_ENABLE=0 (not domain adaptation).
