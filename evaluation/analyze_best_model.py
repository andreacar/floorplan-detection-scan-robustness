#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

from transformers import AutoImageProcessor, RTDetrForObjectDetection
from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset, apply_subset
from config import *


# ------------------------------------------------------------
# Temperature Scaler
# ------------------------------------------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T)

    def forward(self, logits):
        return logits / torch.exp(self.log_T)


# ------------------------------------------------------------
# Collect matched logits + labels (validation only)
# ------------------------------------------------------------
@torch.no_grad()
def collect_logits_and_targets(model, ds_val, device):
    all_logits = []
    all_targets = []

    print("Collecting matched logits + targets (validation set)…")

    for i in tqdm(range(len(ds_val))):
        sample = ds_val[i]

        pixel = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_mask = torch.ones(
            (1, pixel.shape[-2], pixel.shape[-1]),
            dtype=torch.long,
            device=device
        )

        outputs = model(pixel_values=pixel, pixel_mask=pixel_mask)

        # logits: [num_queries, num_classes]
        logits = outputs.logits[0]

        # matched labels from dataset (no-object already excluded)
        targets = sample["labels"]["class_labels"]

        all_logits.append(logits)
        all_targets.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_logits, all_targets


# ------------------------------------------------------------
# Detection-level score analysis
# ------------------------------------------------------------
@torch.no_grad()
def analyze_detections(model, processor, ds_val, device, T=None):
    pred_counter = Counter()
    score_stats = defaultdict(list)

    for i in tqdm(range(len(ds_val))):
        sample = ds_val[i]

        pixel = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_mask = torch.ones(
            (1, pixel.shape[-2], pixel.shape[-1]),
            dtype=torch.long,
            device=device
        )

        outputs = model(pixel_values=pixel, pixel_mask=pixel_mask)

        if T is not None:
            outputs.logits = outputs.logits / T

        H, W = sample["labels"]["orig_size"].tolist()
        processed = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.tensor([[H, W]], device=device)
        )[0]

        for lbl, score in zip(processed["labels"].cpu(), processed["scores"].cpu()):
            lbl = int(lbl)
            score = float(score)
            pred_counter[lbl] += 1
            score_stats[lbl].append(score)

    return pred_counter, score_stats


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BEST model + processor…")
    ckpt = os.path.join(CKPT_DIR, "best")
    processor = AutoImageProcessor.from_pretrained(ckpt)
    model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    id2label = model.config.id2label

    print("Loading validation dataset…")
    val_dirs = load_split_list(VAL_TXT)
    ds_val = GraphRTDetrDataset(val_dirs, processor, "hierarchy_config.py", augment=False)
    apply_subset(ds_val, SUBSET_VAL)

    # --------------------------------------------------------
    # BEFORE calibration
    # --------------------------------------------------------
    print("\n=== ANALYSIS BEFORE TEMPERATURE SCALING ===")
    pred_before, scores_before = analyze_detections(
        model, processor, ds_val, device, T=None
    )

    for k in sorted(scores_before):
        s = scores_before[k]
        print(
            f"{id2label[k]:>8s} | "
            f"mean={np.mean(s):.3f} "
            f"p90={np.percentile(s,90):.3f} "
            f"max={np.max(s):.3f}"
        )

    # --------------------------------------------------------
    # Collect logits + targets
    # --------------------------------------------------------
    all_logits, all_targets = collect_logits_and_targets(model, ds_val, device)

    # --------------------------------------------------------
    # Optimize temperature
    # --------------------------------------------------------
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(all_logits), all_targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = torch.exp(scaler.log_T).item()
    print(f"\n>>> Optimal temperature learned: T = {T:.3f}")

    # --------------------------------------------------------
    # AFTER calibration
    # --------------------------------------------------------
    print("\n=== ANALYSIS AFTER TEMPERATURE SCALING ===")
    pred_after, scores_after = analyze_detections(
        model, processor, ds_val, device, T=T
    )

    for k in sorted(scores_after):
        s = scores_after[k]
        print(
            f"{id2label[k]:>8s} | "
            f"mean={np.mean(s):.3f} "
            f"p90={np.percentile(s,90):.3f} "
            f"max={np.max(s):.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
