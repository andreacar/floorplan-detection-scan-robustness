#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------
# Imports from project
# ------------------------------------------------------------------
from transformers import AutoImageProcessor
from data.dataset import GraphRTDetrDataset, apply_subset
from utils.paths import load_split_list
from utils.geometry import clamp_bbox_xywh
import config as config_module
from config import *
import hierarchy_config as hier


# ------------------------------------------------------------------
# Filtering diagnostics (mirrors GraphRTDetrDataset logic)
# ------------------------------------------------------------------
def count_filtered_samples(folders, max_nodes):
    total = 0
    kept = 0
    filtered_over_max = 0
    filtered_other = 0

    for folder in folders:
        total += 1

        img = os.path.join(folder, config_module.IMAGE_FILENAME)
        graph = os.path.join(folder, "graph.json")

        if not (os.path.exists(img) and os.path.exists(graph)):
            filtered_other += 1
            continue

        with open(graph, "r") as f:
            data = json.load(f)

        with Image.open(img) as im:
            img_w, img_h = im.size

        mapped_count = 0
        valid_count = 0
        total_count = 0

        for node in data.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue

            total_count += 1
            clamped = clamp_bbox_xywh(bbox, img_w, img_h)
            if clamped is None:
                continue

            valid_count += 1
            raw = node.get("data_class", "") or node.get("category", "")
            l2 = hier.map_raw_to_l2(str(raw).strip())
            if l2 is not None:
                mapped_count += 1

        if total_count > max_nodes:
            filtered_over_max += 1
            continue

        if mapped_count == 0:
            filtered_other += 1
            continue

        if valid_count > 0 and (mapped_count / valid_count) < MIN_MAPPED_RATIO:
            filtered_other += 1
            continue

        kept += 1

    return {
        "total": total,
        "kept": kept,
        "filtered_over_max": filtered_over_max,
        "filtered_other": filtered_other,
    }


# ------------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------------
def summarize_dataset(name, dataset):
    img_per_class = defaultdict(int)
    inst_per_class = defaultdict(int)

    for i in tqdm(range(len(dataset)), desc=f"Scanning {name}"):
        sample = dataset[i]
        labels = sample["labels"]

        if "class_labels" not in labels:
            continue

        classes = labels["class_labels"].tolist()
        unique_classes = set(classes)

        for c in unique_classes:
            img_per_class[c] += 1
        for c in classes:
            inst_per_class[c] += 1

    print(f"\n=== DATASET SUMMARY: {name} ===")
    print(f"Total images: {len(dataset)}")
    print(f"{'Class':<12} {'Imgs':>6} {'Inst':>6}")
    print("-" * 28)

    for cid, cname in dataset.id2label.items():
        print(
            f"{cname:<12} "
            f"{img_per_class.get(cid, 0):>6} "
            f"{inst_per_class.get(cid, 0):>6}"
        )

    print("-" * 28)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading splits...")
    train_dirs = load_split_list(TRAIN_TXT)
    val_dirs = load_split_list(VAL_TXT)
    test_dirs = load_split_list(TEST_TXT)

    print("\n=== FILTERING DIAGNOSTICS ===")
    for max_nodes in [100, 200]:
        print(f"\n--- MAX_NODES = {max_nodes} ---")
        for name, folders in [
            ("TRAIN", train_dirs),
            ("VAL", val_dirs),
            ("TEST", test_dirs),
        ]:
            stats = count_filtered_samples(folders, max_nodes)
            print(
                f"{name}: total={stats['total']} | "
                f"kept={stats['kept']} | "
                f">{max_nodes} nodes filtered={stats['filtered_over_max']} | "
                f"other filtered={stats['filtered_other']}"
            )
    print("================================\n")

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(BACKBONE)

    processor.do_resize = RESIZE_ENABLE
    if RESIZE_FIXED:
        processor.size = {"height": RESIZE_HEIGHT, "width": RESIZE_WIDTH}
    else:
        processor.size = {
            "shortest_edge": RESIZE_SHORTEST_EDGE,
            "longest_edge": RESIZE_LONGEST_EDGE,
        }

    processor.do_pad = PAD_ENABLE
    processor.pad_size = {"height": PAD_SIZE, "width": PAD_SIZE} if PAD_ENABLE else None

    print("Building datasets...")
    ds_train = GraphRTDetrDataset(train_dirs, processor, "hierarchy_config.py", augment=False)
    ds_val = GraphRTDetrDataset(val_dirs, processor, "hierarchy_config.py", augment=False)
    ds_test = GraphRTDetrDataset(test_dirs, processor, "hierarchy_config.py", augment=False)

    apply_subset(ds_train, SUBSET_TRAIN)
    apply_subset(ds_val, SUBSET_VAL)
    apply_subset(ds_test, SUBSET_TEST)

    summarize_dataset("TRAIN", ds_train)
    summarize_dataset("VAL", ds_val)
    summarize_dataset("TEST", ds_test)


if __name__ == "__main__":
    main()
