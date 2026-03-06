import os
import json
from collections import Counter

from PIL import Image, ImageDraw
import torch

from utils.geometry import clamp_bbox_xywh
from config import CLASS_COLORS, MAX_VIS
from models.detector_utils import detector_predict_post


# ============================================================
# FINAL INFERENCE POLICY (use for visualization + export later)
# ============================================================
THR = 0.01          # low to keep WALL (and other conservative scores)
TOPK_PRE = 150      # take more first
FINAL_K = 100       # final number of boxes you want to "ship"

# Prevent one class (often RAILING) from hogging the budget.
PER_CLASS_CAP = {
    "WALL": 60,
    "DOOR": 30,
    "WINDOW": 30,
    "COLUMN": 20,
    "STAIR": 20,
    "RAILING": 20,
}

PRINT_ANALYTICS = True


def _label_name(dataset, lab_int: int) -> str:
    # dataset.id2label is usually int->str; fall back safely.
    if hasattr(dataset, "id2label") and isinstance(dataset.id2label, dict):
        return dataset.id2label.get(lab_int, str(lab_int))
    return str(lab_int)


def _apply_final_policy(post, dataset):
    """
    post: dict with keys boxes, labels, scores (torch tensors)
    Returns filtered post + counts
    """
    # 1) Top-K globally (pre-cap)
    if len(post["scores"]) > TOPK_PRE:
        keep = torch.topk(post["scores"], TOPK_PRE).indices
        for k in post:
            post[k] = post[k][keep]

    # 2) Per-class cap + final K (greedy by descending score)
    kept_idx = []
    counts = Counter()

    order = torch.argsort(post["scores"], descending=True)
    for idx in order.tolist():
        lab = int(post["labels"][idx].item())
        name = _label_name(dataset, lab)
        cap = PER_CLASS_CAP.get(name, 20)  # default cap for unexpected classes

        if counts[name] < cap:
            kept_idx.append(idx)
            counts[name] += 1

        if len(kept_idx) >= FINAL_K:
            break

    if len(kept_idx) == 0:
        return post, counts  # nothing to keep (rare)

    kept_idx = torch.tensor(kept_idx, device=post["scores"].device, dtype=torch.long)
    for k in post:
        post[k] = post[k][kept_idx]

    return post, counts


def visualize_predictions(model, dataset, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # IMPORTANT: your dataset.samples list is the source of (img_path, graph_path, ...)
    n = min(MAX_VIS, len(dataset.samples))

    for i in range(n):
        img_path, graph_path, _ = dataset.samples[i]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # -------------------------
        # Draw GT (left)
        # -------------------------
        gt_img = img.copy()
        dr_gt = ImageDraw.Draw(gt_img)

        with open(graph_path, "r") as f:
            target = json.load(f)

        for node in target.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue
            clamped = clamp_bbox_xywh(bbox, w, h)
            if clamped is None:
                continue
            x, y, bw, bh = clamped
            dr_gt.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 0), width=3)

        # -------------------------
        # Model inference
        # -------------------------
        # Post-process with a LOW threshold (proposal-friendly)
        post = detector_predict_post(model, dataset, img, device, score_thresh=THR)

        # Analytics BEFORE caps
        if PRINT_ANALYTICS:
            pre_counts = Counter()
            wall_scores = []
            for lab, score in zip(post["labels"], post["scores"]):
                name = _label_name(dataset, int(lab.item()))
                pre_counts[name] += 1
                if name == "WALL":
                    wall_scores.append(float(score))
            print(f"[VIS {i}] pre_policy total={len(post['labels'])} | per-class={dict(pre_counts)}")
            if wall_scores:
                print(
                    f"[VIS {i}] pre_policy WALL scores: "
                    f"min={min(wall_scores):.3f} mean={sum(wall_scores)/len(wall_scores):.3f} max={max(wall_scores):.3f}"
                )

        # Apply FINAL POLICY (caps + final K)
        post, post_counts = _apply_final_policy(post, dataset)

        # Analytics AFTER caps
        if PRINT_ANALYTICS:
            wall_scores2 = []
            for lab, score in zip(post["labels"], post["scores"]):
                if _label_name(dataset, int(lab.item())) == "WALL":
                    wall_scores2.append(float(score))
            print(f"[VIS {i}] post_policy total={len(post['labels'])} | per-class={dict(post_counts)}")
            if wall_scores2:
                print(
                    f"[VIS {i}] post_policy WALL scores: "
                    f"min={min(wall_scores2):.3f} mean={sum(wall_scores2)/len(wall_scores2):.3f} max={max(wall_scores2):.3f}"
                )
            else:
                print(f"[VIS {i}] post_policy WALL: NONE kept")

        # -------------------------
        # Draw predictions (right)
        # -------------------------
        pred_img = img.copy()
        dr_pr = ImageDraw.Draw(pred_img)

        for box, lab, score in zip(post["boxes"], post["labels"], post["scores"]):
            x1, y1, x2, y2 = box.tolist()
            label = _label_name(dataset, int(lab.item()))
            color = CLASS_COLORS.get(label, (255, 0, 0))
            dr_pr.rectangle([x1, y1, x2, y2], outline=color, width=3)
            dr_pr.text((x1 + 3, y1 + 3), f"{label} {float(score):.2f}", fill=color)

        # Side-by-side canvas
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(gt_img, (0, 0))
        canvas.paste(pred_img, (w, 0))
        canvas.save(os.path.join(out_dir, f"vis_{i}.png"))


def debug_visualize_dataset(dataset, out_dir="./debug_visuals", max_items=10):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(max_items, len(dataset.samples))):
        img_path, graph_path, _ = dataset.samples[i]
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        with open(graph_path, "r") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue
            clamped = clamp_bbox_xywh(bbox, img.width, img.height)
            if clamped is None:
                continue
            x, y, bw, bh = clamped
            draw.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 0), width=3)

            raw = node.get("data_class", "") or node.get("category", "")
            l2 = dataset.map_raw_to_l2(raw) if hasattr(dataset, "map_raw_to_l2") else None
            if l2:
                draw.text((x + 3, y + 3), l2, fill=(0, 255, 0))

        img.save(os.path.join(out_dir, f"debug_{i}.png"))
