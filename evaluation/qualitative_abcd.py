#!/usr/bin/env python3
"""
Qualitative grid for CAD/Scan/C/D variants across Model A/B.

Produces a 2x4 panel per layout:
  Row 1: Model A (CAD-trained)
  Row 2: Model B (scan-trained)
  Cols : CAD | Scan | C | D
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoImageProcessor

from utils.paths import load_split_list
from utils.geometry import clamp_bbox_xywh, compute_iou
from data.dataset import GraphRTDetrDataset
from models.detector_utils import detector_predict_post
from models.faster_rcnn_detector import FasterRCNNDetector
from models.retinanet_detector import RetinaNetDetector
from models.rtdetr_detector import RTDetrDetector

import config as config_module
from config import (
    BACKBONE,
    RESIZE_ENABLE,
    RESIZE_FIXED,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    RESIZE_SHORTEST_EDGE,
    RESIZE_LONGEST_EDGE,
    PAD_ENABLE,
    PAD_SIZE,
    TEST_TXT,
)


GT_COLOR = (0, 160, 0)
TP_COLOR = (0, 114, 178)
FP_COLOR = (220, 50, 32)
PRED_COLOR = TP_COLOR
GT_WIDTH = 3
PRED_WIDTH = 4
FILL_ALPHA = 0.0
GT_FILL_ALPHA = 0.0


def build_processor():
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
    return processor


def load_detector(detector_name: str, model_dir: str, device: torch.device):
    name = detector_name.lower()
    if name == "rtdetr":
        detector = RTDetrDetector.load(model_dir, device)
    elif name == "fasterrcnn":
        detector = FasterRCNNDetector.load(model_dir, device)
    elif name == "retinanet":
        detector = RetinaNetDetector.load(model_dir, device)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")
    detector.eval()
    return detector


def parse_indices(text: str, max_index: int) -> List[int]:
    if not text:
        return []
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0 or idx >= max_index:
            raise ValueError(f"Index {idx} out of range (0..{max_index - 1})")
        out.append(idx)
    return out


def read_graph_boxes(graph_path: str, label2id: Dict[str, int], map_raw_to_l2, img_w: int, img_h: int):
    with open(graph_path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)

    boxes_xyxy = []
    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        clamped = clamp_bbox_xywh(bbox, img_w, img_h)
        if clamped is None:
            continue
        x, y, bw, bh = clamped

        raw = node.get("data_class", "") or node.get("category", "")
        l2 = map_raw_to_l2(raw)
        if l2 not in label2id:
            continue

        boxes_xyxy.append((x, y, x + bw, y + bh, int(label2id[l2])))
    return boxes_xyxy


def filter_predictions(post, score_thresh: float, topk: int):
    boxes = post["boxes"]
    scores = post["scores"]
    labels = post["labels"]

    if scores.numel() == 0:
        return [], [], []

    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if scores.numel() == 0:
        return [], [], []

    if topk is not None and scores.numel() > topk:
        order = torch.argsort(scores, descending=True)[:topk]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

    return boxes.cpu().tolist(), scores.cpu().tolist(), labels.cpu().tolist()


def scale_boxes_xyxy(boxes: Sequence[Tuple[float, float, float, float, int]], sx: float, sy: float):
    out = []
    for x1, y1, x2, y2, lab in boxes:
        out.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy, lab))
    return out


def scale_pred_boxes(boxes: Sequence[Sequence[float]], sx: float, sy: float):
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, fill, font, bg=None):
    if not text:
        return
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
    else:
        w, h = font.getsize(text)
    if bg is not None:
        x, y = xy
        draw.rectangle([x - 1, y - 1, x + w + 2, y + h + 2], fill=bg)
    draw.text(xy, text, fill=fill, font=font)


def _rgba(color, alpha: float):
    r, g, b = color
    a = int(max(0.0, min(1.0, alpha)) * 255)
    return (int(r), int(g), int(b), a)


def _draw_dashed_rect(draw: ImageDraw.ImageDraw, box, color, width: int = 2, dash: int = 6, gap: int = 4):
    x1, y1, x2, y2 = box
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    if x2 <= x1 or y2 <= y1:
        return

    def _draw_segment(xa, ya, xb, yb):
        draw.line([(xa, ya), (xb, yb)], fill=color, width=width)

    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        _draw_segment(x, y1, x_end, y1)
        _draw_segment(x, y2, x_end, y2)
        x += dash + gap

    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        _draw_segment(x1, y, x1, y_end)
        _draw_segment(x2, y, x2, y_end)
        y += dash + gap


def _match_predictions(
    gt_boxes: Sequence[Tuple[float, float, float, float, int]],
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    pred_labels: Sequence[int],
    match_iou: float,
    match_class: bool,
) -> List[bool]:
    flags: List[bool] = []
    for (x1, y1, x2, y2), lab in zip(pred_boxes, pred_labels):
        best_iou = 0.0
        for gx1, gy1, gx2, gy2, glab in gt_boxes:
            if match_class and int(glab) != int(lab):
                continue
            iou = compute_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
        flags.append(best_iou >= float(match_iou))
    return flags


def _match_ground_truths(
    gt_boxes: Sequence[Tuple[float, float, float, float, int]],
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    pred_labels: Sequence[int],
    match_iou: float,
    match_class: bool,
) -> List[bool]:
    flags: List[bool] = []
    for gx1, gy1, gx2, gy2, glab in gt_boxes:
        best_iou = 0.0
        for (x1, y1, x2, y2), lab in zip(pred_boxes, pred_labels):
            if match_class and int(glab) != int(lab):
                continue
            iou = compute_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
        flags.append(best_iou >= float(match_iou))
    return flags


def render_panel(
    img: Image.Image,
    gt_boxes: Sequence[Tuple[float, float, float, float, int]],
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    pred_scores: Sequence[float],
    pred_labels: Sequence[int],
    id2label: Dict[int, str],
    title: str,
    pred_style: str,
    match_iou: float,
    match_class: bool,
    gt_style: str,
    show_labels: bool,
    fp_score_min: float,
):
    panel = img.convert("RGBA")
    overlay = None
    draw_fill = None
    if FILL_ALPHA > 0.0 or GT_FILL_ALPHA > 0.0:
        overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        draw_fill = ImageDraw.Draw(overlay)
    font = ImageFont.load_default() if show_labels else None

    tp_flags = _match_predictions(
        gt_boxes,
        pred_boxes,
        pred_labels,
        match_iou=match_iou,
        match_class=match_class,
    )
    gt_flags = _match_ground_truths(
        gt_boxes,
        pred_boxes,
        pred_labels,
        match_iou=match_iou,
        match_class=match_class,
    )

    if draw_fill is not None:
        for x1, y1, x2, y2, _lab in gt_boxes:
            draw_fill.rectangle([x1, y1, x2, y2], fill=_rgba(GT_COLOR, GT_FILL_ALPHA))
        for (x1, y1, x2, y2), _score, _lab, is_tp in zip(
            pred_boxes, pred_scores, pred_labels, tp_flags
        ):
            if pred_style == "tp_fp":
                color = TP_COLOR if is_tp else FP_COLOR
            else:
                color = PRED_COLOR
            draw_fill.rectangle([x1, y1, x2, y2], fill=_rgba(color, FILL_ALPHA))
        panel = Image.alpha_composite(panel, overlay)

    panel = panel.convert("RGB")
    draw = ImageDraw.Draw(panel)

    for (x1, y1, x2, y2, _lab), matched in zip(gt_boxes, gt_flags):
        if gt_style == "matched" and not matched:
            _draw_dashed_rect(draw, (x1, y1, x2, y2), GT_COLOR, width=GT_WIDTH)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=GT_WIDTH)

    for (x1, y1, x2, y2), score, lab, is_tp in zip(pred_boxes, pred_scores, pred_labels, tp_flags):
        if pred_style == "tp_fp":
            if not is_tp and score < float(fp_score_min):
                continue
            color = TP_COLOR if is_tp else FP_COLOR
            if is_tp:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=PRED_WIDTH)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=PRED_WIDTH)
        else:
            color = PRED_COLOR
            draw.rectangle([x1, y1, x2, y2], outline=color, width=PRED_WIDTH)
        if show_labels:
            label_name = id2label.get(int(lab), str(int(lab)))
            draw_text(draw, (x1 + 2, y1 + 2), f"{label_name} {score:.2f}", color, font, bg=(255, 255, 255))

    # keep title out of panel per request
    return panel


def build_grid(panels: List[List[Image.Image]], row_labels: List[str], gap: int = 8, row_label_pad: int = 110):
    rows = len(panels)
    cols = len(panels[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("No panels to render.")

    panel_w, panel_h = panels[0][0].size
    width = row_label_pad + cols * panel_w + (cols - 1) * gap
    height = rows * panel_h + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for r in range(rows):
        y = r * (panel_h + gap)
        draw_text(draw, (10, y + 10), row_labels[r], (0, 0, 0), font, bg=(255, 255, 255))
        for c in range(cols):
            x = row_label_pad + c * (panel_w + gap)
            canvas.paste(panels[r][c], (x, y))

    return canvas


def _safe_filename(text: str, default_ext: str = ".png") -> str:
    base, ext = os.path.splitext(text)
    cleaned = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    safe_base = "".join(cleaned).strip("_") or "panel"
    safe_ext = ext if ext else default_ext
    return f"{safe_base}{safe_ext}"


def main():
    parser = argparse.ArgumentParser(description="Qualitative A/B on CAD/Scan/C/D variants.")
    parser.add_argument("--model-a", required=True, help="Path to Model A (CAD-trained) checkpoint dir.")
    parser.add_argument("--model-b", required=True, help="Path to Model B (scan-trained) checkpoint dir.")
    parser.add_argument("--detector", default="rtdetr", help="Detector type: rtdetr|fasterrcnn|retinanet")
    parser.add_argument("--out-dir", default="./qualitative_abcd", help="Output directory.")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Prediction score threshold.")
    parser.add_argument("--topk", type=int, default=100, help="Max detections per panel.")
    parser.add_argument("--indices", default="", help="Comma-separated indices into eligible test layouts.")
    parser.add_argument("--max-layouts", type=int, default=3, help="Number of layouts to render.")
    parser.add_argument("--all", action="store_true", help="Render all eligible layouts.")
    parser.add_argument("--list-indices", action="store_true", help="List eligible layout indices and exit.")
    parser.add_argument("--list-max", type=int, default=200, help="Max indices to print when listing.")
    parser.add_argument("--square-only", action="store_true", help="Restrict to layouts with square CAD images.")
    parser.add_argument("--square-tol", type=float, default=0.0, help="Allow aspect ratio up to 1+tol when using --square-only.")
    parser.add_argument("--pred-style", choices=["all", "tp_fp"], default="tp_fp", help="Prediction styling: all|tp_fp.")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for TP/FP matching.")
    parser.add_argument("--match-class", action="store_true", help="Require class match for TP.")
    parser.add_argument("--gt-style", choices=["all", "matched"], default="all", help="GT styling: all|matched (dashed for misses).")
    parser.add_argument("--show-labels", action="store_true", help="Show class/score labels on predictions.")
    parser.add_argument("--fp-score-min", type=float, default=0.60, help="Minimum score to draw FP boxes when using tp_fp.")
    parser.add_argument("--cad", default="model_baked.png", help="CAD image filename.")
    parser.add_argument("--scan", default="F1_scaled.png", help="Scan image filename.")
    parser.add_argument("--variant-c", default="four_final_variants/03_scan_inside_boxes.png", help="C variant filename.")
    parser.add_argument("--variant-d", default="four_final_variants/04_svg_clean_plus_scan_outside.png", help="D variant filename.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    indiv_dir = os.path.join(args.out_dir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor()

    config_module.IMAGE_FILENAME = args.cad
    test_dirs = load_split_list(TEST_TXT)
    dataset = GraphRTDetrDataset(test_dirs, processor, "hierarchy_config.py", augment=False)

    sample_folders = []
    for img_path, graph_path, _ in dataset.samples:
        folder = os.path.dirname(img_path)
        sample_folders.append((folder, graph_path))

    variants = [
        ("CAD", args.cad),
        ("Scan", args.scan),
        ("C", args.variant_c),
        ("D", args.variant_d),
    ]

    eligible = []
    for folder, graph_path in sample_folders:
        ok = True
        for _, name in variants:
            if not os.path.exists(os.path.join(folder, name)):
                ok = False
                break
        if ok and args.square_only:
            cad_path = os.path.join(folder, args.cad)
            with Image.open(cad_path) as cad_img:
                w, h = cad_img.size
                if w == 0 or h == 0:
                    ok = False
                else:
                    ratio = max(w, h) / float(min(w, h))
                    if ratio > 1.0 + float(args.square_tol):
                        ok = False
        if ok:
            eligible.append((folder, graph_path))

    if not eligible:
        raise SystemExit("No eligible layouts found with all variants present.")

    if args.list_indices:
        limit = max(1, int(args.list_max))
        for i, (folder, _graph_path) in enumerate(eligible[:limit]):
            name = os.path.basename(folder.rstrip(os.sep)) or f"layout_{i}"
            print(f"{i}: {name}  ({folder})")
        if len(eligible) > limit:
            print(f"... {len(eligible) - limit} more (use --list-max to show more)")
        return

    indices = parse_indices(args.indices, len(eligible))
    if indices:
        chosen = [eligible[i] for i in indices]
    else:
        chosen = eligible if args.all else eligible[: args.max_layouts]

    detector_a = load_detector(args.detector, args.model_a, device)
    detector_b = load_detector(args.detector, args.model_b, device)

    for idx, (folder, graph_path) in enumerate(chosen):
        panels: List[List[Image.Image]] = [[], []]

        layout_name = os.path.basename(folder.rstrip(os.sep)) or f"layout_{idx}"
        base_img = Image.open(os.path.join(folder, variants[0][1])).convert("RGB")
        base_w, base_h = base_img.size

        for col_idx, (title, rel_name) in enumerate(variants):
            img_path = os.path.join(folder, rel_name)
            orig_img = Image.open(img_path).convert("RGB")
            gt_boxes = read_graph_boxes(
                graph_path,
                dataset.label2id,
                dataset.map_raw_to_l2,
                orig_img.size[0],
                orig_img.size[1],
            )

            if orig_img.size != (base_w, base_h):
                sx = base_w / float(orig_img.size[0])
                sy = base_h / float(orig_img.size[1])
                img = orig_img.resize((base_w, base_h), resample=Image.BILINEAR)
                gt_scaled = scale_boxes_xyxy(gt_boxes, sx, sy)
            else:
                sx = sy = 1.0
                img = orig_img
                gt_scaled = gt_boxes

            post_a = detector_predict_post(detector_a, dataset, orig_img, device, score_thresh=0.0)
            pred_boxes_a, pred_scores_a, pred_labels_a = filter_predictions(
                post_a, args.score_thresh, args.topk
            )
            if sx != 1.0 or sy != 1.0:
                pred_boxes_a = scale_pred_boxes(pred_boxes_a, sx, sy)

            panel_a = render_panel(
                img,
                gt_scaled,
                pred_boxes_a,
                pred_scores_a,
                pred_labels_a,
                dataset.id2label,
                title,
                pred_style=args.pred_style,
                match_iou=args.match_iou,
                match_class=args.match_class,
                gt_style=args.gt_style,
                show_labels=args.show_labels,
                fp_score_min=args.fp_score_min,
            )
            panels[0].append(panel_a)
            panel_a_name = f"{idx:02d}_{layout_name}_ModelA_{title}.png"
            panel_a.save(os.path.join(indiv_dir, _safe_filename(panel_a_name)))

            post_b = detector_predict_post(detector_b, dataset, orig_img, device, score_thresh=0.0)
            pred_boxes_b, pred_scores_b, pred_labels_b = filter_predictions(
                post_b, args.score_thresh, args.topk
            )
            if sx != 1.0 or sy != 1.0:
                pred_boxes_b = scale_pred_boxes(pred_boxes_b, sx, sy)

            panel_b = render_panel(
                img,
                gt_scaled,
                pred_boxes_b,
                pred_scores_b,
                pred_labels_b,
                dataset.id2label,
                title,
                pred_style=args.pred_style,
                match_iou=args.match_iou,
                match_class=args.match_class,
                gt_style=args.gt_style,
                show_labels=args.show_labels,
                fp_score_min=args.fp_score_min,
            )
            panels[1].append(panel_b)
            panel_b_name = f"{idx:02d}_{layout_name}_ModelB_{title}.png"
            panel_b.save(os.path.join(indiv_dir, _safe_filename(panel_b_name)))

        grid = build_grid(panels, row_labels=["Model A", "Model B"])
        out_path = os.path.join(args.out_dir, f"{idx:02d}_{layout_name}.png")
        grid.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
