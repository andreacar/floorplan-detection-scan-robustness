from utils.paper_io import figure_path
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as config_module
from utils.geometry import compute_iou
from paper_experiments.common import (
    DEFAULT_PER_CLASS_CAP,
    infer_predictions,
    load_gt_from_graph,
    load_image,
    load_label_maps,
    load_model,
    load_test_dirs,
    safe_makedirs,
)

try:
    from paper_experiments.factorized_degradation import DEGRADATIONS
except ImportError:
    DEGRADATIONS = {}


GT_COLOR = (0, 160, 0)
PRED_COLOR = (220, 50, 32)
TEXT_COLOR = (30, 30, 30)
BG_COLOR = (255, 255, 255)
ROW_LABEL_WIDTH = 90


def _parse_make_list(text: str) -> List[str]:
    if not text or text.strip().lower() == "all":
        return [
            "pairing",
            "invariance",
            "iou",
            "per_class",
            "size",
            "abcd",
            "taxonomy",
            "failure_cases",
        ]
    return [t.strip().lower() for t in text.split(",") if t.strip()]


def _resize_with_letterbox(
    img: Image.Image,
    size: int,
    fill: Tuple[int, int, int] = BG_COLOR,
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=fill)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, (ox, oy)


def _draw_boxes(
    img: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    id2label: Dict[int, str],
    color_map: Dict[str, Tuple[int, int, int]],
    width: int = 2,
    show_label: bool = False,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = box
        name = id2label.get(int(lab), str(lab))
        color = color_map.get(name, PRED_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if show_label:
            draw.text((x1 + 2, y1 + 2), name, fill=color, font=font)
    return out


def _draw_gt(
    img: Image.Image,
    boxes: Sequence[Sequence[float]],
    width: int = 2,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=width)
    return out


def _pick_folder(
    dirs: Iterable[str],
    required_files: Sequence[str],
    sample_id: Optional[str],
    sample_index: Optional[int],
) -> Optional[str]:
    candidates = []
    for d in dirs:
        ok = True
        for f in required_files:
            if not os.path.exists(os.path.join(d, f)):
                ok = False
                break
        if ok:
            candidates.append(d)

    if not candidates:
        return None

    if sample_id:
        for d in candidates:
            if os.path.basename(d.rstrip("/")) == sample_id:
                return d

    if sample_index is not None:
        if 0 <= sample_index < len(candidates):
            return candidates[sample_index]

    return candidates[0]


def _compose_row(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    cell_size: int,
    pad: int = 12,
    label_h: int = 28,
) -> Image.Image:
    cols = len(images)
    canvas_w = pad + cols * (cell_size + pad)
    canvas_h = label_h + pad + cell_size + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(images, labels)):
        x = pad + i * (cell_size + pad)
        y = label_h + pad
        canvas.paste(img, (x, y))
        draw.text((x + 4, 4), label, fill=TEXT_COLOR, font=font)
    return canvas


def _stack_rows(rows: Sequence[Image.Image], pad: int = 10) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), color=BG_COLOR)
    w = max(r.size[0] for r in rows)
    h = sum(r.size[1] for r in rows) + pad * (len(rows) + 1)
    canvas = Image.new("RGB", (w + pad * 2, h), color=BG_COLOR)
    y = pad
    for row in rows:
        canvas.paste(row, (pad, y))
        y += row.size[1] + pad
    return canvas


def make_pairing_schematic(out_path: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#f4f4f4",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
        )

    box(0.5, 4.2, 2.2, 1.1, "SVG layout\n+ annotations")
    box(3.1, 4.2, 1.8, 1.1, "Rasterize")
    box(5.4, 4.2, 2.1, 1.1, "CAD image")

    box(0.5, 1.2, 2.2, 1.1, "Same layout ID")
    box(3.1, 1.2, 1.8, 1.1, "Scanned raster")
    box(5.4, 1.2, 2.1, 1.1, "Scan image")

    box(7.9, 2.6, 2.0, 1.2, "C: scan-geometry\n(mask swap)")
    box(7.9, 0.2, 2.0, 1.2, "D: scan-background\n(region swap)")

    arrow(2.7, 4.75, 3.1, 4.75)
    arrow(4.9, 4.75, 5.4, 4.75)
    arrow(2.7, 1.75, 3.1, 1.75)
    arrow(4.9, 1.75, 5.4, 1.75)
    arrow(7.5, 4.2, 7.9, 3.2)
    arrow(7.5, 1.2, 7.9, 0.8)

    ax.text(
        0.6,
        3.6,
        "Pairing enforced at layout level",
        fontsize=9,
        color="#111111",
    )
    ax.text(0.6, 3.3, "Boxes unchanged", fontsize=9, color="#111111")

    fig.tight_layout()
    fig.savefig(figure_path(out_path), dpi=200)
    plt.close(fig)


def make_invariance_figure(
    folder: str,
    clean_name: str,
    scan_name: str,
    out_path: str,
):
    clean_path = os.path.join(folder, clean_name)
    scan_path = os.path.join(folder, scan_name)
    graph_path = os.path.join(folder, "graph.json")
    if not (os.path.exists(clean_path) and os.path.exists(scan_path) and os.path.exists(graph_path)):
        raise FileNotFoundError("Missing clean/scan image or graph.json.")

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    clean = load_image(clean_path)
    scan = load_image(scan_path)

    gt_boxes, gt_labels, _ = load_gt_from_graph(
        graph_path, clean.size[0], clean.size[1], map_raw_to_l2, label2id
    )
    if not gt_boxes:
        raise RuntimeError("No GT boxes found for invariance figure.")

    if scan.size != clean.size:
        scan = scan.resize(clean.size, resample=Image.BILINEAR)

    clean_gt = _draw_gt(clean, gt_boxes, width=3)
    scan_gt = _draw_gt(scan, gt_boxes, width=3)
    blend = Image.blend(clean, scan, alpha=0.5)
    blend_gt = _draw_gt(blend, gt_boxes, width=2)

    clean_gt = _add_title(clean_gt, "CAD + GT")
    scan_gt = _add_title(scan_gt, "Scan + same GT")
    blend_gt = _add_title(blend_gt, "Blend (alignment)")

    panel = _join_h([clean_gt, scan_gt, blend_gt], pad=10)
    panel.save(figure_path(out_path))


def _add_title(img: Image.Image, title: str, height: int = 28) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + height), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill=TEXT_COLOR, font=font)
    canvas.paste(img, (0, height))
    return canvas


def _join_h(images: Sequence[Image.Image], pad: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), color=BG_COLOR)
    w = sum(i.size[0] for i in images) + pad * (len(images) - 1)
    h = max(i.size[1] for i in images)
    canvas = Image.new("RGB", (w, h), color=BG_COLOR)
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.size[0] + pad
    return canvas


def _find_class_box(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    target_label: int,
) -> Optional[Tuple[float, float, float, float]]:
    best = None
    best_area = -1.0
    for box, lab in zip(boxes, labels):
        if int(lab) != int(target_label):
            continue
        x1, y1, x2, y2 = box
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def _crop_for_boxes(
    img: Image.Image,
    boxes: Sequence[Sequence[float]],
    pad: float = 0.4,
) -> Tuple[Image.Image, Tuple[float, float]]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    w = x2 - x1
    h = y2 - y1
    pad_px = max(w, h) * pad
    cx1 = max(0, int(x1 - pad_px))
    cy1 = max(0, int(y1 - pad_px))
    cx2 = min(img.size[0], int(x2 + pad_px))
    cy2 = min(img.size[1], int(y2 + pad_px))
    cropped = img.crop((cx1, cy1, cx2, cy2))
    return cropped, (cx1, cy1)


def _label_row_image(row_img: Image.Image, label: str, width: int = ROW_LABEL_WIDTH) -> Image.Image:
    canvas = Image.new("RGB", (row_img.width + width, row_img.height), color=BG_COLOR)
    canvas.paste(row_img, (width, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
    except AttributeError:
        w, h = font.getsize(label)
        bbox = (0, 0, w, h)
    text_height = bbox[3] - bbox[1]
    y = max(6, (row_img.height - text_height) // 2)
    draw.text((6, y), label, fill=TEXT_COLOR, font=font)
    return canvas


def _collect_dirs_with_files(
    folders: Sequence[str],
    required_files: Sequence[str],
) -> List[str]:
    out = []
    for folder in folders:
        if all(os.path.exists(os.path.join(folder, f)) for f in required_files):
            out.append(folder)
    return out


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    cleaned = cleaned.strip("_")
    return cleaned or "panel"


def _save_small_panels(
    panels: Sequence[Image.Image],
    row_label: str,
    column_labels: Sequence[str],
    dest_dir: str,
    iou_values: Sequence[float],
):
    os.makedirs(dest_dir, exist_ok=True)
    for i, (label, panel) in enumerate(zip(column_labels, panels)):
        iou_val = iou_values[i] if i < len(iou_values) else None
        iou_suffix = f"_IoU_{iou_val:.2f}" if iou_val is not None else ""
        name = f"{row_label}_{_sanitize_label(label)}{iou_suffix}.png"
        path = os.path.join(dest_dir, name)
        panel.save(path)


def _shift_box(box: Tuple[float, float, float, float], dx: float, dy: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _scale_box(box: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _find_pred_for_range(
    gt: Tuple[float, float, float, float],
    target_min: float,
    target_max: float,
) -> Tuple[Tuple[float, float, float, float], float]:
    w = gt[2] - gt[0]
    h = gt[3] - gt[1]
    scales = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    shifts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    for scale in scales:
        scaled = _scale_box(gt, scale)
        for sx in shifts:
            for sy in shifts:
                cand = _shift_box(scaled, w * sx, h * sy)
                iou = compute_iou(cand, gt)
                if target_min <= iou < target_max:
                    return cand, iou
    cand = _scale_box(gt, 1.2)
    return cand, compute_iou(cand, gt)


def make_iou_band_figure(
    folder: str,
    image_name: str,
    class_name: str,
    out_path: str,
):
    img_path = os.path.join(folder, image_name)
    graph_path = os.path.join(folder, "graph.json")
    if not (os.path.exists(img_path) and os.path.exists(graph_path)):
        raise FileNotFoundError("Missing image or graph.json for IoU band figure.")

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    if class_name not in label2id:
        raise ValueError(f"Unknown class: {class_name}")

    img = load_image(img_path)
    gt_boxes, gt_labels, _ = load_gt_from_graph(
        graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
    )
    gt = _find_class_box(gt_boxes, gt_labels, label2id[class_name])
    if gt is None:
        raise RuntimeError(f"No GT box found for class {class_name}.")

    missed = _shift_box(gt, (gt[2] - gt[0]) * 1.2, 0.0)
    missed_iou = compute_iou(missed, gt)
    loose, loose_iou = _find_pred_for_range(gt, 0.5, 0.85)
    tight, tight_iou = _find_pred_for_range(gt, 0.85, 1.0)

    crop, offset = _crop_for_boxes(img, [gt, missed, loose, tight], pad=0.6)
    ox, oy = offset

    def _rel(box):
        return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

    panels = []
    labels = ["missed (IoU < 0.50)", "loose (0.50-0.85)", "tight (>= 0.85)"]
    preds = [(missed, missed_iou), (loose, loose_iou), (tight, tight_iou)]

    for label, (pred, iou_val) in zip(labels, preds):
        panel = crop.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle(_rel(gt), outline=GT_COLOR, width=3)
        draw.rectangle(_rel(pred), outline=PRED_COLOR, width=3)
        draw.text((6, 6), f"{label}  IoU={iou_val:.2f}", fill=TEXT_COLOR, font=ImageFont.load_default())
        panels.append(panel)

    joined = _join_h(panels, pad=10)
    joined.save(figure_path(out_path))


def make_failure_cases_figure(
    folder: str,
    clean_name: str,
    scan_name: str,
    class_name: str,
    out_path: str,
    save_small_images: bool = True,
):
    clean_path = os.path.join(folder, clean_name)
    scan_path = os.path.join(folder, scan_name)
    graph_path = os.path.join(folder, "graph.json")
    if not (os.path.exists(clean_path) and os.path.exists(scan_path) and os.path.exists(graph_path)):
        raise FileNotFoundError("Missing CAD/scan image or graph.json for failure cases figure.")

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    if class_name not in label2id:
        raise ValueError(f"Unknown class: {class_name}")

    clean = load_image(clean_path)
    scan = load_image(scan_path)
    if scan.size != clean.size:
        scan = scan.resize(clean.size, resample=Image.BILINEAR)

    gt_boxes, gt_labels, _ = load_gt_from_graph(
        graph_path, clean.size[0], clean.size[1], map_raw_to_l2, label2id
    )
    gt = _find_class_box(gt_boxes, gt_labels, label2id[class_name])
    if gt is None:
        raise RuntimeError(f"No GT box found for class {class_name}.")

    categories = [
        ("Tight match", 0.85, 1.0, "IoU ≥ 0.85"),
        ("Loose match", 0.5, 0.85, "0.5 ≤ IoU < 0.85"),
        ("Missed detection", 0.0, 0.5, "IoU < 0.5"),
    ]
    column_labels = [
        "Tight match (IoU ≥ 0.85)",
        "Loose match (0.5 ≤ IoU < 0.85)",
        "Missed detection (IoU < 0.5)",
    ]

    cad_panels = []
    scan_panels = []
    cad_ious: List[float] = []

    for name, min_iou, max_iou, _ in categories:
        if "missed" in name.lower():
            pred = _shift_box(gt, (gt[2] - gt[0]) * 1.2, 0.0)
            iou_val = compute_iou(pred, gt)
        else:
            pred, iou_val = _find_pred_for_range(gt, min_iou, max_iou)

        boxes = [gt, pred]
        crop, (ox, oy) = _crop_for_boxes(clean, boxes, pad=0.6)
        window = (
            max(0, ox),
            max(0, oy),
            min(clean.size[0], ox + crop.size[0]),
            min(clean.size[1], oy + crop.size[1]),
        )

        def _rel(box):
            return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

        cad_panel = crop.copy()
        cad_draw = ImageDraw.Draw(cad_panel)
        cad_draw.rectangle(_rel(gt), outline=GT_COLOR, width=3)
        cad_draw.rectangle(_rel(pred), outline=PRED_COLOR, width=3)
        cad_panels.append(cad_panel)
        cad_ious.append(iou_val)

        scan_panel = scan.crop(window)
        scan_draw = ImageDraw.Draw(scan_panel)
        scan_draw.rectangle(_rel(gt), outline=GT_COLOR, width=3)
        scan_draw.rectangle(_rel(pred), outline=PRED_COLOR, width=3)
        scan_panels.append(scan_panel)

    if not cad_panels or not scan_panels:
        raise RuntimeError("Unable to build failure cases panels.")

    if save_small_images:
        sample_id = os.path.basename(os.path.normpath(folder))
        small_dir = os.path.join(os.path.dirname(out_path), "failure_cases_small", sample_id)
        _save_small_panels(cad_panels, "CAD", column_labels, os.path.join(small_dir, "cad"), cad_ious)

    all_panels = cad_panels + scan_panels
    cell_size = max(max(panel.width for panel in all_panels), max(panel.height for panel in all_panels))

    cad_row = _compose_row(cad_panels, column_labels, cell_size=cell_size, pad=10, label_h=32)
    scan_row = _compose_row(scan_panels, column_labels, cell_size=cell_size, pad=10, label_h=32)
    cad_row = _label_row_image(cad_row, "CAD")
    scan_row = _label_row_image(scan_row, "Scanned")

    grid = _stack_rows([cad_row, scan_row], pad=10)
    grid.save(figure_path(out_path))


def _load_per_class(path: str, key: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if key in data:
        return data[key]
    if "per_class_ap" in data:
        return data["per_class_ap"]
    if "per_class" in data:
        return data["per_class"]
    raise KeyError(f"Key {key} not found in {path}")


def make_per_class_degradation(
    clean_json: str,
    scan_json: str,
    out_path: str,
    key: str = "per_class_ap",
):
    import matplotlib.pyplot as plt

    classes, _, _, _ = load_label_maps()
    clean = _load_per_class(clean_json, key)
    scan = _load_per_class(scan_json, key)

    rows = []
    for cls in classes:
        if cls in clean and cls in scan:
            rows.append((cls, float(clean[cls]), float(scan[cls])))

    rows.sort(key=lambda r: (r[1] - r[2]), reverse=True)
    names = [r[0] for r in rows]
    clean_vals = np.array([r[1] for r in rows])
    scan_vals = np.array([r[2] for r in rows])

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hlines(y, scan_vals, clean_vals, color="#999999", lw=2)
    ax.scatter(clean_vals, y, color="#0072b2", label="clean")
    ax.scatter(scan_vals, y, color="#d55e00", label="scan")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("AP")
    ax.set_title("Per-class AP drop (clean to scan)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path(out_path), dpi=200)
    plt.close(fig)


def make_size_distribution(
    test_dirs: List[str],
    image_name: str,
    out_path: str,
    resize_fixed: bool,
    resize_h: int,
    resize_w: int,
):
    import matplotlib.pyplot as plt

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    per_class = {c: [] for c in classes}

    for folder in test_dirs:
        img_path = os.path.join(folder, image_name)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(img_path) and os.path.exists(graph_path)):
            continue
        img = load_image(img_path)
        gt_boxes, gt_labels, gt_areas = load_gt_from_graph(
            graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
        )
        if not gt_boxes:
            continue
        scale = 1.0
        if resize_fixed and img.size[0] > 0 and img.size[1] > 0:
            sx = resize_w / img.size[0]
            sy = resize_h / img.size[1]
            scale = sx * sy
        for area, lab in zip(gt_areas, gt_labels):
            cls = id2label[lab]
            per_class[cls].append(max(area * scale, 1e-6))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [np.log(v) for v in per_class.values() if v]
    labels = [k for k, v in per_class.items() if v]
    if not data:
        raise RuntimeError("No boxes found for size distribution.")
    ax.violinplot(data, showmeans=True, showmedians=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("log(area)")
    ax.set_title("Object size distribution per class")
    fig.tight_layout()
    fig.savefig(figure_path(out_path), dpi=200)
    plt.close(fig)


def _render_variant_panel(
    img: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    pred_boxes: Optional[np.ndarray],
    pred_labels: Optional[np.ndarray],
    id2label: Dict[int, str],
    panel_size: int,
    draw_gt: bool,
):
    canvas, scale, offset = _resize_with_letterbox(img, panel_size)
    ox, oy = offset

    scaled_gt = []
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        scaled_gt.append((x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy))

    draw = ImageDraw.Draw(canvas)
    if draw_gt:
        for box in scaled_gt:
            draw.rectangle(box, outline=GT_COLOR, width=2)

    if pred_boxes is not None and pred_labels is not None and pred_boxes.size > 0:
        for box, lab in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box
            x1 = x1 * scale + ox
            y1 = y1 * scale + oy
            x2 = x2 * scale + ox
            y2 = y2 * scale + oy
            name = id2label.get(int(lab), str(lab))
            color = config_module.CLASS_COLORS.get(name, PRED_COLOR)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return canvas


def make_abcd_grid(
    folders: Sequence[str],
    clean_name: str,
    scan_name: str,
    variant_c: str,
    variant_d: str,
    out_path: str,
    panel_size: int,
    ckpt_cad: Optional[str],
    ckpt_scan: Optional[str],
    device,
    score_thresh: float,
    final_k: int,
    draw_gt: bool,
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    per_class_cap = DEFAULT_PER_CLASS_CAP.copy()

    model_cad = None
    model_scan = None
    processor_cad = None
    processor_scan = None
    if ckpt_cad:
        processor_cad, model_cad = load_model(ckpt_cad, device)
    if ckpt_scan:
        processor_scan, model_scan = load_model(ckpt_scan, device)

    rows = []
    for folder in folders:
        graph_path = os.path.join(folder, "graph.json")
        if not os.path.exists(graph_path):
            continue

        img_paths = [
            os.path.join(folder, clean_name),
            os.path.join(folder, scan_name),
            os.path.join(folder, variant_c),
            os.path.join(folder, variant_d),
        ]
        if not all(os.path.exists(p) for p in img_paths):
            continue

        clean_img = load_image(img_paths[0])
        gt_boxes, gt_labels, _ = load_gt_from_graph(
            graph_path, clean_img.size[0], clean_img.size[1], map_raw_to_l2, label2id
        )
        if not gt_boxes:
            continue

        panels = []
        for img_path in img_paths:
            img = load_image(img_path)
            pred_boxes = None
            pred_labels = None
            if model_cad is not None and processor_cad is not None:
                boxes, scores, labels = infer_predictions(
                    model_cad,
                    processor_cad,
                    img,
                    device,
                    id2label,
                    score_thresh=score_thresh,
                    final_k=final_k,
                    per_class_cap=per_class_cap,
                )
                pred_boxes = boxes
                pred_labels = labels
            panel = _render_variant_panel(
                img,
                gt_boxes,
                gt_labels,
                pred_boxes,
                pred_labels,
                id2label,
                panel_size,
                draw_gt=draw_gt,
            )
            panels.append(panel)

        row = _compose_row(panels, ["CAD", "Scan", "C", "D"], panel_size)
        rows.append(row)

        if model_scan is not None and processor_scan is not None:
            panels = []
            for img_path in img_paths:
                img = load_image(img_path)
                boxes, scores, labels = infer_predictions(
                    model_scan,
                    processor_scan,
                    img,
                    device,
                    id2label,
                    score_thresh=score_thresh,
                    final_k=final_k,
                    per_class_cap=per_class_cap,
                )
                panel = _render_variant_panel(
                    img,
                    gt_boxes,
                    gt_labels,
                    boxes,
                    labels,
                    id2label,
                    panel_size,
                    draw_gt=draw_gt,
                )
                panels.append(panel)
            row = _compose_row(panels, ["CAD", "Scan", "C", "D"], panel_size)
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows generated for ABCD grid.")

    grid = _stack_rows(rows, pad=12)
    grid.save(figure_path(out_path))


def make_failure_taxonomy(
    folder: str,
    image_name: str,
    class_name: str,
    out_path: str,
):
    if not DEGRADATIONS:
        raise RuntimeError("Degradation functions are unavailable.")

    img_path = os.path.join(folder, image_name)
    graph_path = os.path.join(folder, "graph.json")
    if not (os.path.exists(img_path) and os.path.exists(graph_path)):
        raise FileNotFoundError("Missing image or graph.json for failure taxonomy.")

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    if class_name not in label2id:
        raise ValueError(f"Unknown class: {class_name}")

    img = load_image(img_path)
    gt_boxes, gt_labels, _ = load_gt_from_graph(
        graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
    )
    gt = _find_class_box(gt_boxes, gt_labels, label2id[class_name])
    if gt is None:
        raise RuntimeError(f"No GT box found for class {class_name}.")

    rows = []
    factors = [("blur", "missed"), ("clutter", "loose"), ("thicken", "tight")]
    for factor, label in factors:
        if factor not in DEGRADATIONS:
            continue
        degraded = DEGRADATIONS[factor](img, "medium", np.random.default_rng(42))
        if label == "missed":
            pred = _shift_box(gt, (gt[2] - gt[0]) * 1.2, 0.0)
        elif label == "loose":
            pred, _ = _find_pred_for_range(gt, 0.5, 0.85)
        else:
            pred, _ = _find_pred_for_range(gt, 0.85, 1.0)

        crop, offset = _crop_for_boxes(degraded, [gt, pred], pad=0.6)
        ox, oy = offset
        rel_gt = (gt[0] - ox, gt[1] - oy, gt[2] - ox, gt[3] - oy)
        rel_pred = (pred[0] - ox, pred[1] - oy, pred[2] - ox, pred[3] - oy)

        panel = crop.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle(rel_gt, outline=GT_COLOR, width=3)
        draw.rectangle(rel_pred, outline=PRED_COLOR, width=3)
        panel = _add_title(panel, f"{factor}: {label}")
        rows.append(panel)

    if not rows:
        raise RuntimeError("No taxonomy rows generated.")
    grid = _stack_rows(rows, pad=10)
    grid.save(figure_path(out_path))


def main():
    parser = argparse.ArgumentParser(description="Generate paper visuals for CAD/scan experiments.")
    parser.add_argument("--out-dir", default="paper_experiments/out/paper_visuals", help="Output directory.")
    parser.add_argument("--make", default="all", help="Comma list: pairing,invariance,iou,failure_cases,per_class,size,abcd,taxonomy.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--sample-id", default=None, help="Pick sample by folder name.")
    parser.add_argument("--sample-index", type=int, default=None, help="Pick sample by index from split list.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of layouts for ABCD grid.")
    parser.add_argument("--failure-count", type=int, default=1, help="Number of failure-case samples to render.")
    parser.add_argument("--clean-name", default="model_baked.png", help="CAD image filename.")
    parser.add_argument("--scan-name", default="F1_scaled.png", help="Scan image filename.")
    parser.add_argument("--variant-c", default="four_final_variants/03_scan_inside_boxes.png", help="C variant filename.")
    parser.add_argument("--variant-d", default="four_final_variants/04_svg_clean_plus_scan_outside.png", help="D variant filename.")
    parser.add_argument("--class-name", default="DOOR", help="Class to use for IoU examples.")
    parser.add_argument("--panel-size", type=int, default=360, help="Panel size for ABCD grid.")
    parser.add_argument("--ckpt-cad", default=None, help="Checkpoint dir for CAD-trained model (optional).")
    parser.add_argument("--ckpt-scan", default=None, help="Checkpoint dir for scan-trained model (optional).")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if unset).")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold for predictions.")
    parser.add_argument("--final-k", type=int, default=100, help="Max detections per image.")
    parser.add_argument("--no-gt", action="store_true", help="Do not draw GT boxes on ABCD grid.")
    parser.add_argument("--per-class-clean", default=None, help="JSON with per_class_ap for clean.")
    parser.add_argument("--per-class-scan", default=None, help="JSON with per_class_ap for scan.")
    parser.add_argument("--ap-key", default="per_class_ap", help="Key to read per-class AP.")

    args = parser.parse_args()
    tasks = _parse_make_list(args.make)
    # Note: visuals are always written under the artifacts figures root via `figure_path`.
    # Create the requested output directory there so the printed paths match what gets saved.
    figure_path(args.out_dir).mkdir(parents=True, exist_ok=True)

    test_dirs = load_test_dirs(args.test_txt)
    if not test_dirs:
        print("No test dirs found. Check TEST_TXT or --test-txt.")
        return

    required_invariance = [args.clean_name, args.scan_name, "graph.json"]
    required_iou = [args.clean_name, "graph.json"]
    required_abcd = [args.clean_name, args.scan_name, args.variant_c, args.variant_d, "graph.json"]

    folder_invariance = _pick_folder(test_dirs, required_invariance, args.sample_id, args.sample_index)
    folder_iou = _pick_folder(test_dirs, required_iou, args.sample_id, args.sample_index)
    abcd_folders = []
    for d in test_dirs:
        ok = True
        for f in required_abcd:
            if not os.path.exists(os.path.join(d, f)):
                ok = False
                break
        if ok:
            abcd_folders.append(d)
        if len(abcd_folders) >= args.sample_count:
            break

    required_failure = [args.clean_name, args.scan_name, "graph.json"]
    failure_candidates = _collect_dirs_with_files(test_dirs, required_failure)
    if args.sample_id:
        for idx, folder in enumerate(failure_candidates):
            if os.path.basename(folder.rstrip(os.sep)) == args.sample_id:
                failure_candidates.insert(0, failure_candidates.pop(idx))
                break
    if args.sample_index is not None:
        if 0 <= args.sample_index < len(failure_candidates):
            failure_candidates = failure_candidates[args.sample_index:]
        else:
            failure_candidates = []
    failure_count = max(0, args.failure_count)
    failure_candidates = failure_candidates[:failure_count]

    if "pairing" in tasks:
        out_path = "fig_pairing_schematic.png"
        make_pairing_schematic(out_path)
        print(f"[OK] pairing schematic -> {figure_path(out_path)}")

    if "invariance" in tasks:
        if folder_invariance:
            out_path = "fig_annotation_invariance.png"
            make_invariance_figure(folder_invariance, args.clean_name, args.scan_name, out_path)
            print(f"[OK] annotation invariance -> {figure_path(out_path)}")
        else:
            print("[SKIP] annotation invariance: no matching sample found.")

    if "iou" in tasks:
        if folder_iou:
            out_path = "fig_iou_bands.png"
            make_iou_band_figure(folder_iou, args.clean_name, args.class_name, out_path)
            print(f"[OK] IoU bands -> {figure_path(out_path)}")
        else:
            print("[SKIP] IoU bands: no matching sample found.")

    if "failure_cases" in tasks:
        if failure_candidates and failure_count > 0:
            for idx, folder in enumerate(failure_candidates):
                sample_id = os.path.basename(os.path.normpath(folder))
                out_name = f"fig_failure_cases_{sample_id}.png"
                out_path = os.path.join(args.out_dir, out_name)
                make_failure_cases_figure(
                    folder, args.clean_name, args.scan_name, args.class_name, out_path
                )
                print(f"[OK] failure cases -> {figure_path(out_path)}")
        else:
            print("[SKIP] failure cases: no matching sample found.")

    if "per_class" in tasks:
        if args.per_class_clean and args.per_class_scan:
            out_path = "fig_per_class_drop.png"
            make_per_class_degradation(args.per_class_clean, args.per_class_scan, out_path, key=args.ap_key)
            print(f"[OK] per-class drop -> {figure_path(out_path)}")
        else:
            print("[SKIP] per-class drop: provide --per-class-clean and --per-class-scan.")

    if "size" in tasks:
        out_path = "fig_size_distribution.png"
        make_size_distribution(
            test_dirs,
            args.clean_name,
            out_path,
            resize_fixed=config_module.RESIZE_FIXED,
            resize_h=config_module.RESIZE_HEIGHT,
            resize_w=config_module.RESIZE_WIDTH,
        )
        print(f"[OK] size distribution -> {figure_path(out_path)}")

    if "abcd" in tasks:
        if abcd_folders:
            out_path = "fig_abcd_grid.png"
            device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_name)
            make_abcd_grid(
                abcd_folders,
                args.clean_name,
                args.scan_name,
                args.variant_c,
                args.variant_d,
                out_path,
                panel_size=args.panel_size,
                ckpt_cad=args.ckpt_cad,
                ckpt_scan=args.ckpt_scan,
                device=device,
                score_thresh=args.score_thresh,
                final_k=args.final_k,
                draw_gt=not args.no_gt,
            )
            print(f"[OK] ABCD grid -> {figure_path(out_path)}")
        else:
            print("[SKIP] ABCD grid: no matching folders with all variants.")

    if "taxonomy" in tasks:
        if folder_iou:
            out_path = "fig_failure_taxonomy.png"
            make_failure_taxonomy(folder_iou, args.clean_name, args.class_name, out_path)
            print(f"[OK] failure taxonomy -> {figure_path(out_path)}")
        else:
            print("[SKIP] failure taxonomy: no matching sample found.")


if __name__ == "__main__":
    main()
