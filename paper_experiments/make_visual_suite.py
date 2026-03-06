from utils.paper_io import figure_path
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as config_module  # noqa: E402
from paper_experiments.common import (  # noqa: E402
    DEFAULT_PER_CLASS_CAP,
    infer_predictions,
    iou_xyxy,
    load_gt_from_graph,
    load_image,
    load_label_maps,
    load_model,
    load_test_dirs,
    safe_makedirs,
)
from paper_experiments.factorized_degradation import DEGRADATIONS, LEVELS  # noqa: E402
from paper_experiments.make_paper_visuals import make_abcd_grid  # noqa: E402


Image.MAX_IMAGE_PIXELS = None

GT_COLOR = (0, 160, 0)
PRED_COLOR = (220, 50, 32)
TEXT_COLOR = (30, 30, 30)
BG_COLOR = (255, 255, 255)

DEFAULT_FONT_PATH = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"


def _load_font(path: Optional[str], size: int) -> ImageFont.ImageFont:
    if path and os.path.exists(path):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            return ImageFont.load_default()
    if os.path.exists(DEFAULT_FONT_PATH):
        try:
            return ImageFont.truetype(DEFAULT_FONT_PATH, size=size)
        except Exception:
            return ImageFont.load_default()
    return ImageFont.load_default()


@dataclass
class MatchExample:
    folder: str
    img_path: str
    gt_box: List[float]
    pred_box: Optional[List[float]]
    iou: float
    label: str
    area: float


def _letterbox(img: Image.Image, size: int) -> Tuple[Image.Image, float, Tuple[int, int]]:
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=BG_COLOR)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, (ox, oy)


def _draw_overlay(
    img: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Optional[Sequence[Sequence[float]]],
    pred_labels: Optional[Sequence[int]],
    id2label: Dict[int, str],
    width_gt: int = 2,
    width_pred: int = 2,
    show_labels: bool = False,
    font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = font or ImageFont.load_default()
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=width_gt)
    if pred_boxes is not None and pred_labels is not None:
        for box, lab in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box
            name = id2label.get(int(lab), str(lab))
            color = config_module.CLASS_COLORS.get(name, PRED_COLOR)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width_pred)
            if show_labels:
                draw.text((x1 + 2, y1 + 2), name, fill=color, font=font)
    return out


def _crop_for_boxes(
    img: Image.Image,
    boxes: Sequence[Sequence[float]],
    pad: float = 0.5,
    min_size: int = 0,
) -> Tuple[Image.Image, Tuple[int, int]]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    px = w * pad
    py = h * pad
    cx1 = max(int(x1 - px), 0)
    cy1 = max(int(y1 - py), 0)
    cx2 = min(int(x2 + px), img.size[0])
    cy2 = min(int(y2 + py), img.size[1])

    if min_size > 0:
        cur_w = cx2 - cx1
        cur_h = cy2 - cy1
        if cur_w < min_size:
            expand = int((min_size - cur_w) / 2)
            cx1 = max(cx1 - expand, 0)
            cx2 = min(cx2 + expand, img.size[0])
        if cur_h < min_size:
            expand = int((min_size - cur_h) / 2)
            cy1 = max(cy1 - expand, 0)
            cy2 = min(cy2 + expand, img.size[1])

    crop = img.crop((cx1, cy1, cx2, cy2))
    return crop, (cx1, cy1)


def _join_h(panels: Sequence[Image.Image], pad: int = 10) -> Image.Image:
    if not panels:
        return Image.new("RGB", (1, 1), color=BG_COLOR)
    widths = [p.width for p in panels]
    heights = [p.height for p in panels]
    out = Image.new("RGB", (sum(widths) + pad * (len(panels) - 1), max(heights)), color=BG_COLOR)
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + pad
    return out


def _join_v(panels: Sequence[Image.Image], pad: int = 10) -> Image.Image:
    if not panels:
        return Image.new("RGB", (1, 1), color=BG_COLOR)
    widths = [p.width for p in panels]
    heights = [p.height for p in panels]
    out = Image.new("RGB", (max(widths), sum(heights) + pad * (len(panels) - 1)), color=BG_COLOR)
    y = 0
    for panel in panels:
        out.paste(panel, (0, y))
        y += panel.height + pad
    return out


def _label_panel(
    panel: Image.Image,
    text: str,
    font: Optional[ImageFont.ImageFont] = None,
    pad_y: int = 6,
) -> Image.Image:
    if not text:
        return panel
    font = font or ImageFont.load_default()
    bbox = font.getbbox(text)
    label_h = (bbox[3] - bbox[1]) + pad_y * 2
    out = Image.new("RGB", (panel.width, panel.height + label_h), color=BG_COLOR)
    out.paste(panel, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, pad_y), text, fill=TEXT_COLOR, font=font)
    return out


def _save_output(img: Image.Image, out_path: str, dpi: int = 300):
    out_file = figure_path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        img.save(out_file, dpi=(dpi, dpi))
        return
    if ext == ".svg":
        import matplotlib.pyplot as plt  # local import

        w_in = max(img.width / float(dpi), 1e-6)
        h_in = max(img.height / float(dpi), 1e-6)
        fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(out_file, format="svg", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return
    img.save(out_file)


def _best_iou_for_gt(
    gt_box: Sequence[float],
    gt_label: int,
    pred_boxes: Sequence[Sequence[float]],
    pred_labels: Sequence[int],
) -> Tuple[float, Optional[List[float]]]:
    best_iou = 0.0
    best_box = None
    for box, lab in zip(pred_boxes, pred_labels):
        if int(lab) != int(gt_label):
            continue
        iou_val = iou_xyxy(list(gt_box), list(box))
        if iou_val > best_iou:
            best_iou = iou_val
            best_box = list(box)
    return best_iou, best_box


def _per_gt_best_iou(
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    pred_boxes: Sequence[Sequence[float]],
    pred_labels: Sequence[int],
) -> Tuple[List[float], List[Optional[List[float]]]]:
    best_iou = []
    best_boxes = []
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        iou_val, pred_box = _best_iou_for_gt(gt_box, gt_label, pred_boxes, pred_labels)
        best_iou.append(iou_val)
        best_boxes.append(pred_box)
    return best_iou, best_boxes


def _layout_stats(best_iou: Sequence[float]) -> Dict[str, float]:
    total = len(best_iou)
    if total == 0:
        return {"tight": 0.0, "loose": 0.0, "missed": 0.0, "mean_iou": 0.0}
    tight = sum(1 for v in best_iou if v >= 0.85) / total
    loose = sum(1 for v in best_iou if 0.5 <= v < 0.85) / total
    missed = sum(1 for v in best_iou if v < 0.5) / total
    matched = [v for v in best_iou if v >= 0.5]
    mean_iou = float(sum(matched) / max(len(matched), 1))
    return {"tight": tight, "loose": loose, "missed": missed, "mean_iou": mean_iou}


def _ensure_image(path: str) -> Image.Image:
    img = load_image(path)
    return img


def _iter_folders(test_dirs: List[str], limit: int) -> Iterable[str]:
    if limit and limit > 0:
        return test_dirs[:limit]
    return test_dirs


def _load_id_list(path: Optional[str]) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        return []
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            val = line.strip()
            if val:
                ids.append(val)
    return ids


def _filter_dirs_by_ids(test_dirs: List[str], exclude_ids: List[str]) -> List[str]:
    if not exclude_ids:
        return test_dirs
    exclude = set(exclude_ids)
    kept = []
    for d in test_dirs:
        layout_id = os.path.basename(d.rstrip("/"))
        if layout_id in exclude:
            continue
        kept.append(d)
    return kept


def _include_only_ids(test_dirs: List[str], include_ids: List[str]) -> List[str]:
    if not include_ids:
        return test_dirs
    include = set(include_ids)
    kept = []
    for d in test_dirs:
        layout_id = os.path.basename(d.rstrip("/"))
        if layout_id in include:
            kept.append(d)
    return kept


def _write_ids(path: Optional[str], ids: List[str]) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        safe_makedirs(out_dir)
    with open(path, "w", encoding="utf-8") as handle:
        for val in ids:
            handle.write(f"{val}\n")

def _infer_for_image(
    model,
    processor,
    img: Image.Image,
    device: torch.device,
    id2label: Dict[int, str],
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    boxes, scores, labels = infer_predictions(
        model,
        processor,
        img,
        device,
        id2label,
        score_thresh=score_thresh,
        topk_pre=topk_pre,
        final_k=final_k,
        per_class_cap=DEFAULT_PER_CLASS_CAP,
        use_per_class_thresh=use_per_class_thresh,
    )
    return boxes, labels


def make_paired_clean_scan(
    test_dirs: List[str],
    clean_name: str,
    scan_name: str,
    ckpt_clean: str,
    ckpt_scan: str,
    out_path: str,
    device: torch.device,
    panel_size: int,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    limit: int,
    num_rows: int,
    font: ImageFont.ImageFont,
    width_gt: int,
    width_pred: int,
    dpi: int,
    log_ids: Optional[str],
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor_clean, model_clean = load_model(ckpt_clean, device)
    processor_scan, model_scan = load_model(ckpt_scan, device)

    ranked = []
    for folder in _iter_folders(test_dirs, limit):
        clean_path = os.path.join(folder, clean_name)
        scan_path = os.path.join(folder, scan_name)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(clean_path) and os.path.exists(scan_path) and os.path.exists(graph_path)):
            continue

        clean_img = _ensure_image(clean_path)
        scan_img = _ensure_image(scan_path)
        if scan_img.size != clean_img.size:
            scan_img = scan_img.resize(clean_img.size, resample=Image.BILINEAR)

        gt_boxes, gt_labels, _ = load_gt_from_graph(
            graph_path, clean_img.size[0], clean_img.size[1], map_raw_to_l2, label2id
        )
        if not gt_boxes:
            continue

        clean_boxes, clean_labels = _infer_for_image(
            model_clean,
            processor_clean,
            clean_img,
            device,
            id2label,
            score_thresh,
            topk_pre,
            final_k,
            use_per_class_thresh,
        )
        scan_boxes, scan_labels = _infer_for_image(
            model_scan,
            processor_scan,
            scan_img,
            device,
            id2label,
            score_thresh,
            topk_pre,
            final_k,
            use_per_class_thresh,
        )

        clean_iou, _ = _per_gt_best_iou(gt_boxes, gt_labels, clean_boxes.tolist(), clean_labels.tolist())
        scan_iou, _ = _per_gt_best_iou(gt_boxes, gt_labels, scan_boxes.tolist(), scan_labels.tolist())

        clean_stats = _layout_stats(clean_iou)
        scan_stats = _layout_stats(scan_iou)
        score = (clean_stats["tight"] - scan_stats["tight"]) + 0.5 * (scan_stats["loose"] - clean_stats["loose"])
        ranked.append((score, folder, clean_boxes, clean_labels, scan_boxes, scan_labels, gt_boxes))

    ranked.sort(key=lambda r: r[0], reverse=True)
    selected = ranked[: max(1, num_rows)]

    rows = []
    used_ids: List[str] = []
    for _, folder, clean_boxes, clean_labels, scan_boxes, scan_labels, gt_boxes in selected:
        clean_img = _ensure_image(os.path.join(folder, clean_name))
        scan_img = _ensure_image(os.path.join(folder, scan_name))
        if scan_img.size != clean_img.size:
            scan_img = scan_img.resize(clean_img.size, resample=Image.BILINEAR)

        clean_overlay = _draw_overlay(
            clean_img,
            gt_boxes,
            clean_boxes,
            clean_labels,
            id2label,
            width_gt=width_gt,
            width_pred=width_pred,
            font=font,
        )
        scan_overlay = _draw_overlay(
            scan_img,
            gt_boxes,
            scan_boxes,
            scan_labels,
            id2label,
            width_gt=width_gt,
            width_pred=width_pred,
            font=font,
        )

        clean_panel, _, _ = _letterbox(clean_overlay, panel_size)
        scan_panel, _, _ = _letterbox(scan_overlay, panel_size)
        row = _join_h(
            [
                _label_panel(clean_panel, "CAD", font=font),
                _label_panel(scan_panel, "Scan", font=font),
            ],
            pad=10,
        )
        rows.append(row)
        used_ids.append(os.path.basename(folder.rstrip("/")))

    if not rows:
        raise RuntimeError("No paired rows generated.")
    grid = _join_v(rows, pad=12)
    _save_output(grid, out_path, dpi=dpi)
    _write_ids(log_ids, used_ids)


def make_iou_montage(
    test_dirs: List[str],
    scan_name: str,
    ckpt: str,
    out_path: str,
    device: torch.device,
    panel_size: int,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    limit: int,
    per_category: int,
    font: ImageFont.ImageFont,
    width_gt: int,
    width_pred: int,
    dpi: int,
    log_ids: Optional[str],
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(ckpt, device)

    tight = []
    loose = []
    missed = []

    for folder in _iter_folders(test_dirs, limit):
        img_path = os.path.join(folder, scan_name)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(img_path) and os.path.exists(graph_path)):
            continue

        img = _ensure_image(img_path)
        gt_boxes, gt_labels, _ = load_gt_from_graph(
            graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
        )
        if not gt_boxes:
            continue

        pred_boxes, pred_labels = _infer_for_image(
            model, processor, img, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
        )
        pred_boxes_list = pred_boxes.tolist()
        pred_labels_list = pred_labels.tolist()

        best_iou, best_boxes = _per_gt_best_iou(gt_boxes, gt_labels, pred_boxes_list, pred_labels_list)
        for gt, lab, iou_val, pred_box in zip(gt_boxes, gt_labels, best_iou, best_boxes):
            area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            entry = MatchExample(folder, img_path, list(gt), pred_box, iou_val, classes[lab], area)
            if iou_val >= 0.85:
                tight.append(entry)
            elif iou_val >= 0.5:
                loose.append(entry)
            else:
                missed.append(entry)

    def _pick_unique(
        entries: List[MatchExample],
        target: float,
        count: int,
        used: set,
        reverse: bool = False,
    ) -> List[MatchExample]:
        if not entries or count <= 0:
            return []
        entries.sort(key=lambda e: abs(e.iou - target), reverse=reverse)
        picked = []
        for entry in entries:
            layout_id = os.path.basename(entry.folder.rstrip("/"))
            if layout_id in used:
                continue
            picked.append(entry)
            used.add(layout_id)
            if len(picked) >= count:
                break
        return picked

    used_layouts: set = set()
    tight_pick = _pick_unique(tight, 0.9, per_category, used_layouts, reverse=True)
    loose_pick = _pick_unique(loose, 0.65, per_category, used_layouts)
    missed_pick = _pick_unique(missed, 0.1, per_category, used_layouts)

    def _render_entry(entry: MatchExample) -> Image.Image:
        img = _ensure_image(entry.img_path)
        boxes = [entry.gt_box]
        if entry.pred_box is not None:
            boxes.append(entry.pred_box)
        crop, (ox, oy) = _crop_for_boxes(img, boxes, pad=0.6)
        draw = ImageDraw.Draw(crop)

        def _rel(box):
            return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

        draw.rectangle(_rel(entry.gt_box), outline=GT_COLOR, width=width_gt)
        if entry.pred_box is not None:
            draw.rectangle(_rel(entry.pred_box), outline=PRED_COLOR, width=width_pred)
        draw.text((6, 6), f"{entry.label} IoU={entry.iou:.2f}", fill=TEXT_COLOR, font=font)
        panel, _, _ = _letterbox(crop, panel_size)
        return panel

    cols = []
    used_ids: List[str] = []
    for label, entries in [
        ("Tight (IoU ≥ 0.85)", tight_pick),
        ("Loose (0.50–0.85)", loose_pick),
        ("Missed (<0.50)", missed_pick),
    ]:
        panels = [_render_entry(e) for e in entries]
        col = _join_v(panels, pad=8)
        col = _label_panel(col, label, font=font)
        cols.append(col)
        for entry in entries:
            used_ids.append(os.path.basename(entry.folder.rstrip("/")))

    grid = _join_h(cols, pad=12)
    _save_output(grid, out_path, dpi=dpi)
    _write_ids(log_ids, used_ids)


def make_thin_failure_examples(
    test_dirs: List[str],
    clean_name: str,
    scan_name: str,
    ckpt: str,
    out_path: str,
    device: torch.device,
    panel_size: int,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    limit: int,
    area_max: float,
    font: ImageFont.ImageFont,
    width_gt: int,
    width_pred: int,
    dpi: int,
    min_crop: int,
    log_ids: Optional[str],
    crop_pad: float,
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(ckpt, device)

    needed = {"COLUMN": {"shifted": None, "missed": None}, "RAILING": {"shifted": None, "missed": None}}
    used_layouts: set = set()

    for pass_idx in (0, 1):
        for folder in _iter_folders(test_dirs, limit):
            clean_path = os.path.join(folder, clean_name)
            scan_path = os.path.join(folder, scan_name)
            graph_path = os.path.join(folder, "graph.json")
            if not (os.path.exists(clean_path) and os.path.exists(scan_path) and os.path.exists(graph_path)):
                continue

            layout_id = os.path.basename(folder.rstrip("/"))
            if pass_idx == 0 and layout_id in used_layouts:
                continue

            clean_img = _ensure_image(clean_path)
            scan_img = _ensure_image(scan_path)
            if scan_img.size != clean_img.size:
                scan_img = scan_img.resize(clean_img.size, resample=Image.BILINEAR)

            gt_boxes, gt_labels, _ = load_gt_from_graph(
                graph_path, clean_img.size[0], clean_img.size[1], map_raw_to_l2, label2id
            )
            if not gt_boxes:
                continue

            clean_boxes, clean_labels = _infer_for_image(
                model, processor, clean_img, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
            )
            scan_boxes, scan_labels = _infer_for_image(
                model, processor, scan_img, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
            )

            clean_iou, clean_best = _per_gt_best_iou(gt_boxes, gt_labels, clean_boxes.tolist(), clean_labels.tolist())
            scan_iou, scan_best = _per_gt_best_iou(gt_boxes, gt_labels, scan_boxes.tolist(), scan_labels.tolist())

            filled_any = False
            for gt, lab, ci, si, cb, sb in zip(gt_boxes, gt_labels, clean_iou, scan_iou, clean_best, scan_best):
                name = classes[lab]
                if name not in needed:
                    continue
                area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                if area > area_max:
                    continue
                if ci < 0.5:
                    continue
                if si >= 0.5 and si < 0.85 and needed[name]["shifted"] is None:
                    needed[name]["shifted"] = (folder, list(gt), cb, sb, ci, si, area)
                    filled_any = True
                if si < 0.5 and needed[name]["missed"] is None:
                    needed[name]["missed"] = (folder, list(gt), cb, sb, ci, si, area)
                    filled_any = True

            if filled_any:
                used_layouts.add(layout_id)

            if all(needed[c]["shifted"] and needed[c]["missed"] for c in needed):
                break
        if all(needed[c]["shifted"] and needed[c]["missed"] for c in needed):
            break

    rows = []
    used_ids: List[str] = []
    for cls_name, cases in needed.items():
        for case_name in ["shifted", "missed"]:
            entry = cases[case_name]
            if entry is None:
                continue
            folder, gt_box, clean_pred, scan_pred, ci, si, area = entry
            clean_img = _ensure_image(os.path.join(folder, clean_name))
            scan_img = _ensure_image(os.path.join(folder, scan_name))
            if scan_img.size != clean_img.size:
                scan_img = scan_img.resize(clean_img.size, resample=Image.BILINEAR)

            boxes = [gt_box]
            if clean_pred is not None:
                boxes.append(clean_pred)
            if scan_pred is not None:
                boxes.append(scan_pred)

            clean_crop, (ox, oy) = _crop_for_boxes(clean_img, boxes, pad=crop_pad, min_size=min_crop)
            scan_crop = scan_img.crop((ox, oy, ox + clean_crop.size[0], oy + clean_crop.size[1]))

            def _rel(box):
                return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

            draw = ImageDraw.Draw(clean_crop)
            draw.rectangle(_rel(gt_box), outline=GT_COLOR, width=width_gt)
            if clean_pred is not None:
                draw.rectangle(_rel(clean_pred), outline=PRED_COLOR, width=width_pred)

            draw = ImageDraw.Draw(scan_crop)
            draw.rectangle(_rel(gt_box), outline=GT_COLOR, width=width_gt)
            if scan_pred is not None:
                draw.rectangle(_rel(scan_pred), outline=PRED_COLOR, width=width_pred)

            clean_panel, _, _ = _letterbox(clean_crop, panel_size)
            scan_panel, _, _ = _letterbox(scan_crop, panel_size)
            label = f"{cls_name} {case_name} | area≈{int(area)}"
            row = _join_h(
                [
                    _label_panel(clean_panel, "CAD", font=font),
                    _label_panel(scan_panel, "Scan", font=font),
                ],
                pad=10,
            )
            row = _label_panel(row, label, font=font)
            rows.append(row)
            used_ids.append(os.path.basename(folder.rstrip("/")))

    if not rows:
        raise RuntimeError("No thin-structure examples found.")
    grid = _join_v(rows, pad=12)
    _save_output(grid, out_path, dpi=dpi)
    _write_ids(log_ids, used_ids)


def make_degradation_grid(
    test_dirs: List[str],
    layout_id: Optional[str],
    clean_name: str,
    ckpt: str,
    out_path: str,
    device: torch.device,
    panel_size: int,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    seed: int,
    font: ImageFont.ImageFont,
    width_gt: int,
    width_pred: int,
    dpi: int,
    log_ids: Optional[str],
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(ckpt, device)

    folder = None
    if layout_id:
        for d in test_dirs:
            if os.path.basename(d.rstrip("/")) == layout_id:
                folder = d
                break
    if folder is None:
        folder = test_dirs[0] if test_dirs else None
    if folder is None:
        raise RuntimeError("No test folders found.")

    img_path = os.path.join(folder, clean_name)
    graph_path = os.path.join(folder, "graph.json")
    if not (os.path.exists(img_path) and os.path.exists(graph_path)):
        raise FileNotFoundError("Missing image or graph.json for degradation grid.")

    img = _ensure_image(img_path)
    gt_boxes, gt_labels, _ = load_gt_from_graph(
        graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
    )

    rng = np.random.default_rng(seed)
    factors = ["blur", "thicken", "texture", "clutter"]
    levels = LEVELS

    rows = []
    for factor in factors:
        panels = []
        for level in ["clean"] + levels:
            if level == "clean":
                degraded = img
            else:
                degraded = DEGRADATIONS[factor](img, level, rng)
            boxes, labels = _infer_for_image(
                model, processor, degraded, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
            )
            overlay = _draw_overlay(
                degraded,
                gt_boxes,
                boxes,
                labels,
                id2label,
                width_gt=width_gt,
                width_pred=width_pred,
                font=font,
            )
            panel, _, _ = _letterbox(overlay, panel_size)
            panels.append(panel)
        row = _join_h(panels, pad=8)
        row = _label_panel(row, factor, font=font)
        rows.append(row)

    grid = _join_v(rows, pad=12)
    _save_output(grid, out_path, dpi=dpi)
    _write_ids(log_ids, [os.path.basename(folder.rstrip("/"))])


def make_mitigation_comparison(
    test_dirs: List[str],
    scan_name: str,
    ckpt_base: str,
    ckpt_mitig: str,
    out_path: str,
    device: torch.device,
    panel_size: int,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    limit: int,
    font: ImageFont.ImageFont,
    width_gt: int,
    width_pred: int,
    dpi: int,
    min_crop: int,
    log_ids: Optional[str],
    crop_pad: float,
):
    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    proc_base, model_base = load_model(ckpt_base, device)
    proc_mit, model_mit = load_model(ckpt_mitig, device)

    success = None
    fail = None

    for pass_idx in (0, 1):
        for folder in _iter_folders(test_dirs, limit):
            img_path = os.path.join(folder, scan_name)
            graph_path = os.path.join(folder, "graph.json")
            if not (os.path.exists(img_path) and os.path.exists(graph_path)):
                continue

            img = _ensure_image(img_path)
            gt_boxes, gt_labels, _ = load_gt_from_graph(
                graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
            )
            if not gt_boxes:
                continue

            base_boxes, base_labels = _infer_for_image(
                model_base, proc_base, img, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
            )
            mit_boxes, mit_labels = _infer_for_image(
                model_mit, proc_mit, img, device, id2label, score_thresh, topk_pre, final_k, use_per_class_thresh
            )

            base_iou, base_best = _per_gt_best_iou(gt_boxes, gt_labels, base_boxes.tolist(), base_labels.tolist())
            mit_iou, mit_best = _per_gt_best_iou(gt_boxes, gt_labels, mit_boxes.tolist(), mit_labels.tolist())

            for gt, lab, bi, mi, bb, mb in zip(gt_boxes, gt_labels, base_iou, mit_iou, base_best, mit_best):
                name = classes[lab]
                if name not in ("COLUMN", "RAILING"):
                    continue
                if success is None and bi < 0.5 and mi >= 0.5:
                    success = (folder, list(gt), bb, mb, name, bi, mi)
                if fail is None and bi < 0.5 and mi < 0.5:
                    if pass_idx == 0 and success is not None and folder == success[0]:
                        continue
                    fail = (folder, list(gt), bb, mb, name, bi, mi)
                if success and fail:
                    break
            if success and fail:
                break
        if success and fail:
            break

    rows = []
    used_ids: List[str] = []
    for tag, entry in [("Improves", success), ("Still fails", fail)]:
        if entry is None:
            continue
        folder, gt_box, base_pred, mit_pred, cls_name, bi, mi = entry
        img = _ensure_image(os.path.join(folder, scan_name))

        boxes = [gt_box]
        if base_pred is not None:
            boxes.append(base_pred)
        if mit_pred is not None:
            boxes.append(mit_pred)

        crop, (ox, oy) = _crop_for_boxes(img, boxes, pad=crop_pad, min_size=min_crop)
        base_crop = crop.copy()
        mit_crop = crop.copy()

        def _rel(box):
            return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

        draw = ImageDraw.Draw(base_crop)
        draw.rectangle(_rel(gt_box), outline=GT_COLOR, width=width_gt)
        if base_pred is not None:
            draw.rectangle(_rel(base_pred), outline=PRED_COLOR, width=width_pred)

        draw = ImageDraw.Draw(mit_crop)
        draw.rectangle(_rel(gt_box), outline=GT_COLOR, width=width_gt)
        if mit_pred is not None:
            draw.rectangle(_rel(mit_pred), outline=PRED_COLOR, width=width_pred)

        base_panel, _, _ = _letterbox(base_crop, panel_size)
        mit_panel, _, _ = _letterbox(mit_crop, panel_size)
        label = f"{tag} ({cls_name}) base IoU={bi:.2f} → mitig IoU={mi:.2f}"
        row = _join_h(
            [
                _label_panel(base_panel, "Baseline", font=font),
                _label_panel(mit_panel, "Mitigation", font=font),
            ],
            pad=10,
        )
        row = _label_panel(row, label, font=font)
        rows.append(row)
        used_ids.append(os.path.basename(folder.rstrip("/")))

    if not rows:
        raise RuntimeError("No mitigation examples found.")
    grid = _join_v(rows, pad=12)
    _save_output(grid, out_path, dpi=dpi)
    _write_ids(log_ids, used_ids)


def main():
    parser = argparse.ArgumentParser(description="Make qualitative figure panels for the paper.")
    parser.add_argument("--mode", default="paired", choices=["paired", "abcd", "iou", "thin", "degrade", "mitigation"])
    parser.add_argument("--out", default="paper_experiments/out/visual_suite.png", help="Output image path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=200, help="Limit number of layouts when auto-selecting.")
    parser.add_argument("--num-rows", type=int, default=4, help="Rows for paired clean/scan.")
    parser.add_argument("--per-category", type=int, default=3, help="Examples per category for IoU montage.")
    parser.add_argument("--area-max", type=float, default=1200.0, help="Max area (px) for thin examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle layout order using --seed.")
    parser.add_argument("--layout-id", default=None, help="Specific layout id for degradation grid.")
    parser.add_argument("--font-path", default=None, help="Path to a TTF/OTF font (LaTeX-like).")
    parser.add_argument("--font-size", type=int, default=18, help="Label font size.")
    parser.add_argument("--width-gt", type=int, default=3, help="GT box line width.")
    parser.add_argument("--width-pred", type=int, default=3, help="Pred box line width.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster and SVG embedding.")
    parser.add_argument("--min-crop", type=int, default=500, help="Minimum crop size for tiny objects (px).")
    parser.add_argument("--crop-pad", type=float, default=0.9, help="Crop padding ratio (lower = more zoom).")
    parser.add_argument("--exclude-ids", default=None, help="Text file with layout IDs to exclude.")
    parser.add_argument("--include-ids", default=None, help="Text file with layout IDs to include (restricts selection).")
    parser.add_argument("--log-ids", default=None, help="Write used layout IDs to this file.")

    parser.add_argument("--ckpt-cad", default=os.path.join(ROOT_DIR, "stable_runs/20260115_134958/exp1_clean/checkpoints/best"))
    parser.add_argument("--ckpt-scan", default=os.path.join(ROOT_DIR, "stable_runs/20260115_134958/exp2_scanned/checkpoints/best"))
    parser.add_argument("--ckpt-mitig", default=os.path.join(ROOT_DIR, "runs/mech_B_thin_20260128_140010/exp2_scanned/checkpoints/best"))

    parser.add_argument("--clean-name", default="four_final_variants/01_svg_clean.png")
    parser.add_argument("--scan-name", default="four_final_variants/02_scan_raw.png")
    parser.add_argument("--variant-c", default="four_final_variants/03_scan_inside_boxes.png")
    parser.add_argument("--variant-d", default="four_final_variants/04_svg_clean_plus_scan_outside.png")

    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--topk-pre", type=int, default=150)
    parser.add_argument("--final-k", type=int, default=100)
    parser.add_argument("--use-per-class-thresh", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    test_dirs = load_test_dirs()
    include_ids = _load_id_list(args.include_ids)
    if include_ids:
        test_dirs = _include_only_ids(test_dirs, include_ids)
    exclude_ids = _load_id_list(args.exclude_ids)
    if exclude_ids:
        test_dirs = _filter_dirs_by_ids(test_dirs, exclude_ids)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(test_dirs)
    font = _load_font(args.font_path, args.font_size)

    if args.mode == "paired":
        make_paired_clean_scan(
            test_dirs,
            args.clean_name,
            args.scan_name,
            args.ckpt_cad,
            args.ckpt_scan,
            args.out,
            device,
            args.panel_size,
            args.score_thresh,
            args.topk_pre,
            args.final_k,
            args.use_per_class_thresh,
            args.limit,
            args.num_rows,
            font,
            args.width_gt,
            args.width_pred,
            args.dpi,
            args.log_ids,
        )
        return

    if args.mode == "abcd":
        selected = []
        required = [args.clean_name, args.scan_name, args.variant_c, args.variant_d, "graph.json"]
        for folder in test_dirs:
            ok = True
            for rel in required:
                if not os.path.exists(os.path.join(folder, rel)):
                    ok = False
                    break
            if ok:
                selected.append(folder)
            if len(selected) >= args.num_rows:
                break
        out_path = args.out
        tmp_path = out_path
        if out_path.lower().endswith(".svg"):
            tmp_path = os.path.splitext(out_path)[0] + ".png"
        make_abcd_grid(
            folders=selected,
            clean_name=args.clean_name,
            scan_name=args.scan_name,
            variant_c=args.variant_c,
            variant_d=args.variant_d,
            out_path=tmp_path,
            panel_size=args.panel_size,
            ckpt_cad=args.ckpt_cad,
            ckpt_scan=None,
            device=device,
            score_thresh=args.score_thresh,
            final_k=args.final_k,
            draw_gt=True,
        )
        if out_path.lower().endswith(".svg"):
            try:
                img = Image.open(tmp_path).convert("RGB")
                _save_output(img, out_path, dpi=args.dpi)
                os.remove(tmp_path)
            except Exception:
                pass
        if args.log_ids:
            _write_ids(args.log_ids, [os.path.basename(d.rstrip("/")) for d in selected])
        return

    if args.mode == "iou":
        make_iou_montage(
            test_dirs,
            args.scan_name,
            args.ckpt_scan,
            args.out,
            device,
            args.panel_size,
            args.score_thresh,
            args.topk_pre,
            args.final_k,
            args.use_per_class_thresh,
            args.limit,
            args.per_category,
            font,
            args.width_gt,
            args.width_pred,
            args.dpi,
            args.log_ids,
        )
        return

    if args.mode == "thin":
        make_thin_failure_examples(
            test_dirs,
            args.clean_name,
            args.scan_name,
            args.ckpt_cad,
            args.out,
            device,
            args.panel_size,
            args.score_thresh,
            args.topk_pre,
            args.final_k,
            args.use_per_class_thresh,
            args.limit,
            args.area_max,
            font,
            args.width_gt,
            args.width_pred,
            args.dpi,
            args.min_crop,
            args.log_ids,
            args.crop_pad,
        )
        return

    if args.mode == "degrade":
        make_degradation_grid(
            test_dirs,
            args.layout_id,
            args.clean_name,
            args.ckpt_cad,
            args.out,
            device,
            args.panel_size,
            args.score_thresh,
            args.topk_pre,
            args.final_k,
            args.use_per_class_thresh,
            args.seed,
            font,
            args.width_gt,
            args.width_pred,
            args.dpi,
            args.log_ids,
        )
        return

    if args.mode == "mitigation":
        make_mitigation_comparison(
            test_dirs,
            args.scan_name,
            args.ckpt_scan,
            args.ckpt_mitig,
            args.out,
            device,
            args.panel_size,
            args.score_thresh,
            args.topk_pre,
            args.final_k,
            args.use_per_class_thresh,
            args.limit,
            font,
            args.width_gt,
            args.width_pred,
            args.dpi,
            args.min_crop,
            args.log_ids,
            args.crop_pad,
        )
        return


if __name__ == "__main__":
    main()
