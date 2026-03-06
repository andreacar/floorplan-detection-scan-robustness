import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrForObjectDetection

import config as config_module
import hierarchy_config as hier
from utils.geometry import clamp_bbox_xywh
from utils.paths import load_split_list


DEFAULT_PER_CLASS_CAP = {
    "WALL": 60,
    "DOOR": 30,
    "WINDOW": 30,
    "COLUMN": 20,
    "STAIR": 20,
    "RAILING": 20,
}

PER_CLASS_THRESH = {
    "WALL": 0.011,
    "COLUMN": 0.031,
    "STAIR": 0.040,
    "RAILING": 0.036,
    "DOOR": 0.015,
    "WINDOW": 0.043,
}


def load_test_dirs(test_txt: Optional[str] = None) -> List[str]:
    path = test_txt or config_module.TEST_TXT
    return load_split_list(path)


def load_label_maps():
    level2, _ = hier.load_level2_classes_and_mapping(None)
    label2id = {c: i for i, c in enumerate(level2)}
    id2label = {i: c for c, i in label2id.items()}
    return level2, label2id, id2label, hier.map_raw_to_l2


def iter_image_folders(
    folders: Iterable[str],
    image_name: str,
    require_graph: bool = True,
):
    for folder in folders:
        img_path = os.path.join(folder, image_name)
        graph_path = os.path.join(folder, "graph.json")
        if not os.path.exists(img_path):
            continue
        if require_graph and (not os.path.exists(graph_path)):
            continue
        yield folder, img_path, graph_path


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_gt_from_graph(
    graph_path: str,
    img_w: int,
    img_h: int,
    map_raw_to_l2,
    label2id: Dict[str, int],
) -> Tuple[List[List[float]], List[int], List[float]]:
    with open(graph_path, "r") as f:
        graph = json.load(f)

    boxes = []
    labels = []
    areas = []

    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        clamped = clamp_bbox_xywh(bbox, img_w, img_h)
        if clamped is None:
            continue
        x, y, bw, bh = clamped

        raw = node.get("data_class", "") or node.get("category", "")
        l2 = map_raw_to_l2(str(raw).strip())
        if l2 not in label2id:
            continue

        boxes.append([x, y, x + bw, y + bh])
        labels.append(label2id[l2])
        areas.append(float(bw * bh))

    return boxes, labels, areas


def load_model(ckpt_path: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = RTDetrForObjectDetection.from_pretrained(ckpt_path).to(device)
    model.eval()
    return processor, model


def _clip_and_filter_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    w: int,
    h: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    b = boxes.clone()
    b[:, 0] = b[:, 0].clamp(0, w)
    b[:, 2] = b[:, 2].clamp(0, w)
    b[:, 1] = b[:, 1].clamp(0, h)
    b[:, 3] = b[:, 3].clamp(0, h)

    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep], scores[keep], labels[keep]


def apply_policy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    id2label: Dict[int, str],
    score_thresh: float = 0.0,
    topk_pre: int = 0,
    final_k: int = 0,
    per_class_cap: Optional[Dict[str, int]] = None,
    default_cap: int = 200,
    use_per_class_thresh: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    if score_thresh > 0.0 or use_per_class_thresh:
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, (lab, sc) in enumerate(zip(labels, scores)):
            name = id2label.get(int(lab.item()), str(int(lab.item())))
            thr = score_thresh
            if use_per_class_thresh:
                thr = PER_CLASS_THRESH.get(name, score_thresh)
            if sc >= thr:
                keep_mask[i] = True
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]
        if boxes.numel() == 0:
            return boxes, scores, labels

    if topk_pre and scores.numel() > int(topk_pre):
        keep = torch.argsort(scores, descending=True)[: int(topk_pre)]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    if final_k or per_class_cap:
        per_class_cap = per_class_cap or {}
        order = torch.argsort(scores, descending=True)
        kept_idx = []
        counts: Dict[str, int] = {}
        for idx in order.tolist():
            lab = int(labels[idx].item())
            name = id2label.get(lab, str(lab))
            cap = int(per_class_cap.get(name, default_cap))
            counts.setdefault(name, 0)
            if counts[name] < cap:
                kept_idx.append(idx)
                counts[name] += 1
            if final_k and len(kept_idx) >= int(final_k):
                break

        if not kept_idx:
            return boxes[:0], scores[:0], labels[:0]
        kept = torch.tensor(kept_idx, dtype=torch.long, device=boxes.device)
        return boxes[kept], scores[kept], labels[kept]

    return boxes, scores, labels


def infer_predictions(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    id2label: Dict[int, str],
    score_thresh: float = 0.0,
    topk_pre: int = 0,
    final_k: int = 0,
    per_class_cap: Optional[Dict[str, int]] = None,
    default_cap: int = 200,
    use_per_class_thresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = image.size
    enc = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
    post = processor.post_process_object_detection(
        outputs,
        threshold=0.0,
        target_sizes=torch.tensor([[h, w]], device=device),
    )[0]

    boxes = post["boxes"].detach().cpu()
    scores = post["scores"].detach().cpu()
    labels = post["labels"].detach().cpu()

    boxes, scores, labels = _clip_and_filter_xyxy(boxes, scores, labels, w, h)
    boxes, scores, labels = apply_policy(
        boxes,
        scores,
        labels,
        id2label=id2label,
        score_thresh=score_thresh,
        topk_pre=topk_pre,
        final_k=final_k,
        per_class_cap=per_class_cap,
        default_cap=default_cap,
        use_per_class_thresh=use_per_class_thresh,
    )

    return boxes.numpy(), scores.numpy(), labels.numpy()


def _select_feature_map(backbone_out):
    tensors = []

    def _collect(obj):
        if torch.is_tensor(obj):
            tensors.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                _collect(v)
            return
        if hasattr(obj, "tensors") and torch.is_tensor(obj.tensors):
            tensors.append(obj.tensors)
            return
        if hasattr(obj, "tensor") and torch.is_tensor(obj.tensor):
            tensors.append(obj.tensor)
            return
        if hasattr(obj, "feature_maps"):
            _collect(obj.feature_maps)
            return
        if hasattr(obj, "features"):
            _collect(obj.features)
            return

    _collect(backbone_out)
    if not tensors:
        return backbone_out

    float_tensors = [t for t in tensors if torch.is_floating_point(t)]
    if float_tensors:
        candidates = [t for t in float_tensors if t.ndim >= 3]
        return candidates[0] if candidates else float_tensors[0]

    return tensors[0].float()


def extract_embedding(model, processor, image: Image.Image, device: torch.device) -> np.ndarray:
    enc = processor(images=image, return_tensors="pt").to(device)
    pixel_values = enc["pixel_values"]
    pixel_mask = enc.get("pixel_mask")
    if pixel_mask is None:
        pixel_mask = torch.ones(pixel_values.shape[-2:], device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        feats = None
        if hasattr(model, "model") and hasattr(model.model, "backbone"):
            feats = _select_feature_map(model.model.backbone(pixel_values, pixel_mask))
            if torch.is_tensor(feats) and not torch.is_floating_point(feats):
                feats = None

        if feats is None:
            out = model.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(out, "encoder_last_hidden_state") and out.encoder_last_hidden_state is not None:
                feats = out.encoder_last_hidden_state
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                feats = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states:
                feats = out.hidden_states[-1]
            else:
                feats = _select_feature_map(out)

    if feats.ndim == 4:
        pooled = feats.mean(dim=(-2, -1))
    elif feats.ndim == 3:
        pooled = feats.mean(dim=1)
    else:
        pooled = feats

    return pooled.squeeze(0).detach().cpu().numpy()


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0.0 else 0.0


def match_greedy_by_class(
    gt_boxes: List[List[float]],
    gt_labels: List[int],
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    pred_labels: List[int],
    iou_thr: float,
) -> List[float]:
    matched_iou = [0.0 for _ in gt_boxes]
    if not gt_boxes or not pred_boxes:
        return matched_iou

    gt_by_class: Dict[int, List[int]] = {}
    for gi, lab in enumerate(gt_labels):
        gt_by_class.setdefault(int(lab), []).append(gi)

    pred_by_class: Dict[int, List[int]] = {}
    for pi, lab in enumerate(pred_labels):
        pred_by_class.setdefault(int(lab), []).append(pi)

    for cls_id, gt_idx in gt_by_class.items():
        pred_idx = pred_by_class.get(cls_id, [])
        if not pred_idx:
            continue

        pred_idx_sorted = sorted(pred_idx, key=lambda i: pred_scores[i], reverse=True)
        used_gt = set()
        for pi in pred_idx_sorted:
            best_iou = 0.0
            best_gi = None
            for gi in gt_idx:
                if gi in used_gt:
                    continue
                iou_val = iou_xyxy(pred_boxes[pi], gt_boxes[gi])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_gi is not None and best_iou >= iou_thr:
                matched_iou[best_gi] = best_iou
                used_gt.add(best_gi)

    return matched_iou


def compute_ap_for_image(
    gt_boxes: List[List[float]],
    gt_labels: List[int],
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    pred_labels: List[int],
    iou_thr: float,
    class_ids: Iterable[int],
) -> float:
    ap_values = []
    for cls_id in class_ids:
        gt_idx = [i for i, lab in enumerate(gt_labels) if int(lab) == int(cls_id)]
        if not gt_idx:
            continue
        pred_idx = [i for i, lab in enumerate(pred_labels) if int(lab) == int(cls_id)]
        if not pred_idx:
            ap_values.append(0.0)
            continue

        pred_idx_sorted = sorted(pred_idx, key=lambda i: pred_scores[i], reverse=True)
        matched = [False for _ in gt_idx]
        tp = []
        fp = []
        for pi in pred_idx_sorted:
            best_iou = 0.0
            best_g = None
            for gi_pos, gi in enumerate(gt_idx):
                if matched[gi_pos]:
                    continue
                iou_val = iou_xyxy(pred_boxes[pi], gt_boxes[gi])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_g = gi_pos
            if best_g is not None and best_iou >= iou_thr:
                matched[best_g] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(len(gt_idx), 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _ap_from_pr(rec, prec)
        ap_values.append(float(ap))

    if not ap_values:
        return 0.0
    return float(sum(ap_values) / len(ap_values))


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        rank_val = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = rank_val
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)
