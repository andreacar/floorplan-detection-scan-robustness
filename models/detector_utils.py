import json

import torch
from PIL import Image
from tqdm import tqdm

from data.coco_utils import rtdetr_to_coco_json


def _processed_image_size(enc):
    h, w = enc["pixel_values"].shape[-2:]
    if "pixel_mask" in enc:
        mask = enc["pixel_mask"].squeeze(0)
        valid_rows = (mask.sum(dim=1) > 0).sum().item()
        valid_cols = (mask.sum(dim=0) > 0).sum().item()
        if valid_rows > 0 and valid_cols > 0:
            h = int(valid_rows)
            w = int(valid_cols)
    return int(h), int(w)


def detector_predict_post(detector, dataset, img: Image.Image, device, score_thresh: float = 0.0):
    detector_type = getattr(detector, "detector_type", "rtdetr")
    w, h = img.size

    if detector_type not in ("fasterrcnn", "retinanet"):
        enc = dataset.image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(detector, "forward_eval") and hasattr(detector, "detector_type"):
                batch = {
                    "pixel_values": enc["pixel_values"],
                    "pixel_mask": enc.get("pixel_mask"),
                }
                out = detector.forward_eval(batch)
            else:
                out = detector(**enc)
        post = dataset.image_processor.post_process_object_detection(
            out,
            threshold=float(score_thresh),
            target_sizes=torch.tensor([[h, w]], device=device),
        )[0]
        return post

    enc = dataset.image_processor(images=img, return_tensors="pt")
    pixel_values = enc["pixel_values"].squeeze(0).to(device)
    with torch.no_grad():
        outputs = detector.forward_eval({"images": [pixel_values]})

    if not outputs:
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), device=device, dtype=torch.long),
        }

    pred = outputs[0]
    boxes = pred["boxes"]
    scores = pred["scores"]
    if detector_type == "fasterrcnn":
        labels = pred["labels"] - 1
    else:
        labels = pred["labels"]

    proc_h, proc_w = _processed_image_size(enc)
    if proc_h > 0 and proc_w > 0:
        scale_x = w / float(proc_w)
        scale_y = h / float(proc_h)
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    if hasattr(detector, "num_classes"):
        valid = (labels >= 0) & (labels < detector.num_classes)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    if score_thresh and scores.numel() > 0:
        keep = scores >= float(score_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    return {"boxes": boxes, "scores": scores, "labels": labels}


def detector_to_coco_json(detector, dataset, device, json_path: str) -> None:
    detector_type = getattr(detector, "detector_type", "rtdetr")
    if detector_type not in ("fasterrcnn", "retinanet"):
        model = detector.model if hasattr(detector, "model") else detector
        rtdetr_to_coco_json(model, dataset, device, json_path)
        return

    detector.eval()
    results = []
    for idx, (img_path, _, _) in enumerate(tqdm(dataset.samples, desc="COCO eval export")):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        post = detector_predict_post(detector, dataset, img, device, score_thresh=0.0)
        boxes = post["boxes"]
        scores = post["scores"]
        labels = post["labels"]

        if boxes.numel() == 0:
            continue

        b = boxes.clone()
        b[:, 0] = b[:, 0].clamp(0, w)
        b[:, 2] = b[:, 2].clamp(0, w)
        b[:, 1] = b[:, 1].clamp(0, h)
        b[:, 3] = b[:, 3].clamp(0, h)

        keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        b = b[keep]
        s = scores[keep]
        l = labels[keep]

        for box, score, lab in zip(b, s, l):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": idx,
                    "category_id": int(lab.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                }
            )

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
