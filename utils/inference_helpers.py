import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, RTDetrForObjectDetection

import hierarchy_config as hier


DEFAULT_IMAGE_NAME = "F1_scaled.png"
DEFAULT_CLASS_COLORS = {
    "WALL": (228, 26, 28),
    "COLUMN": (55, 126, 184),
    "STAIR": (77, 175, 74),
    "RAILING": (152, 78, 163),
    "DOOR": (255, 127, 0),
    "WINDOW": (166, 86, 40),
}


def load_model_and_processor(checkpoint_dir: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir, use_fast=False)
    model = RTDetrForObjectDetection.from_pretrained(checkpoint_dir).to(device).eval()
    return processor, model


def resolve_input_image(input_path: str, image_name: str = DEFAULT_IMAGE_NAME) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / image_name
        if not candidate.exists():
            raise FileNotFoundError(f"Expected image '{image_name}' inside drawing folder: {path}")
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return path


def resolve_split_drawing(base_dir: str, split_entry: str) -> Path:
    entry = split_entry.strip()
    if not entry:
        raise ValueError("Empty split entry.")
    path = Path(entry)
    if path.is_absolute():
        return path
    return Path(base_dir).expanduser().resolve() / entry.lstrip("/")


@torch.no_grad()
def infer_image(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    score_thresh: float,
    topk: int,
) -> List[dict]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    encoded = processor(images=rgb, return_tensors="pt").to(device)
    outputs = model(**encoded)
    post = processor.post_process_object_detection(
        outputs,
        threshold=float(score_thresh),
        target_sizes=torch.tensor([[height, width]], device=device),
    )[0]

    scores = post["scores"].detach().cpu()
    labels = post["labels"].detach().cpu()
    boxes = post["boxes"].detach().cpu()
    order = torch.argsort(scores, descending=True)

    predictions: List[dict] = []
    for rank, idx in enumerate(order.tolist()):
        if topk > 0 and rank >= topk:
            break
        label_idx = int(labels[idx].item())
        label_name = hier.LEVEL2_CLASSES[label_idx] if label_idx < len(hier.LEVEL2_CLASSES) else str(label_idx)
        x1, y1, x2, y2 = boxes[idx].tolist()
        predictions.append(
            {
                "rank": rank + 1,
                "label_id": label_idx,
                "label": label_name,
                "score": float(scores[idx].item()),
                "box_xyxy": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
            }
        )
    return predictions


def draw_predictions(image: Image.Image, predictions: Iterable[dict], show_labels: bool = True) -> Image.Image:
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    for pred in predictions:
        x1, y1, x2, y2 = pred["box_xyxy"]
        label = pred["label"]
        score = pred["score"]
        color = DEFAULT_CLASS_COLORS.get(label, (255, 0, 0))
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if show_labels:
            drawer.text((x1 + 3, y1 + 3), f"{label} {score:.2f}", fill=color)
    return canvas


def save_prediction_bundle(
    out_dir: str,
    stem: str,
    source_image: str,
    checkpoint_dir: str,
    score_thresh: float,
    topk: int,
    predictions: List[dict],
    overlay: Image.Image,
) -> Tuple[Path, Path]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_out = output_dir / f"{stem}_predictions.png"
    json_out = output_dir / f"{stem}_predictions.json"
    overlay.save(image_out)
    with json_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_image": source_image,
                "checkpoint_dir": checkpoint_dir,
                "score_threshold": score_thresh,
                "topk": topk,
                "num_predictions": len(predictions),
                "predictions": predictions,
            },
            handle,
            indent=2,
        )
    return image_out, json_out
