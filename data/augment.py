import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps, ImageDraw

from config import (
    AUGMENT_CAD_FILENAMES,
    AUGMENT_APPLY_ALL,
    AUGMENT_SCAN_MIX_ENABLE,
    AUGMENT_SCAN_MIX_FILENAME,
    AUGMENT_SCAN_MIX_PROB,
    AUGMENT_SCAN_RAW_FILENAME,
    AUGMENT_STROKE_BRIGHTNESS_RANGE,
    AUGMENT_STROKE_CONTRAST_RANGE,
    AUGMENT_STROKE_DILATE_PROB,
    AUGMENT_STROKE_ENABLE,
    AUGMENT_STROKE_KERNEL_PROBS,
    AUGMENT_STROKE_KERNEL_SIZES,
    AUGMENT_STROKE_PROB,
    AUGMENT_STROKE_SIGMA_RANGE,
    AUGMENT_LINE_DROPOUT_ENABLE,
    AUGMENT_LINE_DROPOUT_PROB,
    AUGMENT_LINE_DROPOUT_COUNT_RANGE,
    AUGMENT_LINE_DROPOUT_WIDTH_RANGE,
    AUGMENT_DEPICTION_ENABLE,
    AUGMENT_DEPICTION_PROB,
    AUGMENT_DEPICTION_KERNEL_SIZES,
    AUGMENT_DEPICTION_KERNEL_PROBS,
    AUGMENT_DEPICTION_SIGMA_RANGE,
    AUGMENT_DEPICTION_THRESHOLD,
    DISTANCE_SCORE_FILE,
    DISTANCE_CURRICULUM_ENABLE,
    DISTANCE_CURRICULUM_THRESHOLDS,
    DISTANCE_CURRICULUM_LOW_KERNELS,
    DISTANCE_CURRICULUM_LOW_SIGMA,
    DISTANCE_CURRICULUM_MID_KERNELS,
    DISTANCE_CURRICULUM_MID_SIGMA,
    DISTANCE_CURRICULUM_HIGH_KERNELS,
    DISTANCE_CURRICULUM_HIGH_SIGMA,
)
from utils.distance_utils import layout_key_from_image, load_distance_scores

_LOG = logging.getLogger(__name__)
_WARNED_SCAN = False


def _image_basename(img: Image.Image) -> Optional[str]:
    path = getattr(img, "filename", None)
    if not path:
        return None
    return os.path.basename(path)


def _is_cad_image(img: Image.Image) -> bool:
    if AUGMENT_APPLY_ALL:
        return True
    if not AUGMENT_CAD_FILENAMES:
        return True
    name = _image_basename(img)
    if not name:
        return False
    return name in AUGMENT_CAD_FILENAMES


def _same_pad(kernel: int) -> tuple[int, int, int, int]:
    if kernel % 2 == 1:
        p = kernel // 2
        return (p, p, p, p)
    left = kernel // 2 - 1
    right = kernel // 2
    return (left, right, left, right)


def _morphology(gray_img: Image.Image, kernel: int, dilate: bool) -> Image.Image:
    if kernel <= 1:
        return gray_img

    arr = np.array(gray_img, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    pad = _same_pad(kernel)
    tensor = F.pad(tensor, pad, mode="replicate")
    if dilate:
        out = F.max_pool2d(tensor, kernel, stride=1)
    else:
        out = -F.max_pool2d(-tensor, kernel, stride=1)
    out = out.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(out, mode="L")


def _apply_stroke_augmentation(img: Image.Image, distance_map: dict[str, float]) -> Image.Image:
    gray = ImageOps.grayscale(img)

    score = distance_map.get(layout_key_from_image(getattr(img, "filename", "")), 0.0)
    kernel_choices = AUGMENT_STROKE_KERNEL_SIZES
    sigma_range = AUGMENT_STROKE_SIGMA_RANGE
    if DISTANCE_CURRICULUM_ENABLE and distance_map:
        bucket = _distance_bucket(score)
        if bucket == "low":
            kernel_choices = DISTANCE_CURRICULUM_LOW_KERNELS
            sigma_range = DISTANCE_CURRICULUM_LOW_SIGMA
        elif bucket == "mid":
            kernel_choices = DISTANCE_CURRICULUM_MID_KERNELS
            sigma_range = DISTANCE_CURRICULUM_MID_SIGMA
        else:
            kernel_choices = DISTANCE_CURRICULUM_HIGH_KERNELS
            sigma_range = DISTANCE_CURRICULUM_HIGH_SIGMA

    kernel = random.choices(kernel_choices, weights=_kernel_probs_for_choices(kernel_choices), k=1)[0]
    dilate = random.random() < AUGMENT_STROKE_DILATE_PROB
    gray = _morphology(gray, kernel, dilate)

    sigma = random.uniform(*sigma_range)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

    contrast = random.uniform(*AUGMENT_STROKE_CONTRAST_RANGE)
    brightness = random.uniform(*AUGMENT_STROKE_BRIGHTNESS_RANGE)
    arr = np.array(gray, dtype=np.float32)
    arr = arr * contrast + brightness
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    gray = Image.fromarray(arr, mode="L")

    return gray.convert("RGB")


def _apply_depiction_augmentation(img: Image.Image) -> Image.Image:
    if not AUGMENT_DEPICTION_KERNEL_SIZES:
        return img
    gray = ImageOps.grayscale(img)
    kernel = random.choices(
        AUGMENT_DEPICTION_KERNEL_SIZES,
        weights=AUGMENT_DEPICTION_KERNEL_PROBS
        if len(AUGMENT_DEPICTION_KERNEL_PROBS) == len(AUGMENT_DEPICTION_KERNEL_SIZES)
        else None,
        k=1,
    )[0]
    gray = _morphology(gray, kernel, dilate=True)

    sigma = random.uniform(*AUGMENT_DEPICTION_SIGMA_RANGE)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

    arr = np.array(gray, dtype=np.uint8)
    thresh = int(AUGMENT_DEPICTION_THRESHOLD)
    arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _apply_line_dropout(img: Image.Image) -> Image.Image:
    if not AUGMENT_LINE_DROPOUT_COUNT_RANGE or not AUGMENT_LINE_DROPOUT_WIDTH_RANGE:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    count = random.randint(*AUGMENT_LINE_DROPOUT_COUNT_RANGE)
    width = random.randint(*AUGMENT_LINE_DROPOUT_WIDTH_RANGE)
    w, h = out.size
    for _ in range(count):
        x0 = random.randint(0, max(w - 1, 0))
        y0 = random.randint(0, max(h - 1, 0))
        x1 = random.randint(0, max(w - 1, 0))
        y1 = random.randint(0, max(h - 1, 0))
        draw.line((x0, y0, x1, y1), fill=(255, 255, 255), width=width)
    return out


def _distance_bucket(score: float) -> str:
    low, mid = DISTANCE_CURRICULUM_THRESHOLDS
    if score < low:
        return "low"
    if score < mid:
        return "mid"
    return "high"


def _kernel_probs_for_choices(choices: list[int]) -> list[float]:
    if not choices:
        return [1.0]
    probs: list[float] = []
    for k in choices:
        if k in AUGMENT_STROKE_KERNEL_SIZES:
            idx = AUGMENT_STROKE_KERNEL_SIZES.index(k)
            probs.append(AUGMENT_STROKE_KERNEL_PROBS[idx])
        else:
            probs.append(1.0)
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _build_scan_mix(img: Image.Image, annotations: Optional[list[dict]]) -> Optional[Image.Image]:
    global _WARNED_SCAN
    path = getattr(img, "filename", None)
    if not path:
        return None

    base_dir = os.path.dirname(path)
    mix_path = os.path.join(base_dir, AUGMENT_SCAN_MIX_FILENAME)
    if os.path.exists(mix_path):
        mix_img = Image.open(mix_path).convert("RGB")
        if mix_img.size != img.size:
            mix_img = mix_img.resize(img.size, resample=Image.BILINEAR)
        return mix_img

    raw_path = os.path.join(base_dir, AUGMENT_SCAN_RAW_FILENAME)
    if not os.path.exists(raw_path):
        if not _WARNED_SCAN:
            _LOG.warning("Scan mix image not found: %s", mix_path)
            _WARNED_SCAN = True
        return None

    scan_img = Image.open(raw_path).convert("RGB")
    if scan_img.size != img.size:
        scan_img = scan_img.resize(img.size, resample=Image.BILINEAR)

    if not annotations:
        return scan_img

    w, h = img.size
    mask = np.zeros((h, w), dtype=bool)
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        x0 = max(int(round(x)), 0)
        y0 = max(int(round(y)), 0)
        x1 = min(int(round(x + bw)), w)
        y1 = min(int(round(y + bh)), h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True

    fg = np.array(img.convert("RGB"), dtype=np.uint8)
    bg = np.array(scan_img, dtype=np.uint8)
    out = bg.copy()
    out[mask] = fg[mask]
    return Image.fromarray(out, mode="RGB")


class TrainingAugmentProcessor:
    def __init__(self, base_processor: Any):
        self._base = base_processor
        self._distance_map = load_distance_scores(DISTANCE_SCORE_FILE)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, images=None, annotations=None, **kwargs):
        if images is None:
            return self._base(images=images, annotations=annotations, **kwargs)

        if (
            not AUGMENT_STROKE_ENABLE
            and not AUGMENT_SCAN_MIX_ENABLE
            and not AUGMENT_LINE_DROPOUT_ENABLE
            and not AUGMENT_DEPICTION_ENABLE
        ):
            return self._base(images=images, annotations=annotations, **kwargs)

        if isinstance(images, (list, tuple)):
            out_images = []
            anns = annotations if isinstance(annotations, (list, tuple)) else [annotations] * len(images)
            for img, ann in zip(images, anns):
                out_images.append(self._augment_one(img, ann))
            return self._base(images=out_images, annotations=annotations, **kwargs)

        return self._base(images=self._augment_one(images, annotations), annotations=annotations, **kwargs)

    def _augment_one(self, img: Image.Image, annotations: Optional[Dict[str, Any]]) -> Image.Image:
        if not _is_cad_image(img):
            return img

        scan_prob = AUGMENT_SCAN_MIX_PROB if AUGMENT_SCAN_MIX_ENABLE else 0.0
        stroke_prob = AUGMENT_STROKE_PROB if AUGMENT_STROKE_ENABLE else 0.0
        depiction_prob = AUGMENT_DEPICTION_PROB if AUGMENT_DEPICTION_ENABLE else 0.0
        line_prob = AUGMENT_LINE_DROPOUT_PROB if AUGMENT_LINE_DROPOUT_ENABLE else 0.0
        if scan_prob <= 0.0 and stroke_prob <= 0.0 and depiction_prob <= 0.0 and line_prob <= 0.0:
            return img

        r = random.random()
        if scan_prob > 0.0 and r < scan_prob:
            anns = annotations.get("annotations") if isinstance(annotations, dict) else None
            mixed = _build_scan_mix(img, anns)
            if mixed is not None:
                return mixed
            return img
        out = img
        if depiction_prob > 0.0 and random.random() < depiction_prob:
            out = _apply_depiction_augmentation(out)
        if stroke_prob > 0.0 and random.random() < stroke_prob:
            out = _apply_stroke_augmentation(out, self._distance_map)
        if line_prob > 0.0 and random.random() < line_prob:
            out = _apply_line_dropout(out)
        return out


def build_train_processor(base_processor: Any) -> Any:
    if not AUGMENT_STROKE_ENABLE and not AUGMENT_SCAN_MIX_ENABLE:
        return base_processor
    return TrainingAugmentProcessor(base_processor)
