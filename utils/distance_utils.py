import csv
import os
from pathlib import Path
from typing import Dict, Optional

import config as config_module


def _normalize_layout_path(path: str) -> str:
    if not path:
        return ""
    base_dir = config_module.BASE_DIR
    try:
        rel = os.path.relpath(path, base_dir)
    except ValueError:
        rel = path
    return os.path.normpath(rel)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def load_distance_scores(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    scores: Dict[str, float] = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    raw_vals: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            layout, value = row[0].strip(), row[1].strip()
            if not layout:
                continue
            try:
                score = float(value)
            except ValueError:
                continue
            key = _normalize_layout_path(
                layout if os.path.isabs(layout) else os.path.join(config_module.BASE_DIR, layout)
            )
            raw = max(0.0, score)
            scores[key] = raw
            raw_vals.append(raw)
    if not scores:
        return {}

    # Clip at percentile and normalize to keep weights in [1, 1+alpha].
    raw_vals.sort()
    clip_pct = getattr(config_module, "DISTANCE_SCORE_CLIP_PCT", 95.0)
    clip_val = _percentile(raw_vals, clip_pct)
    if clip_val <= 0:
        clip_val = max(raw_vals) if raw_vals else 0.0
    if clip_val <= 0:
        return scores
    for key, val in list(scores.items()):
        clipped = min(val, clip_val)
        scores[key] = max(0.0, min(1.0, clipped / clip_val))
    return scores


def layout_key_from_folder(folder: str) -> str:
    return _normalize_layout_path(folder)


def layout_key_from_image(image_path: str) -> str:
    if not image_path:
        return ""
    folder = os.path.dirname(image_path)
    return _normalize_layout_path(folder)
