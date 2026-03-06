import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from typing import List, Tuple
from datetime import datetime

# Single-file config mode:
# - Prefer env vars at runtime.
# - Otherwise, set DEFAULT_BASE_DIR below once and forget about YAML/env.
DEFAULT_BASE_DIR = ""

# Optional local overrides (gitignored): `configs/config_local.py`.
try:  # pragma: no cover
    from . import config_local as _config_local  # type: ignore
except Exception:  # pragma: no cover
    _config_local = None

_LOCAL_DEFAULT_BASE_DIR = ""
_LOCAL_BASE_DIR = ""
if _config_local is not None:  # pragma: no cover
    _LOCAL_DEFAULT_BASE_DIR = str(getattr(_config_local, "DEFAULT_BASE_DIR", "") or "")
    _LOCAL_BASE_DIR = str(getattr(_config_local, "BASE_DIR", "") or "")
    if _LOCAL_DEFAULT_BASE_DIR:
        DEFAULT_BASE_DIR = _LOCAL_DEFAULT_BASE_DIR

def _parse_int_list(value: str, default: List[int]) -> List[int]:
    if not value:
        return default
    parts = [p.strip() for p in value.replace("+", ",").split(",") if p.strip()]
    out: List[int] = []
    for part in parts:
        out.append(int(part))
    return out or default


def _parse_float_list(value: str, default: List[float]) -> List[float]:
    if not value:
        return default
    parts = [p.strip() for p in value.replace("+", ",").split(",") if p.strip()]
    out: List[float] = []
    for part in parts:
        out.append(float(part))
    return out or default


def _parse_str_list(value: str, default: List[str]) -> List[str]:
    if not value:
        return default
    parts = [p.strip() for p in value.replace("+", ",").split(",") if p.strip()]
    return parts or default


def _parse_int_pair(value: str, default: Tuple[int, int]) -> Tuple[int, int]:
    parsed = _parse_int_list(value, list(default))
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return default


def _parse_float_pair(value: str, default: Tuple[float, float]) -> Tuple[float, float]:
    parsed = _parse_float_list(value, list(default))
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return default

# ===============================================================
# CONFIG
# ===============================================================

SEED = 42

IMAGE_FILENAME = "F1_scaled.png"
# IMAGE_FILENAME = "model_baked.png"

BASE_DIR = (
    os.environ.get("DATASET_BASE_DIR")
    or os.environ.get("BASE_DIR")
    or _LOCAL_BASE_DIR
    or DEFAULT_BASE_DIR
)
if not BASE_DIR:
    raise ValueError(
        "BASE_DIR is not set. Either:\n"
        "- set env var `DATASET_BASE_DIR` (or `BASE_DIR`), or\n"
        "- set `DEFAULT_BASE_DIR` in `configs/config.py`, or\n"
        "- create `configs/config_local.py` (see `configs/config_local_example.py`)."
    )

TRAIN_TXT = os.environ.get("TRAIN_TXT") or os.path.join(BASE_DIR, "train.txt")
VAL_TXT = os.environ.get("VAL_TXT") or os.path.join(BASE_DIR, "val.txt")
TEST_TXT = os.environ.get("TEST_TXT") or os.path.join(BASE_DIR, "test.txt")

RUNS_DIR = os.environ.get("RUNS_DIR") or "./runs"
RUN_NAME = os.environ.get("RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_NAME)
RUN_EXPERIMENTS = os.environ.get("RUN_EXPERIMENTS", "clean+scanned").lower()
CURRICULUM_ENABLE = os.environ.get("CURRICULUM_ENABLE", "0") == "1"
CURRICULUM_ADAPTIVE = os.environ.get("CURRICULUM_ADAPTIVE", "0") == "1"

OUT_DIR = RUN_DIR
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR = os.path.join(OUT_DIR, "visualizations")
MET_DIR = os.path.join(OUT_DIR, "metrics")
LOG_DIR = os.path.join(OUT_DIR, "logs")

BACKBONE = "PekingU/rtdetr_r50vd"
DETECTOR = os.environ.get("DETECTOR", "rtdetr").lower()
EPOCHS = int(os.environ.get("EPOCHS", "100"))
# If > 0, training stops after this many optimizer steps (steps-matched runs).
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LR = float(os.environ.get("LR", "1e-4"))
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_VIS = 10

CLASS_BOOST = {
    "STAIR": 5.0,
    "COLUMN": 3.0,
    "RAILING": 2.5,
}

COCO_MAX_DETS = [1, 10, 100]
NO_OBJECT_WEIGHT = 1.0
MIN_MAPPED_RATIO = 0.05
MAX_NODES = 100

RESIZE_ENABLE = True
# If True, force a fixed (height, width) resize so batches stack without padding.
# If False, use shortest/longest edge resizing and rely on padding.

dimension = int(os.environ.get("IMAGE_SIZE", "1024"))
RESIZE_FIXED = True
RESIZE_HEIGHT = dimension
RESIZE_WIDTH = dimension
RESIZE_SHORTEST_EDGE = dimension
RESIZE_LONGEST_EDGE = dimension

PAD_ENABLE = False
PAD_SIZE = 1024

SUBSET_TRAIN = int(os.environ.get("SUBSET_TRAIN", "0"))
SUBSET_VAL = int(os.environ.get("SUBSET_VAL", "0"))
SUBSET_TEST = int(os.environ.get("SUBSET_TEST", "0"))

# ===============================================================
# Stroke-aware augmentation (training only)
# ===============================================================

AUGMENT_STROKE_ENABLE = os.environ.get("AUGMENT_STROKE_ENABLE", "0") == "1"
AUGMENT_STROKE_PROB = float(os.environ.get("AUGMENT_STROKE_PROB", "0.5"))
AUGMENT_STROKE_KERNEL_SIZES = _parse_int_list(os.environ.get("AUGMENT_STROKE_KERNEL_SIZES", ""), [1, 2, 3])
AUGMENT_STROKE_KERNEL_PROBS = tuple(
    _parse_float_list(os.environ.get("AUGMENT_STROKE_KERNEL_PROBS", ""), [0.4, 0.4, 0.2])
)
AUGMENT_STROKE_DILATE_PROB = float(os.environ.get("AUGMENT_STROKE_DILATE_PROB", "0.7"))
AUGMENT_STROKE_SIGMA_RANGE = _parse_float_pair(os.environ.get("AUGMENT_STROKE_SIGMA_RANGE", ""), (0.3, 1.2))
AUGMENT_STROKE_CONTRAST_RANGE = _parse_float_pair(os.environ.get("AUGMENT_STROKE_CONTRAST_RANGE", ""), (0.9, 1.1))
AUGMENT_STROKE_BRIGHTNESS_RANGE = _parse_float_pair(os.environ.get("AUGMENT_STROKE_BRIGHTNESS_RANGE", ""), (-5.0, 5.0))

if len(AUGMENT_STROKE_KERNEL_SIZES) != len(AUGMENT_STROKE_KERNEL_PROBS):
    raise ValueError(
        "AUGMENT_STROKE_KERNEL_SIZES and AUGMENT_STROKE_KERNEL_PROBS must have the same length."
    )

# CAD image basenames eligible for stroke augmentation.
AUGMENT_APPLY_ALL = os.environ.get("AUGMENT_APPLY_ALL", "0") == "1"
AUGMENT_CAD_FILENAMES = _parse_str_list(
    os.environ.get("AUGMENT_CAD_FILENAMES", ""),
    ["model_baked.png", "01_svg_clean.png"],
)

# Optional scanned background mixing (training only)
AUGMENT_SCAN_MIX_ENABLE = os.environ.get("AUGMENT_SCAN_MIX_ENABLE", "0") == "1"
AUGMENT_SCAN_MIX_PROB = float(os.environ.get("AUGMENT_SCAN_MIX_PROB", "0.2"))
AUGMENT_SCAN_MIX_FILENAME = os.environ.get(
    "AUGMENT_SCAN_MIX_FILENAME",
    "four_final_variants/04_svg_clean_plus_scan_outside.png",
)
AUGMENT_SCAN_RAW_FILENAME = os.environ.get("AUGMENT_SCAN_RAW_FILENAME", "F1_scaled.png")

# ===============================================================
# Box jitter / tolerant regression (training only)
# ===============================================================

AUGMENT_BOX_JITTER_ENABLE = os.environ.get("AUGMENT_BOX_JITTER_ENABLE", "0") == "1"
AUGMENT_BOX_JITTER_PX = float(os.environ.get("AUGMENT_BOX_JITTER_PX", "0.0"))
AUGMENT_BOX_JITTER_SCALE = float(os.environ.get("AUGMENT_BOX_JITTER_SCALE", "0.0"))
AUGMENT_BOX_EXPAND_RATIO = float(os.environ.get("AUGMENT_BOX_EXPAND_RATIO", "0.0"))

# ===============================================================
# Line dropout (training only)
# ===============================================================

AUGMENT_LINE_DROPOUT_ENABLE = os.environ.get("AUGMENT_LINE_DROPOUT_ENABLE", "0") == "1"
AUGMENT_LINE_DROPOUT_PROB = float(os.environ.get("AUGMENT_LINE_DROPOUT_PROB", "0.3"))
AUGMENT_LINE_DROPOUT_COUNT_RANGE = _parse_int_pair(
    os.environ.get("AUGMENT_LINE_DROPOUT_COUNT_RANGE", ""), (3, 8)
)
AUGMENT_LINE_DROPOUT_WIDTH_RANGE = _parse_int_pair(
    os.environ.get("AUGMENT_LINE_DROPOUT_WIDTH_RANGE", ""), (1, 2)
)

# ===============================================================
# Depiction deformation (dilate -> blur -> threshold, training only)
# ===============================================================

AUGMENT_DEPICTION_ENABLE = os.environ.get("AUGMENT_DEPICTION_ENABLE", "0") == "1"
AUGMENT_DEPICTION_PROB = float(os.environ.get("AUGMENT_DEPICTION_PROB", "0.5"))
AUGMENT_DEPICTION_KERNEL_SIZES = _parse_int_list(
    os.environ.get("AUGMENT_DEPICTION_KERNEL_SIZES", ""), [2, 3, 4]
)
AUGMENT_DEPICTION_KERNEL_PROBS = tuple(
    _parse_float_list(os.environ.get("AUGMENT_DEPICTION_KERNEL_PROBS", ""), [0.3, 0.4, 0.3])
)
AUGMENT_DEPICTION_SIGMA_RANGE = _parse_float_pair(
    os.environ.get("AUGMENT_DEPICTION_SIGMA_RANGE", ""), (0.6, 2.0)
)
AUGMENT_DEPICTION_THRESHOLD = int(os.environ.get("AUGMENT_DEPICTION_THRESHOLD", "170"))

# ===============================================================
# Curriculum (D -> C -> Scan)
# ===============================================================

CURRICULUM_STAGE_ORDER = os.environ.get("CURRICULUM_STAGE_ORDER", "D,C,Scan")
CURRICULUM_STAGE_IMAGES = {
    "D": "four_final_variants/04_svg_clean_plus_scan_outside.png",
    "C": "four_final_variants/03_scan_inside_boxes.png",
    "Scan": "F1_scaled.png",
}
CURRICULUM_EPOCHS = _parse_int_list(
    os.environ.get("CURRICULUM_EPOCHS", ""),
    [EPOCHS, EPOCHS, EPOCHS],
)

# ===============================================================
# ROI-weighted sampling
# ===============================================================

ROI_WEIGHT_ENABLE = os.environ.get("ROI_WEIGHT_ENABLE", "0") == "1"
ROI_WEIGHT_SCALE = float(os.environ.get("ROI_WEIGHT_SCALE", "1.0"))
ROI_WEIGHT_BIAS = float(os.environ.get("ROI_WEIGHT_BIAS", "1.0"))

# ===============================================================
# Distance-guided sampling / curriculum
# ===============================================================

DISTANCE_SCORE_FILE = os.environ.get("DISTANCE_SCORE_FILE")
DISTANCE_SCORE_ALPHA = float(os.environ.get("DISTANCE_SCORE_ALPHA", "0.0"))
DISTANCE_SCORE_CLIP_PCT = float(os.environ.get("DISTANCE_SCORE_CLIP_PCT", "95.0"))

DISTANCE_CURRICULUM_ENABLE = os.environ.get("DISTANCE_CURRICULUM_ENABLE", "0") == "1"
DISTANCE_CURRICULUM_THRESHOLDS = tuple(
    _parse_float_list(os.environ.get("DISTANCE_CURRICULUM_THRESHOLDS", ""), [0.5, 0.85])
)
DISTANCE_CURRICULUM_LOW_SIGMA = tuple(
    _parse_float_list(os.environ.get("DISTANCE_CURRICULUM_LOW_SIGMA", ""), [0.0, 0.8])
)
DISTANCE_CURRICULUM_MID_SIGMA = tuple(
    _parse_float_list(os.environ.get("DISTANCE_CURRICULUM_MID_SIGMA", ""), [0.4, 1.6])
)
DISTANCE_CURRICULUM_HIGH_SIGMA = tuple(
    _parse_float_list(os.environ.get("DISTANCE_CURRICULUM_HIGH_SIGMA", ""), [1.0, 2.2])
)
DISTANCE_CURRICULUM_LOW_KERNELS = _parse_int_list(
    os.environ.get("DISTANCE_CURRICULUM_LOW_KERNELS", ""), [0, 1]
)
DISTANCE_CURRICULUM_MID_KERNELS = _parse_int_list(
    os.environ.get("DISTANCE_CURRICULUM_MID_KERNELS", ""), [1, 2, 3]
)
DISTANCE_CURRICULUM_HIGH_KERNELS = _parse_int_list(
    os.environ.get("DISTANCE_CURRICULUM_HIGH_KERNELS", ""), [2, 3, 4]
)

# ===============================================================
# Color Palette
# ===============================================================

'''CLASS_COLORS = {
    "WALL": (255, 215, 0),
    "COLUMN": (139, 69, 19),
    "STAIR": (255, 140, 0),
    "RAILING": (199, 21, 133),
    "DOOR": (30, 144, 255),
    "WINDOW": (0, 191, 255),
    "SPACE_LIVING": (34, 139, 34),
    "SPACE_BED": (46, 139, 87),
    "SPACE_KITCHEN": (218, 165, 32),
    "SPACE_BATH": (186, 85, 211),
    "SPACE_CIRC": (154, 205, 50),
    "SPACE_OUTDOOR": (72, 209, 204),
    "SPACE_OTHER": (112, 128, 144),
    "TOILET": (178, 34, 34),
    "BATH_SHOWER": (255, 160, 122),
    "KITCHEN_UNIT": (255, 165, 0),
    "FURNITURE_OTHER": (65, 105, 225),
}'''

CLASS_COLORS = {
    "WALL":   (255, 215, 0),     # keep
    "COLUMN": (110, 110, 110),   # neutral stone gray (not red!)
    "STAIR":  (255, 140, 0),     # keep
    "RAILING": (199, 21, 133),   # keep
    "DOOR":   (160, 82, 45),     # warm brown (Sienna)
    "WINDOW": (0, 191, 255),    # keep
}
