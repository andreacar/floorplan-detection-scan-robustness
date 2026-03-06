import os
from typing import Dict, Any


# Preset-driven environment overrides for paper experiments.
# All values are strings to match os.environ conventions.
PRESETS: Dict[str, Dict[str, Any]] = {
    # Core clean CAD training.
    "clean": {
        "mode": "core_main",
        "env": {
            "RUN_EXPERIMENTS": "clean",
        },
    },
    # Core scanned (scanned) training.
    "scan": {
        "mode": "core_main",
        "env": {
            "RUN_EXPERIMENTS": "scanned",
        },
    },
    # Clean + scan in one run.
    "clean_scan": {
        "mode": "core_main",
        "env": {
            "RUN_EXPERIMENTS": "clean+scanned",
        },
    },
    # Curriculum experiments (if configured in RT_DETR_final/config.py).
    "curriculum": {
        "mode": "core_main",
        "env": {
            "RUN_EXPERIMENTS": "curriculum",
            "CURRICULUM_ENABLE": "1",
        },
    },
    # Cleaned scans (from main_3_experiments_cleaned_scans.py).
    "cleaned_scans": {
        "mode": "single_experiment",
        "exp_name": "exp_cleaned_scans",
        "image_filename": "four_final_variants/05_scan_boxes_polygon.png",
        "env": {
            "FAST_MODE": "1",
            "FAST_VAL_MAX": "200",
            "FAST_VAL_INTERVAL": "5",
            "FAST_SKIP_VIS": "1",
            "FAST_SKIP_AWR": "1",
        },
    },
    # ROI-weighted sampling preset (for robustness / ROI experiments).
    "roi": {
        "mode": "core_main",
        "env": {
            "RUN_EXPERIMENTS": "scanned",
            "ROI_WEIGHT_ENABLE": "1",
            "ROI_WEIGHT_SCALE": "1.0",
            "ROI_WEIGHT_BIAS": "1.0",
        },
    },
}


def resolve_preset(name: str) -> Dict[str, Any]:
    preset = PRESETS.get(name)
    if not preset:
        raise KeyError(f"Unknown preset: {name}")
    return preset


def apply_env(env_map: Dict[str, Any]) -> None:
    for key, val in env_map.items():
        os.environ[str(key)] = str(val)
