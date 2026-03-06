#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.paper_pipeline_configs import resolve_preset, apply_env
from utils.rtdetr_core import import_core_main_3_experiments
from utils.run_io import ensure_dir, write_json


def _parse_kv_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected KEY=VALUE): {item}")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _apply_overrides(env_map: Dict[str, Any], overrides: Dict[str, str]) -> None:
    for key, val in overrides.items():
        env_map[key] = val


def _import_core():
    # Import only after env overrides are applied so core reads updated env.
    return import_core_main_3_experiments(ROOT_DIR)


def run() -> None:
    parser = argparse.ArgumentParser(description="Paper pipeline runner")
    parser.add_argument("--preset", default="clean", help="Preset name from configs.py")
    parser.add_argument("--run-name", default="", help="Override RUN_NAME")
    parser.add_argument("--detector", default="", help="Override DETECTOR")
    parser.add_argument(
        "--base-dir",
        default="",
        help="Override dataset base dir (sets DATASET_BASE_DIR).",
    )
    parser.add_argument("--set", action="append", default=[], help="Extra env overrides KEY=VALUE")
    args = parser.parse_args()

    preset = resolve_preset(args.preset)
    mode = preset.get("mode", "core_main")
    env_map: Dict[str, Any] = dict(preset.get("env", {}))

    if args.run_name:
        env_map["RUN_NAME"] = args.run_name
    if args.detector:
        env_map["DETECTOR"] = args.detector
    if args.base_dir:
        env_map["DATASET_BASE_DIR"] = args.base_dir

    overrides = _parse_kv_pairs(args.set)
    _apply_overrides(env_map, overrides)
    apply_env(env_map)

    core = _import_core()

    if mode == "core_main":
        core.main()
        return

    if mode != "single_experiment":
        raise ValueError(f"Unknown preset mode: {mode}")

    exp_name = preset.get("exp_name", "exp_custom")
    image_filename = preset.get("image_filename")
    if not image_filename:
        raise ValueError(f"Preset {args.preset} missing image_filename")

    master_run_dir = core.RUN_DIR
    ensure_dir(master_run_dir)
    res = core.run_experiment(
        master_run_dir=master_run_dir,
        exp_name=exp_name,
        image_filename=image_filename,
        init_weights_dir=None,
    )
    summary = {"master_run_dir": master_run_dir, "experiments": [res]}
    write_json(os.path.join(master_run_dir, "ALL_EXPERIMENTS_SUMMARY.json"), summary)


if __name__ == "__main__":
    run()
