#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_metrics(exp_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    # Priority: test_metrics.json > experiment_summary.json > latest_epoch.json
    for name in ("test_metrics.json", "experiment_summary.json", "latest_epoch.json"):
        data = _load_json(exp_dir / name)
        if data:
            return data, name
    return None, ""


def _as_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _bool_str(val: Any) -> str:
    return "yes" if bool(val) else "no"


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _find_exp_dirs(runs_root: Path) -> list[Path]:
    exp_dirs = []
    if not runs_root.exists():
        return exp_dirs
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        # common layout: runs/<run_name>/<exp_name>/
        for exp_dir in sorted(run_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if (exp_dir / "config.json").exists():
                exp_dirs.append(exp_dir)
    return exp_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize RT_DETR runs (AP50/AP75, finished, dataset subset, etc.)."
    )
    parser.add_argument(
        "--runs-root",
        default="RT_DETR_final/runs",
        help="Root folder that contains run directories.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional JSON output path (array of rows).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Substring filter on run or experiment path.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    exp_dirs = _find_exp_dirs(runs_root)

    rows = []
    for exp_dir in exp_dirs:
        rel = exp_dir.relative_to(runs_root)
        rel_str = str(rel)
        if args.filter and args.filter not in rel_str:
            continue

        config = _load_json(exp_dir / "config.json") or {}
        metrics, metrics_src = _pick_metrics(exp_dir)

        epochs_target = config.get("EPOCHS")
        epochs_ran = None
        if metrics:
            epochs_ran = metrics.get("epochs_ran") or metrics.get("epoch")

        finished = False
        if metrics_src in ("test_metrics.json", "experiment_summary.json"):
            finished = True
        elif epochs_target is not None and epochs_ran is not None:
            try:
                finished = int(epochs_ran) >= int(epochs_target)
            except Exception:
                finished = False

        subset_train = config.get("SUBSET_TRAIN")
        subset_val = config.get("SUBSET_VAL")
        subset_test = config.get("SUBSET_TEST")
        uses_full = (
            (subset_train == 0 or subset_train is None)
            and (subset_val == 0 or subset_val is None)
            and (subset_test == 0 or subset_test is None)
        )

        row = {
            "run/exp": rel_str,
            "metrics_src": metrics_src or "-",
            "ap50": _as_float(metrics.get("best_ap50")) if metrics else None,
            "ap75": _as_float(metrics.get("best_ap75")) if metrics else None,
            "ap85": _as_float(metrics.get("best_ap85")) if metrics else None,
            "recall": _as_float(metrics.get("best_recall")) if metrics else None,
            "recall85": _as_float(metrics.get("best_recall85")) if metrics else None,
            "best_score": _as_float(metrics.get("best_score")) if metrics else None,
            "epochs_ran": epochs_ran,
            "epochs_target": epochs_target,
            "finished": finished,
            "full_dataset": uses_full,
            "subset_train": subset_train,
            "subset_val": subset_val,
            "subset_test": subset_test,
            "distance_alpha": config.get("DISTANCE_SCORE_ALPHA"),
            "stroke_aug": config.get("AUGMENT_STROKE_ENABLE"),
            "scan_mix": config.get("AUGMENT_SCAN_MIX_ENABLE"),
        }
        rows.append(row)

    # Sort: best AP50 desc, then AP75 desc
    rows.sort(key=lambda r: (r["ap50"] or -1, r["ap75"] or -1), reverse=True)

    headers = [
        "run/exp",
        "ap50",
        "ap75",
        "ap85",
        "recall",
        "recall85",
        "best_score",
        "epochs_ran",
        "epochs_target",
        "finished",
        "full_dataset",
        "subset_train",
        "distance_alpha",
        "stroke_aug",
        "scan_mix",
        "metrics_src",
    ]

    # Print table
    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            val = r.get(h)
            text = _fmt(_bool_str(val) if h in ("finished", "full_dataset") else val)
            col_widths[h] = max(col_widths[h], len(text))

    def _row_line(r: Dict[str, Any]) -> str:
        parts = []
        for h in headers:
            val = r.get(h)
            if h in ("finished", "full_dataset"):
                text = _bool_str(val)
            else:
                text = _fmt(val)
            parts.append(text.ljust(col_widths[h]))
        return "  ".join(parts)

    print(_row_line({h: h for h in headers}))
    print("  ".join("-" * col_widths[h] for h in headers))
    for r in rows:
        print(_row_line(r))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                row = r.copy()
                row["finished"] = _bool_str(row["finished"])
                row["full_dataset"] = _bool_str(row["full_dataset"])
                writer.writerow(row)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
