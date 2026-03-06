#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def summarize_degradation(summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out = {"factors": {}}
    for factor, values in summary.items():
        out["factors"][factor] = {
            "mean_delta_recall85": float(values.get("mean_delta_recall85", 0.0)),
            "mean_dist_l2": float(values.get("mean_dist_l2", 0.0)),
        }
    return out


def summarize_shift(records: List[Dict]) -> Dict[str, Any]:
    if not records:
        return {}
    distances = []
    recalls = []
    for rec in records:
        distances.append(float(rec.get("global_dist", 0.0)))
        recalls.append(float(rec.get("recall85_scanned", 0.0)))
    return {
        "layout_count": len(records),
        "avg_global_dist": float(statistics.mean(distances)),
        "avg_recall85_scanned": float(statistics.mean(recalls)),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate diagnostics from degradation/shift outputs.")
    parser.add_argument(
        "--degradation-summary",
        default="paper_experiments/out/degradation/degradation_summary.json",
        help="Factor-level summary produced by degradation_sweep.",
    )
    parser.add_argument(
        "--shift-summary",
        default="paper_experiments/out/shift_maps/shift_summary.json",
        help="Layout-level shift summary produced by shift_localization.",
    )
    parser.add_argument(
        "--out-json",
        default="paper_experiments/out/analysis_combined.json",
        help="Path to write the combined analysis JSON.",
    )
    args = parser.parse_args()

    degrade = _load_json(args.degradation_summary)
    shift = _load_json(args.shift_summary)

    analysis = {}
    if isinstance(degrade, dict):
        analysis["degradation"] = summarize_degradation(degrade)
    if isinstance(shift, list):
        analysis["shift"] = summarize_shift(shift)

    with open(args.out_json, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis summary written to {args.out_json}")
    if "degradation" in analysis:
        print("Degradation factors:", ", ".join(analysis["degradation"]["factors"].keys()))
    if "shift" in analysis:
        print(
            f"{analysis['shift']['layout_count']} layouts analyzed, avg early global dist "
            f"{analysis['shift']['avg_global_dist']:.3f}, avg scanned recall {analysis['shift']['avg_recall85_scanned']:.3f}"
        )


if __name__ == "__main__":
    main()
