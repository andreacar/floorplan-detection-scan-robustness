#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.paper_io import table_path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.6f}"
    return v


def _subset_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        stats = data.get("coco_stats", {})
        run_name = Path(str(data.get("run_dir", ""))).name
        rows.append(
            {
                "source_file": str(path),
                "run_name": run_name,
                "split": data.get("split", ""),
                "classes": ",".join(data.get("classes", [])),
                "AP": _fmt(float(stats.get("AP", 0.0))),
                "AP50": _fmt(float(stats.get("AP50", 0.0))),
                "AP75": _fmt(float(stats.get("AP75", 0.0))),
                "AR_100": _fmt(float(stats.get("AR_100", 0.0))),
            }
        )
    return rows


def _roi_rows(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        return []

    def _f1_scalar(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            if "weighted_f1_support" in val:
                return float(val["weighted_f1_support"])
            micro = val.get("micro", {})
            if isinstance(micro, dict) and "f1" in micro:
                return float(micro["f1"])
        return 0.0

    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        overall = item.get("overall", {})
        rows.append(
            {
                "model": item.get("model", ""),
                "image_variant": item.get("image_variant", ""),
                "AP": _fmt(float(overall.get("AP", 0.0))),
                "AP50": _fmt(float(overall.get("AP50", 0.0))),
                "AP75": _fmt(float(overall.get("AP75", 0.0))),
                "AR_100": _fmt(float(overall.get("AR_100", 0.0))),
                "f1_all_classes": _fmt(_f1_scalar(item.get("f1_all_classes", 0.0))),
                "f1_popular_classes": _fmt(_f1_scalar(item.get("f1_popular_classes", 0.0))),
            }
        )
    return rows


def _error_rows(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for run_name, run_data in data.items():
        if not isinstance(run_data, dict):
            continue
        overall = run_data.get("overall", {})
        total = int(overall.get("total", 0))
        missed = int(overall.get("missed", 0))
        loose = int(overall.get("loose", 0))
        tight = int(overall.get("tight", 0))
        denom = max(total, 1)
        rows.append(
            {
                "run_name": run_name,
                "total_gt": total,
                "missed": missed,
                "loose": loose,
                "tight": tight,
                "missed_frac": _fmt(missed / denom),
                "loose_frac": _fmt(loose / denom),
                "tight_frac": _fmt(tight / denom),
            }
        )
    return rows


def _size_rows(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for run_name, run_data in data.items():
        if not isinstance(run_data, dict):
            continue
        l50 = run_data.get("logistic_50", {})
        l85 = run_data.get("logistic_85", {})
        rows.append(
            {
                "run_name": run_name,
                "count": int(run_data.get("count", 0)),
                "area_at_50pct_iou50": _fmt(float(l50.get("area_at_50pct", 0.0))),
                "w_iou50": _fmt(float(l50.get("w", 0.0))),
                "b_iou50": _fmt(float(l50.get("b", 0.0))),
                "area_at_50pct_iou85": _fmt(float(l85.get("area_at_50pct", 0.0))),
                "w_iou85": _fmt(float(l85.get("w", 0.0))),
                "b_iou85": _fmt(float(l85.get("b", 0.0))),
            }
        )
    return rows


def _factorized_rows(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    rows: List[Dict[str, Any]] = []
    baseline = data.get("baseline", {})
    if isinstance(baseline, dict):
        rows.append(
            {
                "factor": "clean",
                "level": "baseline",
                "ap50_mean": _fmt(float(baseline.get("ap50_mean", 0.0))),
                "ap75_mean": _fmt(float(baseline.get("ap75_mean", 0.0))),
                "delta_ap50_mean": _fmt(0.0),
                "delta_ap75_mean": _fmt(0.0),
                "missed_frac": _fmt(float(baseline.get("error_decomposition", {}).get("missed_frac", 0.0))),
                "loose_frac": _fmt(float(baseline.get("error_decomposition", {}).get("loose_frac", 0.0))),
                "tight_frac": _fmt(float(baseline.get("error_decomposition", {}).get("tight_frac", 0.0))),
            }
        )

    factors = data.get("factors", {})
    if isinstance(factors, dict):
        for factor_name, levels in factors.items():
            if not isinstance(levels, dict):
                continue
            for level_name, metrics in levels.items():
                if not isinstance(metrics, dict):
                    continue
                rows.append(
                    {
                        "factor": factor_name,
                        "level": level_name,
                        "ap50_mean": _fmt(float(metrics.get("ap50_mean", 0.0))),
                        "ap75_mean": _fmt(float(metrics.get("ap75_mean", 0.0))),
                        "delta_ap50_mean": _fmt(float(metrics.get("delta_ap50_mean", 0.0))),
                        "delta_ap75_mean": _fmt(float(metrics.get("delta_ap75_mean", 0.0))),
                        "missed_frac": _fmt(float(metrics.get("error_decomposition", {}).get("missed_frac", 0.0))),
                        "loose_frac": _fmt(float(metrics.get("error_decomposition", {}).get("loose_frac", 0.0))),
                        "tight_frac": _fmt(float(metrics.get("error_decomposition", {}).get("tight_frac", 0.0))),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables under artifacts/tables.")
    parser.add_argument("--subset-glob", default="artifacts/tables/table_subset_softmax_*.json")
    parser.add_argument("--roi-json", default="AB_on_CD_results/ALL_AB_ON_CD_RESULTS.json")
    parser.add_argument("--error-json", default="paper_experiments/out/error_compare/summary.json")
    parser.add_argument("--size-json", default="paper_experiments/out/size_compare/summary.json")
    parser.add_argument("--factorized-json", default="paper_experiments/out/factorized_pinned/summary.json")
    args = parser.parse_args()

    subset_paths = sorted((ROOT / ".").glob(args.subset_glob))
    roi_path = ROOT / args.roi_json
    error_path = ROOT / args.error_json
    size_path = ROOT / args.size_json
    factorized_path = ROOT / args.factorized_json

    subset_rows = _subset_rows(subset_paths)
    roi_rows = _roi_rows(roi_path)
    error_rows = _error_rows(error_path)
    size_rows = _size_rows(size_path)
    factorized_rows = _factorized_rows(factorized_path)

    outputs = {
        "table_subset_softmax.csv": {
            "rows": len(subset_rows),
            "source": [str(p) for p in subset_paths],
        },
        "table_roi_ab_on_cd.csv": {
            "rows": len(roi_rows),
            "source": str(roi_path),
        },
        "table_error_decomposition.csv": {
            "rows": len(error_rows),
            "source": str(error_path),
        },
        "table_size_success.csv": {
            "rows": len(size_rows),
            "source": str(size_path),
        },
        "table_factorized_degradation.csv": {
            "rows": len(factorized_rows),
            "source": str(factorized_path),
        },
    }

    _write_csv(
        table_path("table_subset_softmax.csv"),
        subset_rows,
        ["source_file", "run_name", "split", "classes", "AP", "AP50", "AP75", "AR_100"],
    )
    _write_csv(
        table_path("table_roi_ab_on_cd.csv"),
        roi_rows,
        [
            "model",
            "image_variant",
            "AP",
            "AP50",
            "AP75",
            "AR_100",
            "f1_all_classes",
            "f1_popular_classes",
        ],
    )
    _write_csv(
        table_path("table_error_decomposition.csv"),
        error_rows,
        ["run_name", "total_gt", "missed", "loose", "tight", "missed_frac", "loose_frac", "tight_frac"],
    )
    _write_csv(
        table_path("table_size_success.csv"),
        size_rows,
        [
            "run_name",
            "count",
            "area_at_50pct_iou50",
            "w_iou50",
            "b_iou50",
            "area_at_50pct_iou85",
            "w_iou85",
            "b_iou85",
        ],
    )
    _write_csv(
        table_path("table_factorized_degradation.csv"),
        factorized_rows,
        [
            "factor",
            "level",
            "ap50_mean",
            "ap75_mean",
            "delta_ap50_mean",
            "delta_ap75_mean",
            "missed_frac",
            "loose_frac",
            "tight_frac",
        ],
    )

    manifest_path = table_path("table_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)

    print(f"[OK] Wrote tables to: {table_path('.').resolve()}")
    print(f"[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
