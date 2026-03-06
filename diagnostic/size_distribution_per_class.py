#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as config_module
import hierarchy_config as hier
from utils.geometry import clamp_bbox_xywh
from utils.paths import load_split_list


def _parse_splits(value: str) -> List[str]:
    if not value:
        return ["train", "val", "test"]
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    return parts or ["train", "val", "test"]


def _resolve_split_dirs(splits: List[str]) -> Dict[str, List[str]]:
    out = {}
    for split in splits:
        if split == "train":
            out[split] = load_split_list(config_module.TRAIN_TXT)
        elif split == "val":
            out[split] = load_split_list(config_module.VAL_TXT)
        elif split == "test":
            out[split] = load_split_list(config_module.TEST_TXT)
        else:
            raise ValueError(f"Unknown split '{split}' (expected train,val,test)")
    return out


def _load_nodes(graph_path: str) -> List[Tuple[List[float], str]]:
    with open(graph_path, "r") as f:
        data = json.load(f)
    nodes = []
    for node in data.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        raw = node.get("data_class", "") or node.get("category", "")
        nodes.append((bbox, str(raw).strip()))
    return nodes


def _passes_filters(nodes: List[Tuple[List[float], str]], img_w: float, img_h: float) -> bool:
    total_count = 0
    valid_count = 0
    mapped_count = 0
    for bbox, raw in nodes:
        total_count += 1
        clamped = clamp_bbox_xywh(bbox, img_w, img_h)
        if clamped is None:
            continue
        valid_count += 1
        l2 = hier.map_raw_to_l2(raw)
        if l2 is not None:
            mapped_count += 1

    if total_count > config_module.MAX_NODES:
        return False
    if mapped_count == 0:
        return False
    if valid_count > 0 and (mapped_count / valid_count) < config_module.MIN_MAPPED_RATIO:
        return False
    return True


def _resize_scales(orig_w: float, orig_h: float) -> Tuple[float, float, int, int]:
    if not config_module.RESIZE_ENABLE:
        return 1.0, 1.0, int(round(orig_w)), int(round(orig_h))
    if config_module.RESIZE_FIXED:
        sx = float(config_module.RESIZE_WIDTH) / float(orig_w)
        sy = float(config_module.RESIZE_HEIGHT) / float(orig_h)
        return sx, sy, int(config_module.RESIZE_WIDTH), int(config_module.RESIZE_HEIGHT)

    shortest = float(config_module.RESIZE_SHORTEST_EDGE)
    longest = float(config_module.RESIZE_LONGEST_EDGE)
    scale = shortest / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > longest:
        scale = longest / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    return new_w / float(orig_w), new_h / float(orig_h), new_w, new_h


def _log_fn(log_base: str):
    if log_base in ("e", "natural"):
        return math.log
    base = float(log_base)
    if base == 10.0:
        return math.log10
    return lambda x: math.log(x) / math.log(base)


def _quantile_sorted(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    count = len(vals)
    mean = sum(vals) / count
    return {
        "count": count,
        "min": vals[0],
        "p05": _quantile_sorted(vals, 0.05),
        "p25": _quantile_sorted(vals, 0.25),
        "p50": _quantile_sorted(vals, 0.50),
        "p75": _quantile_sorted(vals, 0.75),
        "p95": _quantile_sorted(vals, 0.95),
        "max": vals[-1],
        "mean": mean,
    }


def _folder_meta(folder: str) -> Tuple[str, str]:
    rel = os.path.relpath(folder, config_module.BASE_DIR)
    parts = rel.split(os.sep)
    dataset = parts[0] if parts and parts[0] != ".." else "unknown"
    sample_id = parts[-1] if parts else os.path.basename(folder)
    return dataset, sample_id


def _collect_domain(
    domain_label: str,
    image_filename: str,
    split_dirs: Dict[str, List[str]],
    log_base: str,
    log_eps: float,
    writer: csv.writer,
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]],
    Dict[str, int],
]:
    log_area = defaultdict(lambda: defaultdict(list))
    area = defaultdict(lambda: defaultdict(list))
    skip_counts = defaultdict(int)
    log_fn = _log_fn(log_base)

    for split_name, folders in split_dirs.items():
        for folder in tqdm(folders, desc=f"{domain_label}/{split_name}"):
            img_path = os.path.join(folder, image_filename)
            graph_path = os.path.join(folder, "graph.json")
            if not (os.path.exists(img_path) and os.path.exists(graph_path)):
                skip_counts["missing_files"] += 1
                continue

            nodes = _load_nodes(graph_path)
            if not nodes:
                skip_counts["empty_graph"] += 1
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            if not _passes_filters(nodes, img_w, img_h):
                skip_counts["filtered"] += 1
                continue

            sx, sy, new_w, new_h = _resize_scales(img_w, img_h)
            dataset_name, sample_id = _folder_meta(folder)

            for bbox, raw in nodes:
                l2 = hier.map_raw_to_l2(raw)
                if l2 is None:
                    continue
                clamped = clamp_bbox_xywh(bbox, img_w, img_h)
                if clamped is None:
                    continue
                _, _, w, h = clamped
                resized_area = float(w) * float(h) * sx * sy
                if resized_area <= 0:
                    continue
                log_a = log_fn(resized_area + log_eps)

                area[split_name][l2].append(resized_area)
                log_area[split_name][l2].append(log_a)

                writer.writerow([
                    domain_label,
                    split_name,
                    dataset_name,
                    sample_id,
                    l2,
                    resized_area,
                    log_a,
                    img_w,
                    img_h,
                    new_w,
                    new_h,
                ])

    return area, log_area, skip_counts


def _merge_splits(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
    merged = defaultdict(list)
    for split_vals in data.values():
        for cls, vals in split_vals.items():
            merged[cls].extend(vals)
    return merged


def _maybe_plot(
    out_dir: str,
    plot_kind: str,
    log_base: str,
    domain_logs: Dict[str, Dict[str, List[float]]],
    compare_domains: Iterable[str],
    domain_medians: Dict[str, Dict[str, float]],
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skipping plot.")
        return

    classes = list(hier.LEVEL2_CLASSES)
    n = len(classes)
    cols = 3
    rows = int(math.ceil(n / cols))

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    compare_domains = list(compare_domains)
    main_domain = compare_domains[0] if compare_domains else next(iter(domain_logs), "all")
    for idx, cls in enumerate(classes):
        ax = axes[idx // cols][idx % cols]

        if compare_domains:
            for di, domain in enumerate(compare_domains):
                vals = domain_logs.get(domain, {}).get(cls, [])
                if not vals:
                    continue
                if plot_kind == "hist":
                    ax.hist(vals, bins=40, alpha=0.4, label=domain, density=True)
                else:
                    pos = di + 1
                    ax.violinplot(vals, positions=[pos], showextrema=False)
                    ax.set_xticks(list(range(1, len(compare_domains) + 1)))
                    ax.set_xticklabels(list(compare_domains))
        else:
            vals = domain_logs.get(main_domain, {}).get(cls, [])
            if plot_kind == "hist":
                ax.hist(vals, bins=50, alpha=0.8, density=True)
            else:
                ax.violinplot(vals, showextrema=False)

        for domain, medians in domain_medians.items():
            if cls in medians:
                ax.axvline(medians[cls], linestyle="--", linewidth=1.0, label=f"{domain} p50")

        ax.set_title(cls)
        ax.set_xlabel(f"log{log_base}(area)")
        ax.set_ylabel("density")

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    out_path = os.path.join(out_dir, f"size_distribution_{plot_kind}.pdf")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-class log-area distributions after resize."
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to include (train,val,test).",
    )
    parser.add_argument(
        "--image",
        default=config_module.IMAGE_FILENAME,
        help="Image filename to use for the main domain.",
    )
    parser.add_argument("--label", default="all", help="Label for the main domain.")
    parser.add_argument("--cad-image", default="", help="Optional CAD image filename.")
    parser.add_argument("--scan-image", default="", help="Optional scan image filename.")
    parser.add_argument("--cad-label", default="cad", help="Label for CAD domain.")
    parser.add_argument("--scan-label", default="scan", help="Label for scan domain.")
    parser.add_argument(
        "--log-base",
        default="10",
        help="Log base: '10', 'e', or any float value.",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-8,
        help="Epsilon added before log to avoid log(0).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "diagnostic", "outputs", "size_distribution"),
        help="Output directory for CSV/JSON/plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a quick PDF plot (hist or violin).",
    )
    parser.add_argument(
        "--plot-kind",
        default="violin",
        choices=["hist", "violin"],
        help="Plot type for --plot.",
    )
    args = parser.parse_args()

    split_names = _parse_splits(args.splits)
    split_dirs = _resolve_split_dirs(split_names)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "size_distribution_boxes.csv")
    json_path = os.path.join(args.output_dir, "size_distribution_summary.json")

    domains = [(args.label, args.image)]
    if args.cad_image:
        domains.append((args.cad_label, args.cad_image))
    if args.scan_image:
        domains.append((args.scan_label, args.scan_image))

    summary = {
        "config": {
            "resize_enable": config_module.RESIZE_ENABLE,
            "resize_fixed": config_module.RESIZE_FIXED,
            "resize_height": config_module.RESIZE_HEIGHT,
            "resize_width": config_module.RESIZE_WIDTH,
            "resize_shortest_edge": config_module.RESIZE_SHORTEST_EDGE,
            "resize_longest_edge": config_module.RESIZE_LONGEST_EDGE,
            "log_base": args.log_base,
            "log_eps": args.log_eps,
            "splits": split_names,
            "domains": [
                {"label": label, "image_filename": filename}
                for label, filename in domains
            ],
        },
        "domains": {},
    }

    domain_logs_for_plot = {}
    domain_medians_for_plot = {}
    compare_domains = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "split",
            "dataset",
            "sample_id",
            "class_name",
            "area_resized",
            "log_area",
            "orig_w",
            "orig_h",
            "resized_w",
            "resized_h",
        ])

        for domain_label, image_filename in domains:
            areas, log_areas, skips = _collect_domain(
                domain_label,
                image_filename,
                split_dirs,
                args.log_base,
                args.log_eps,
                writer,
            )

            domain_summary = {"splits": {}, "all": {}, "skips": dict(skips)}
            merged_logs = _merge_splits(log_areas)
            merged_areas = _merge_splits(areas)
            domain_logs_for_plot[domain_label] = merged_logs
            domain_medians_for_plot[domain_label] = {
                cls: _summarize(vals).get("p50", float("nan"))
                for cls, vals in merged_logs.items()
                if vals
            }

            for split, split_vals in log_areas.items():
                domain_summary["splits"][split] = {}
                for cls in hier.LEVEL2_CLASSES:
                    stats = _summarize(split_vals.get(cls, []))
                    area_stats = _summarize(areas.get(split, {}).get(cls, []))
                    domain_summary["splits"][split][cls] = {
                        "log_area": stats,
                        "area": area_stats,
                    }

            for cls in hier.LEVEL2_CLASSES:
                domain_summary["all"][cls] = {
                    "log_area": _summarize(merged_logs.get(cls, [])),
                    "area": _summarize(merged_areas.get(cls, [])),
                }

            summary["domains"][domain_label] = domain_summary
            compare_domains.append(domain_label)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.plot:
        _maybe_plot(
            args.output_dir,
            args.plot_kind,
            args.log_base,
            domain_logs_for_plot,
            compare_domains if len(compare_domains) > 1 else [],
            domain_medians_for_plot,
        )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if args.plot:
        print(f"Wrote {os.path.join(args.output_dir, f'size_distribution_{args.plot_kind}.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
