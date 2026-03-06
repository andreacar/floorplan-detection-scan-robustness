#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the exact union polygon of label boxes as a red outline overlay.

Example:
  python RT_DETR_final/diagnostic/render_boxes_union_polygon.py \
    --folder /path/to/dataset/high_quality_architectural/180 \
    --hier-json /path/to/your/hierarchy.json \
    --image F1_scaled.png \
    --graph graph.json \
    --out boxes_union_polygon_overlay.png \
    --mask-out boxes_union_mask.png
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import json
import os
import sys
from PIL import Image, ImageChops, ImageDraw, ImageFilter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(REPO_ROOT)

from RT_DETR_final.utils.geometry import clamp_bbox_xywh
from RT_DETR_final import hierarchy_config as hier


def _node_label(node) -> str:
    return str(node.get("data_class") or node.get("category") or "")


def _is_structural(node) -> bool:
    raw = _node_label(node).lower()
    return any(
        key in raw
        for key in (
            "door",
            "window",
            "wall",
            "column",
            "railing",
            "stair",
        )
    )


def _has_background_neighbor(pix, x, y, w, h) -> bool:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                return True
            if pix[nx, ny] == 0:
                return True
    return False


def _trace_boundary(mask: Image.Image):
    """Trace outer boundary of a binary mask (8-connected)."""
    w, h = mask.size
    pix = mask.load()

    start = None
    for y in range(h):
        for x in range(w):
            if pix[x, y] > 0 and _has_background_neighbor(pix, x, y, w, h):
                start = (x, y)
                break
        if start:
            break
    if start is None:
        return []

    # Moore-Neighbor tracing
    neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0),
                 (1, 1), (0, 1), (-1, 1), (-1, 0)]
    b0 = start
    c0 = (start[0] - 1, start[1])  # left of start (may be outside)
    b = b0
    c = c0
    boundary = [b0]

    def _idx(dx, dy):
        for i, (nx, ny) in enumerate(neighbors):
            if nx == dx and ny == dy:
                return i
        return 0

    while True:
        # starting from neighbor after c, scan clockwise
        idx = _idx(c[0] - b[0], c[1] - b[1])
        found = None
        for k in range(8):
            ni = (idx + 1 + k) % 8
            nx, ny = b[0] + neighbors[ni][0], b[1] + neighbors[ni][1]
            if 0 <= nx < w and 0 <= ny < h and pix[nx, ny] > 0:
                # next boundary pixel
                found = (nx, ny, ni)
                break
        if found is None:
            break
        nx, ny, ni = found
        # c is the neighbor just before the found pixel
        prev_i = (ni - 1) % 8
        c = (b[0] + neighbors[prev_i][0], b[1] + neighbors[prev_i][1])
        b = (nx, ny)
        if b == b0 and c == c0:
            break
        boundary.append(b)

    return boundary


def _polygon_fill_from_mask(mask: Image.Image) -> Image.Image:
    """Create a filled polygon mask from the outer boundary of the mask."""
    boundary = _trace_boundary(mask)
    poly = Image.new("L", mask.size, 0)
    if boundary:
        ImageDraw.Draw(poly).polygon(boundary, fill=255)
    return poly


def build_union_mask(folder, graph_name, hier_json, img_w, img_h):
    with open(os.path.join(folder, graph_name), "r") as f:
        graph = json.load(f)

    level2_classes, raw_to_l2 = hier.load_level2_classes_and_mapping(hier_json)
    label2id = {c: i for i, c in enumerate(level2_classes)}

    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        raw = _node_label(node).strip()
        if raw_to_l2.get(raw) not in label2id:
            continue
        if not _is_structural(node):
            continue
        cl = clamp_bbox_xywh(bbox, img_w, img_h)
        if cl is None:
            continue
        x, y, w, h = cl
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], fill=255)

    return _polygon_fill_from_mask(mask)


def main():
    parser = argparse.ArgumentParser(description="Render union polygon outline of boxes.")
    parser.add_argument("--folder", required=True, help="Folder containing image + graph.json.")
    parser.add_argument("--hier-json", required=True, help="Hierarchy JSON path.")
    parser.add_argument("--image", default="F1_scaled.png", help="Background image filename.")
    parser.add_argument("--graph", default="graph.json", help="Graph JSON filename.")
    parser.add_argument("--out", default="boxes_union_polygon_overlay.png", help="Output overlay filename.")
    parser.add_argument("--mask-out", default=None, help="Optional mask output filename.")
    args = parser.parse_args()

    img_path = os.path.join(args.folder, args.image)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    mask = build_union_mask(args.folder, args.graph, args.hier_json, w, h)

    # Edge = mask - eroded(mask)
    eroded = mask.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.subtract(mask, eroded)

    overlay = img.convert("RGBA")
    red = Image.new("RGBA", overlay.size, (255, 0, 0, 255))
    overlay.paste(red, mask=edge)

    out_path = os.path.join(args.folder, args.out)
    overlay.convert("RGB").save(out_path)
    if args.mask_out:
        mask.save(os.path.join(args.folder, args.mask_out))

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
