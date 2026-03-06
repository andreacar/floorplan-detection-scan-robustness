"""
Run with:
python -m diagnostic.render_4_final_variants
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw
import yaml

# ===============================================================
# PATHS
# ===============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(REPO_ROOT)

try:
    from utils.geometry import clamp_bbox_xywh
    import hierarchy_config as hier
except ModuleNotFoundError:
    # Backward-compatibility for older repo layouts.
    from RT_DETR_final.utils.geometry import clamp_bbox_xywh
    from RT_DETR_final import hierarchy_config as hier


def _load_cfg(path: str | None):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ===============================================================
# GEOMETRY HELPERS
# ===============================================================
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

    neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0),
                 (1, 1), (0, 1), (-1, 1), (-1, 0)]
    b0 = start
    c0 = (start[0] - 1, start[1])
    b = b0
    c = c0
    boundary = [b0]

    def _idx(dx, dy):
        for i, (nx, ny) in enumerate(neighbors):
            if nx == dx and ny == dy:
                return i
        return 0

    while True:
        idx = _idx(c[0] - b[0], c[1] - b[1])
        found = None
        for k in range(8):
            ni = (idx + 1 + k) % 8
            nx, ny = b[0] + neighbors[ni][0], b[1] + neighbors[ni][1]
            if 0 <= nx < w and 0 <= ny < h and pix[nx, ny] > 0:
                found = (nx, ny, ni)
                break
        if found is None:
            break
        nx, ny, ni = found
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

# ===============================================================
# CONFIG
# ===============================================================
ROOT_DATA = None

IMG_SCAN = "F1_scaled.png"        # raw scan
IMG_SVG = "model_baked.png"       # clean SVG raster
GRAPH_NAME = "graph.json"

HIER_JSON = None

# ---- save mode ----
SAVE_MODE = "dataset"      # "dataset" or "comparisons"

# ---- margins ----
BOX_MARGIN_PX_INSIDE = 3    # variant 3 (expanded)
BOX_MARGIN_PX_OUTSIDE = 1   # variant 4 (shrunk → more background)

def _process_folder(
    folder: str,
    raw_to_l2: dict,
    label2id: dict,
    save_mode: str,
):
    name = os.path.basename(folder)
    print(f"\n=== Processing {name} ===")
    required_files = [IMG_SCAN, IMG_SVG, GRAPH_NAME]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(folder, f))]

    if missing:
        print(f"  [SKIP] Missing files in {name}: {missing}")
        return name

    # -----------------------------------------------------------
    # OUTPUT DIRECTORY (toggle)
    # -----------------------------------------------------------
    if save_mode == "dataset":
        out_dir = os.path.join(folder, "four_final_variants")
    elif save_mode == "comparisons":
        out_dir =_attach = os.path.join(CURRENT_DIR, "four_final_variants", name)
    else:
        raise ValueError(f"Unknown SAVE_MODE: {save_mode}")

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------
    img_scan = Image.open(os.path.join(folder, IMG_SCAN)).convert("RGBA")
    img_svg = Image.open(os.path.join(folder, IMG_SVG)).convert("RGBA")

    with open(os.path.join(folder, GRAPH_NAME), "r") as f:
        graph = json.load(f)

    ws, hs = img_scan.size
    wv, hv = img_svg.size

    WORLD_W = max(ws, wv)
    WORLD_H = max(hs, hv)

    # -----------------------------------------------------------
    # WORLD-ALIGNED CANVASES
    # -----------------------------------------------------------
    canvas_scan = Image.new("RGBA", (WORLD_W, WORLD_H), (255, 255, 255, 255))
    canvas_svg = Image.new("RGBA", (WORLD_W, WORLD_H), (255, 255, 255, 255))

    canvas_scan.paste(img_scan, (0, 0))
    canvas_svg.paste(img_svg, (0, 0))

    # -----------------------------------------------------------
    # BUILD MASKS
    # -----------------------------------------------------------
    mask_inside = Image.new("L", (WORLD_W, WORLD_H), 0)
    mask_inside_loose = Image.new("L", (WORLD_W, WORLD_H), 0)
    mask_struct_union = Image.new("L", (WORLD_W, WORLD_H), 0)

    draw_inside = ImageDraw.Draw(mask_inside)
    draw_loose = ImageDraw.Draw(mask_inside_loose)
    draw_struct = ImageDraw.Draw(mask_struct_union)

    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue

        raw = str(node.get("data_class") or node.get("category") or "").strip()
        if raw_to_l2.get(raw) not in label2id:
            continue

        cl = clamp_bbox_xywh(bbox, WORLD_W, WORLD_H)
        if cl is None:
            continue

        x, y, w, h = cl

        # ---- variant 3: expanded boxes ----
        x0 = max(0, x - BOX_MARGIN_PX_INSIDE)
        y0 = max(0, y - BOX_MARGIN_PX_INSIDE)
        x1 = min(WORLD_W, x + w + BOX_MARGIN_PX_INSIDE)
        y1 = min(WORLD_H, y + h + BOX_MARGIN_PX_INSIDE)
        draw_inside.rectangle([x0, y0, x1, y1], fill=255)

        # ---- variant 4: shrunk boxes ----
        x0 = max(0, x + BOX_MARGIN_PX_OUTSIDE)
        y0 = max(0, y + BOX_MARGIN_PX_OUTSIDE)
        x1 = min(WORLD_W, x + w - BOX_MARGIN_PX_OUTSIDE)
        y1 = min(WORLD_H, y + h - BOX_MARGIN_PX_OUTSIDE)

        if x1 > x0 and y1 > y0:
            draw_loose.rectangle([x0, y0, x1, y1], fill=255)

        # ---- structural union (for polygon mask) ----
        if _is_structural(node):
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(WORLD_W, x + w)
            y1 = min(WORLD_H, y + h)
            if x1 > x0 and y1 > y0:
                draw_struct.rectangle([x0, y0, x1, y1], fill=255)

    mask_struct_polygon = _polygon_fill_from_mask(mask_struct_union)

    mask_outside_loose = Image.eval(mask_inside_loose, lambda p: 255 - p)

    # -----------------------------------------------------------
    # VARIANT 1 — CLEAN SVG
    # -----------------------------------------------------------
    canvas_svg.save(os.path.join(out_dir, "01_svg_clean.png"))

    # -----------------------------------------------------------
    # VARIANT 2 — RAW SCAN
    # -----------------------------------------------------------
    canvas_scan.save(os.path.join(out_dir, "02_scan_raw.png"))

    # -----------------------------------------------------------
    # VARIANT 3 — SCAN INSIDE BOXES
    # -----------------------------------------------------------
    scan_inside = Image.new("RGBA", (WORLD_W, WORLD_H), (255, 255, 255, 255))
    scan_inside.paste(canvas_scan, (0, 0), mask=mask_inside)
    scan_inside.save(os.path.join(out_dir, "03_scan_inside_boxes.png"))

    # -----------------------------------------------------------
    # VARIANT 5 — SCAN CLIPPED TO STRUCTURAL POLYGON (box union outline)
    # -----------------------------------------------------------
    scan_union = Image.new("RGBA", (WORLD_W, WORLD_H), (255, 255, 255, 255))
    scan_union.paste(canvas_scan, (0, 0), mask=mask_struct_polygon)
    scan_union.save(os.path.join(out_dir, "05_scan_boxes_polygon.png"))

    # -----------------------------------------------------------
    # VARIANT 4 — CLEAN SVG + SCAN OUTSIDE BOXES
    # -----------------------------------------------------------
    clean_plus_scan_outside = canvas_svg.copy()
    clean_plus_scan_outside.paste(canvas_scan, (0, 0), mask=mask_outside_loose)
    clean_plus_scan_outside.save(
        os.path.join(out_dir, "04_svg_clean_plus_scan_outside.png")
    )

    return name


def main():
    parser = argparse.ArgumentParser(description="Render scan/svg variants with optional multiprocessing.")
    parser.add_argument("--config", default=None, help="Optional YAML file with defaults for root_data/hier_json.")
    parser.add_argument("--root-data", default=ROOT_DATA, help="Root folder containing sample subfolders.")
    parser.add_argument("--hier-json", default=HIER_JSON, help="Hierarchy JSON path.")
    parser.add_argument("--save-mode", default=SAVE_MODE, choices=["dataset", "comparisons"], help="Output mode.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Number of parallel workers.")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    root_data = args.root_data or cfg.get("root_data")
    hier_json = args.hier_json or cfg.get("hier_json")
    if not root_data:
        parser.error("Missing dataset root. Pass --root-data or provide root_data in --config.")

    # Load hierarchy once in parent, pass mappings to workers
    level2_classes, raw_to_l2 = hier.load_level2_classes_and_mapping(hier_json)
    label2id = {c: i for i, c in enumerate(level2_classes)}

    if os.path.isfile(os.path.join(root_data, GRAPH_NAME)):
        folders = [root_data]
    else:
        folders = sorted(
            os.path.join(root_data, d)
            for d in os.listdir(root_data)
            if os.path.isdir(os.path.join(root_data, d))
        )

    print(f"Processing {len(folders)} drawings")
    print(f"Save mode: {args.save_mode}")
    print(f"Jobs: {args.jobs}")

    if args.jobs <= 1:
        for folder in folders:
            _process_folder(folder, raw_to_l2, label2id, args.save_mode)
        print("\nDone. All drawings processed.")
        return

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [
            ex.submit(_process_folder, folder, raw_to_l2, label2id, args.save_mode)
            for folder in folders
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"[PROGRESS] {done}/{len(folders)}")
            except Exception as e:
                print(f"[ERROR] {e}")

    print("\nDone. All drawings processed.")


if __name__ == "__main__":
    main()
