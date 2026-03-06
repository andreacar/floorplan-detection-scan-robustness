"""
Run with:
python -m diagnostic.compare_visualization
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import os
import json
import random
import sys
from PIL import Image, ImageDraw, ImageFont
import yaml

from utils.geometry import clamp_bbox_xywh
import hierarchy_config as hier

# ===============================================================
# PATH FIXES
# ===============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # diagnostic/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # project root
sys.path.append(PROJECT_ROOT)

def _load_cfg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    os.environ.setdefault("CUBICASA_CONFIG", args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


cfg = _load_cfg()
import config  # noqa: E402

# ===============================================================
# CONFIG
# ===============================================================
ROOT_DATA = cfg["root_data"]

IMG_LEFT = "F1_scaled.png"
IMG_RIGHT = "model_baked.png"

GRAPH_NAME = "graph.json"
HIER_JSON = cfg["hier_json"]
N_SAMPLES = 100

CONCAT_IMAGES = True
ADD_CLASS_COLOR_BAND = True   # <<< TOGGLE HERE
SAVE_SCALE_REPORT = True
SAVE_OVERLAY = True
OVERLAY_ALPHA = 0.5
OVERLAY_WITH_BOXES = False
APPLY_PAIR_SCALE = True
PAIRS_JSON = "paper_experiments/out/hq_pairs/pairs.json"

# ===============================================================
# OUTPUT DIR
# ===============================================================
OUT_DIR = os.path.join(CURRENT_DIR, "comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_COLORS = config.CLASS_COLORS
FILL_ALPHA = 66  # 0.5 opacity (0–255)
SCALE_REPORT_PATH = os.path.join(OUT_DIR, "scale_report.json")

# ===============================================================
# LEGEND BAND (CENTERED, READABLE)
# ===============================================================
def add_centered_class_color_band(
    image: Image.Image,
    class_colors: dict,
    enable: bool = True,
    band_height: int = 90,        # bigger → readable in print
    circle_radius: int = 14,
    padding: int = 30,
):
    if not enable:
        return image

    w, h = image.size
    new_img = Image.new("RGB", (w, h + band_height), (255, 255, 255))
    new_img.paste(image.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # -----------------------------------------------------------
    # Compute total legend width (for centering)
    # -----------------------------------------------------------
    item_widths = []
    for cls in class_colors.keys():
        text_w = draw.textlength(cls, font=font)
        item_w = 2 * circle_radius + 8 + text_w + padding
        item_widths.append(item_w)

    total_legend_width = sum(item_widths)
    x = (w - total_legend_width) // 2
    y = h + band_height // 2

    # -----------------------------------------------------------
    # Draw legend
    # -----------------------------------------------------------
    for cls, color in class_colors.items():
        # Circle
        draw.ellipse(
            (
                x,
                y - circle_radius,
                x + 2 * circle_radius,
                y + circle_radius,
            ),
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )

        # Text
        text_x = x + 2 * circle_radius + 8
        text_y = y - 10
        draw.text((text_x, text_y), cls, fill=(0, 0, 0), font=font)

        text_w = draw.textlength(cls, font=font)
        x += 2 * circle_radius + 8 + text_w + padding

    return new_img


def _blend_images(left_img, right_img, alpha):
    if left_img.size != right_img.size:
        left_area = left_img.size[0] * left_img.size[1]
        right_area = right_img.size[0] * right_img.size[1]
        if left_area >= right_area:
            right_img = right_img.resize(left_img.size, Image.BILINEAR)
        else:
            left_img = left_img.resize(right_img.size, Image.BILINEAR)
    return Image.blend(left_img.convert("RGB"), right_img.convert("RGB"), alpha)


def _scale_and_fit(img, scale, target_size, fill=(255, 255, 255, 255)):
    if abs(scale - 1.0) < 1e-6 and img.size == target_size:
        return img
    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = img.resize((new_w, new_h), Image.BILINEAR)
    target_w, target_h = target_size
    if new_w == target_w and new_h == target_h:
        return scaled
    if new_w >= target_w and new_h >= target_h:
        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        return scaled.crop((x0, y0, x0 + target_w, y0 + target_h))
    canvas = Image.new(img.mode, (target_w, target_h), fill)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas.paste(scaled, (x0, y0))
    return canvas


def _norm_path(path):
    return os.path.normpath(path)

# ===============================================================
# LOAD HIERARCHY
# ===============================================================
level2_classes, raw_to_l2 = hier.load_level2_classes_and_mapping(HIER_JSON)
label2id = {c: i for i, c in enumerate(level2_classes)}

# ===============================================================
# COLLECT SAMPLES
# ===============================================================
all_dirs = [
    os.path.join(ROOT_DATA, d)
    for d in os.listdir(ROOT_DATA)
    if os.path.isdir(os.path.join(ROOT_DATA, d))
]

random.shuffle(all_dirs)
selected_dirs = all_dirs[:N_SAMPLES]

print(f"Selected {len(selected_dirs)} folders")
scale_reports = []

pairs_map = {}
if APPLY_PAIR_SCALE and PAIRS_JSON:
    pairs_path = PAIRS_JSON
    if not os.path.isabs(pairs_path):
        pairs_path = os.path.join(PROJECT_ROOT, pairs_path)
    if os.path.exists(pairs_path):
        try:
            with open(pairs_path, "r") as f:
                pairs_data = json.load(f)
            for row in pairs_data:
                folder = row.get("folder")
                if folder:
                    pairs_map[_norm_path(folder)] = row
        except Exception as exc:
            print(f"Failed to load pairs.json ({pairs_path}): {exc}")
    else:
        print(f"pairs.json not found at {pairs_path}; falling back to size-based overlay.")

# ===============================================================
# PROCESS
# ===============================================================
for folder in selected_dirs:
    img_left_path = os.path.join(folder, IMG_LEFT)
    img_right_path = os.path.join(folder, IMG_RIGHT)
    graph_path = os.path.join(folder, GRAPH_NAME)

    if not (os.path.exists(img_left_path) and os.path.exists(img_right_path) and os.path.exists(graph_path)):
        print(f"Skipping {folder} (missing files)")
        continue

    try:
        img_left = Image.open(img_left_path).convert("RGBA")
        img_right = Image.open(img_right_path).convert("RGBA")
        img_left_raw = img_left.copy()
        img_right_raw = img_right.copy()
    except Exception as e:
        print(f"Failed to open images in {folder}: {e}")
        continue

    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except Exception as e:
        print(f"Failed to load graph in {folder}: {e}")
        continue

    wl, hl = img_left.size
    wr, hr = img_right.size

    overlay_left = Image.new("RGBA", img_left.size, (0, 0, 0, 0))
    overlay_right = Image.new("RGBA", img_right.size, (0, 0, 0, 0))

    draw_left = ImageDraw.Draw(overlay_left)
    draw_right = ImageDraw.Draw(overlay_right)

    n_drawn = 0

    # -----------------------------------------------------------
    # Draw GT boxes
    # -----------------------------------------------------------
    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue

        raw = node.get("data_class", "") or node.get("category", "")
        raw = str(raw).strip()
        l2 = raw_to_l2.get(raw)

        if l2 not in label2id:
            continue

        r, g, b = CLASS_COLORS.get(l2, (255, 0, 0))
        fill = (r, g, b, FILL_ALPHA)
        outline = (r, g, b, 255)

        cl_left = clamp_bbox_xywh(bbox, wl, hl)
        if cl_left is not None:
            x, y, w, h = cl_left
            draw_left.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=2)

        cl_right = clamp_bbox_xywh(bbox, wr, hr)
        if cl_right is not None:
            x, y, w, h = cl_right
            draw_right.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=2)

        n_drawn += 1

    if n_drawn == 0:
        print(f"No valid annotations drawn for {folder}")
        continue

    img_left = Image.alpha_composite(img_left, overlay_left)
    img_right = Image.alpha_composite(img_right, overlay_right)

    base = os.path.basename(folder.rstrip("/"))

    if SAVE_OVERLAY:
        overlay_left = img_left if OVERLAY_WITH_BOXES else img_left_raw
        overlay_right = img_right if OVERLAY_WITH_BOXES else img_right_raw
        pair = pairs_map.get(_norm_path(folder))
        if APPLY_PAIR_SCALE and pair and pair.get("scale") is not None:
            overlay_left = _scale_and_fit(overlay_left, float(pair["scale"]), overlay_right.size)
        blended = _blend_images(overlay_left, overlay_right, OVERLAY_ALPHA)
        overlay_path = os.path.join(OUT_DIR, f"{base}_overlay.png")
        blended.save(overlay_path)
        print(f"Saved {overlay_path} (overlay)")

    # ===============================================================
    # STITCH + LEGEND
    # ===============================================================
    if CONCAT_IMAGES:
        scale_info = {
            "folder": folder,
            "left_size": [wl, hl],
            "right_size": [wr, hr],
            "scaled_side": "none",
            "scale": 1.0,
            "scaled_size": [wr, hr],
        }
        if hl >= hr:
            scale = hl / hr
            new_wr = int(wr * scale)
            img_right = img_right.resize((new_wr, hl), Image.BILINEAR)
            scale_info.update(
                {
                    "scaled_side": "right",
                    "scale": float(scale),
                    "scaled_size": [new_wr, hl],
                }
            )
            canvas = Image.new("RGBA", (wl + new_wr, hl))
            canvas.paste(img_left, (0, 0))
            canvas.paste(img_right, (wl, 0))
        else:
            scale = hr / hl
            new_wl = int(wl * scale)
            img_left = img_left.resize((new_wl, hr), Image.BILINEAR)
            scale_info.update(
                {
                    "scaled_side": "left",
                    "scale": float(scale),
                    "scaled_size": [new_wl, hr],
                }
            )
            canvas = Image.new("RGBA", (new_wl + wr, hr))
            canvas.paste(img_left, (0, 0))
            canvas.paste(img_right, (new_wl, 0))

        final_img = add_centered_class_color_band(
            canvas,
            CLASS_COLORS,
            enable=ADD_CLASS_COLOR_BAND,
        )

        out_path = os.path.join(OUT_DIR, f"{base}_comparison.png")
        final_img.save(out_path)
        print(f"Saved {out_path} ({n_drawn} boxes)")
        scale_reports.append(scale_info)

if SAVE_SCALE_REPORT and scale_reports:
    with open(SCALE_REPORT_PATH, "w") as f:
        json.dump(scale_reports, f, indent=2)
    print(f"Saved scale report to {SCALE_REPORT_PATH}")

print("Done.")
