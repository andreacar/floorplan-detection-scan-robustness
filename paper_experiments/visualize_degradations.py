#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_experiments.common import load_image, load_test_dirs  # noqa: E402
from paper_experiments.factorized_degradation import DEGRADATIONS, LEVELS  # noqa: E402


def _resize_with_letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((nw, nh), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _find_sample_by_id(dirs: List[str], sample_id: str) -> Optional[str]:
    for d in dirs:
        if os.path.basename(d.rstrip("/")) == sample_id:
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize factorized degradations on one sample.")
    parser.add_argument("--sample-folder", default=None, help="Path to one sample folder.")
    parser.add_argument("--sample-id", default=None, help="Folder name to match from split list.")
    parser.add_argument("--image-name", default="model_baked.png", help="Image filename in sample folder.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--out", default="paper_experiments/out/degradation_grid.png", help="Output image path.")
    parser.add_argument("--cell-size", type=int, default=320, help="Size (px) of each cell.")
    parser.add_argument("--factors", default="blur,thicken,texture,clutter", help="Comma-separated factors.")
    parser.add_argument("--levels", default="mild,medium,strong", help="Comma-separated levels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unsupported level: {level}")
    for factor in factors:
        if factor not in DEGRADATIONS:
            raise ValueError(f"Unknown factor: {factor}")

    if args.sample_folder:
        folder = args.sample_folder
    else:
        dirs = load_test_dirs(args.test_txt)
        if args.sample_id:
            folder = _find_sample_by_id(dirs, args.sample_id)
        else:
            folder = dirs[0] if dirs else None

    if not folder:
        raise RuntimeError("Could not resolve a sample folder. Provide --sample-folder or --sample-id.")

    img_path = os.path.join(folder, args.image_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    rng = np.random.default_rng(args.seed)
    clean = load_image(img_path)
    clean_cell = _resize_with_letterbox(clean, args.cell_size)

    cols = ["clean"] + levels
    rows = factors
    pad = 10
    label_h = 26
    label_w = 90

    canvas_w = label_w + pad + len(cols) * (args.cell_size + pad)
    canvas_h = label_h + pad + len(rows) * (args.cell_size + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for ci, col in enumerate(cols):
        x = label_w + pad + ci * (args.cell_size + pad)
        draw.text((x + 4, 4), col, fill=(0, 0, 0), font=font)

    for ri, factor in enumerate(rows):
        y = label_h + pad + ri * (args.cell_size + pad)
        draw.text((4, y + 4), factor, fill=(0, 0, 0), font=font)

        # Clean cell (first column)
        x0 = label_w + pad
        canvas.paste(clean_cell, (x0, y))

        # Degraded cells
        for ci, level in enumerate(levels, start=1):
            degrade_fn = DEGRADATIONS[factor]
            degraded = degrade_fn(clean, level, rng)
            cell = _resize_with_letterbox(degraded, args.cell_size)
            x = label_w + pad + ci * (args.cell_size + pad)
            canvas.paste(cell, (x, y))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved grid to {args.out}")


if __name__ == "__main__":
    main()
