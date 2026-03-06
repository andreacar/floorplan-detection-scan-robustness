import argparse
from pathlib import Path

import torch
from PIL import Image

from utils.inference_helpers import (
    DEFAULT_IMAGE_NAME,
    draw_predictions,
    infer_image,
    load_model_and_processor,
    resolve_input_image,
    resolve_split_drawing,
    save_prediction_bundle,
)


def _load_entries(split_file: str):
    with open(split_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run RT-DETR inference on drawing folders listed in a split file.")
    parser.add_argument("--checkpoint-dir", required=True, help="HF-style checkpoint directory (for example checkpoints/best).")
    parser.add_argument("--split-file", required=True, help="Split file listing drawing folders, one per line.")
    parser.add_argument("--base-dir", default="", help="Dataset base directory used to resolve relative split entries.")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME, help="Image filename to use inside each drawing folder.")
    parser.add_argument("--out-dir", default="outputs/inference_split", help="Directory for overlay and JSON outputs.")
    parser.add_argument("--score-thresh", type=float, default=0.2, help="Score threshold passed to RT-DETR post-processing.")
    parser.add_argument("--topk", type=int, default=100, help="Maximum number of predictions to keep after ranking.")
    parser.add_argument("--max-items", type=int, default=20, help="Maximum number of drawings to process from the split.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--hide-labels", action="store_true", help="Draw boxes only, without text labels.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    processor, model = load_model_and_processor(args.checkpoint_dir, device)

    entries = _load_entries(args.split_file)
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for entry in entries:
        drawing_dir = resolve_split_drawing(args.base_dir, entry) if args.base_dir else Path(entry).expanduser().resolve()
        image_path = resolve_input_image(str(drawing_dir), args.image_name)
        image = Image.open(image_path).convert("RGB")
        predictions = infer_image(model, processor, image, device, args.score_thresh, args.topk)
        overlay = draw_predictions(image, predictions, show_labels=not args.hide_labels)

        rel_stem = entry.strip("/").replace("/", "__") or drawing_dir.name
        image_out, json_out = save_prediction_bundle(
            str(output_root),
            rel_stem,
            str(image_path),
            args.checkpoint_dir,
            args.score_thresh,
            args.topk,
            predictions,
            overlay,
        )
        print(f"[{processed + 1}] {image_path}")
        print(f"    overlay: {image_out}")
        print(f"    json:    {json_out}")
        processed += 1
        if args.max_items > 0 and processed >= args.max_items:
            break

    print(f"processed drawings: {processed}")


if __name__ == "__main__":
    main()
