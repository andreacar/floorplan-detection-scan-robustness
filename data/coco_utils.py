import json
from PIL import Image
from tqdm import tqdm
import torch
from utils.geometry import clamp_bbox_xywh


def build_coco_groundtruth(dataset, out_path: str):
    images = []
    annotations = []
    ann_id = 0

    for idx, (img_path, graph_path, _) in enumerate(dataset.samples):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        images.append({
            "id": idx,
            "width": w,
            "height": h,
            "file_name": img_path.split("/")[-1],
        })

        with open(graph_path, "r") as f:
            data = json.load(f)

        for node in data["nodes"]:
            bbox = node.get("bbox")
            if not bbox:
                continue

            clamped = clamp_bbox_xywh(bbox, w, h)
            if clamped is None:
                continue
            x, y, bw, bh = clamped

            raw = node.get("data_class", "") or node.get("category", "")
            l2 = dataset.map_raw_to_l2(raw)
            if l2 not in dataset.label2id:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "category_id": dataset.label2id[l2],
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [
        {"id": dataset.label2id[name], "name": name}
        for name in dataset.level2_classes
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(out_path, "w") as f:
        json.dump(coco_dict, f)


def rtdetr_to_coco_json(model, dataset, device, json_path: str):
    model.eval()
    results = []

    for idx, (img_path, _, _) in enumerate(tqdm(dataset.samples, desc="COCO eval export")):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        enc = dataset.image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)

        post = dataset.image_processor.post_process_object_detection(
            out,
            threshold=0.0,
            target_sizes=torch.tensor([[h, w]], device=device),
        )[0]

        boxes = post["boxes"]  # xyxy
        scores = post["scores"]
        labels = post["labels"]

        # clamp
        boxes[:, 0] = boxes[:, 0].clamp(0, w)
        boxes[:, 2] = boxes[:, 2].clamp(0, w)
        boxes[:, 1] = boxes[:, 1].clamp(0, h)
        boxes[:, 3] = boxes[:, 3].clamp(0, h)

        # valid boxes only
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        for box, score, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": idx,
                "category_id": int(lab.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item()),
            })

    with open(json_path, "w") as f:
        json.dump(results, f)
