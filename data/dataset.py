import os
import json
import random
from typing import List
from PIL import Image
from torch.utils.data import Dataset

from utils.geometry import clamp_bbox_xywh
import config as config_module
from config import MIN_MAPPED_RATIO, CLASS_BOOST
from utils.distance_utils import load_distance_scores, layout_key_from_folder



class GraphRTDetrDataset(Dataset):
    def __init__(self, folders: List[str], processor, hierarchy_json_path: str, augment=False):
        self.augment = augment

        import hierarchy_config as hier

        self.samples = []
        self.roi_cover_ratio = []
        self.distance_scores = []
        distance_data = load_distance_scores(config_module.DISTANCE_SCORE_FILE)
        for folder in folders:
            #img = os.path.join(folder, "model_baked.png")
            img = os.path.join(folder, config_module.IMAGE_FILENAME)

            
            graph = os.path.join(folder, "graph.json")
            if not (os.path.exists(img) and os.path.exists(graph)):
                continue

            with open(graph, "r") as f:
                data = json.load(f)

            with Image.open(img) as im:
                img_w, img_h = im.size

            img_classes = set()
            mapped_count = 0
            valid_count = 0
            total_count = 0
            total_roi_area = 0.0

            for node in data.get("nodes", []):
                bbox = node.get("bbox")
                if not bbox:
                    continue
                total_count += 1
                clamped = clamp_bbox_xywh(bbox, img_w, img_h)
                if clamped is None:
                    continue
                x, y, w, h = clamped
                valid_count += 1

                raw_label = self._extract_raw_category(node)
                l2 = hier.map_raw_to_l2(raw_label)
                if l2 is not None:
                    img_classes.add(l2)
                    mapped_count += 1
                    total_roi_area += w * h

            if total_count > config_module.MAX_NODES:
                continue
            if mapped_count == 0:
                continue
            if valid_count > 0 and (mapped_count / valid_count) < MIN_MAPPED_RATIO:
                continue

            self.samples.append((img, graph, img_classes))
            ratio = 0.0
            if img_w > 0 and img_h > 0:
                ratio = min(total_roi_area / (img_w * img_h), 1.0)
            self.roi_cover_ratio.append(ratio)
            layout_key = layout_key_from_folder(folder)
            self.distance_scores.append(distance_data.get(layout_key, 0.0))

        if len(self.samples) == 0:
            raise RuntimeError("Found zero valid samples.")

        self.image_processor = processor
        self.level2_classes, self.raw_to_l2 = hier.load_level2_classes_and_mapping(hierarchy_json_path)
        self.label2id = {c: i for i, c in enumerate(self.level2_classes)}
        self.id2label = {i: c for c, i in self.label2id.items()}

        self.image_classes = []
        self.image_weights = []

        for _, _, img_classes in self.samples:
            self.image_classes.append(img_classes)
            w = 1.0
            for c in img_classes:
                if c in CLASS_BOOST:
                    w = max(w, CLASS_BOOST[c])
            self.image_weights.append(w)

        self.map_raw_to_l2 = hier.map_raw_to_l2

    def __len__(self):
        return len(self.samples)

    def _extract_raw_category(self, node):
        raw = node.get("data_class", "") or node.get("category", "")
        return str(raw).strip()

    def _maybe_jitter_bbox(self, bbox, img_w: float, img_h: float):
        if not self.augment:
            return bbox
        if not (
            config_module.AUGMENT_BOX_JITTER_ENABLE
            or config_module.AUGMENT_BOX_EXPAND_RATIO > 0.0
        ):
            return bbox
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        if config_module.AUGMENT_BOX_JITTER_ENABLE and config_module.AUGMENT_BOX_JITTER_PX > 0.0:
            jitter = float(config_module.AUGMENT_BOX_JITTER_PX)
            cx += random.uniform(-jitter, jitter)
            cy += random.uniform(-jitter, jitter)

        scale = 1.0
        if config_module.AUGMENT_BOX_JITTER_ENABLE and config_module.AUGMENT_BOX_JITTER_SCALE > 0.0:
            s = float(config_module.AUGMENT_BOX_JITTER_SCALE)
            scale *= 1.0 + random.uniform(-s, s)

        if config_module.AUGMENT_BOX_EXPAND_RATIO > 0.0:
            scale *= 1.0 + float(config_module.AUGMENT_BOX_EXPAND_RATIO)

        w = max(w * scale, 1.0)
        h = max(h * scale, 1.0)
        x = cx - w / 2.0
        y = cy - h / 2.0

        clamped = clamp_bbox_xywh([x, y, w, h], img_w, img_h)
        if clamped is None:
            return None
        return clamped

    def __getitem__(self, idx: int):
        img_path, graph_path, _ = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        with open(graph_path, "r") as f:
            graph = json.load(f)

        anns = []
        ann_id = 0
        for node in graph.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue
            clamped = clamp_bbox_xywh(bbox, img_w, img_h)
            if clamped is None:
                continue
            jittered = self._maybe_jitter_bbox(clamped, img_w, img_h)
            if jittered is None:
                continue
            x, y, w, h = jittered

            raw = self._extract_raw_category(node)
            l2 = self.map_raw_to_l2(raw)
            if l2 not in self.label2id:
                continue

            anns.append({
                "bbox": [x, y, w, h],
                "category_id": self.label2id[l2],
                "area": float(w * h),
                "iscrowd": 0,
                "id": ann_id,
            })
            ann_id += 1

        target = {"image_id": idx, "annotations": anns}

        enc = self.image_processor(images=img, annotations=target, return_tensors="pt")

        item = {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
        }
        if "pixel_mask" in enc:
            item["pixel_mask"] = enc["pixel_mask"].squeeze(0)
        return item


def apply_subset(dataset, max_items: int):
    if not max_items or max_items <= 0:
        return
    dataset.samples = dataset.samples[:max_items]
    if hasattr(dataset, "image_classes"):
        dataset.image_classes = dataset.image_classes[:max_items]
    if hasattr(dataset, "image_weights"):
        dataset.image_weights = dataset.image_weights[:max_items]
    if hasattr(dataset, "roi_cover_ratio"):
        dataset.roi_cover_ratio = dataset.roi_cover_ratio[:max_items]
    if hasattr(dataset, "distance_scores"):
        dataset.distance_scores = dataset.distance_scores[:max_items]
