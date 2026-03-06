import torch


def _labels_to_detection_target(labels, image_shape, label_offset: int):
    boxes = labels["boxes"]
    if boxes.numel() == 0:
        boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.zeros((0,), dtype=torch.int64)
    else:
        if "size" in labels:
            h, w = labels["size"].tolist()
        else:
            h, w = image_shape[-2:]

        # RT-DETR labels are normalized cxcywh; convert to absolute xyxy.
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - 0.5 * bw) * w
        y1 = (cy - 0.5 * bh) * h
        x2 = (cx + 0.5 * bw) * w
        y2 = (cy + 0.5 * bh) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes_xyxy[:, 0].clamp_(0, w)
        boxes_xyxy[:, 2].clamp_(0, w)
        boxes_xyxy[:, 1].clamp_(0, h)
        boxes_xyxy[:, 3].clamp_(0, h)
        class_labels = labels["class_labels"].to(torch.int64) + int(label_offset)

    target = {
        "boxes": boxes_xyxy.to(torch.float32),
        "labels": class_labels,
    }
    if "image_id" in labels:
        target["image_id"] = labels["image_id"]
    if "area" in labels:
        target["area"] = labels["area"].to(torch.float32)
    if "iscrowd" in labels:
        target["iscrowd"] = labels["iscrowd"]

    return target


def make_collate_fn(processor):
    def c(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        pixel_mask = None
        if "pixel_mask" in batch[0]:
            pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
        else:
            # No padding: build a full-ones mask so RT-DETR gets a valid mask.
            pixel_mask = torch.ones(
                (pixel_values.shape[0], pixel_values.shape[-2], pixel_values.shape[-1]),
                dtype=torch.long,
            )
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": [b["labels"] for b in batch],
        }
    return c


def make_fasterrcnn_collate_fn():
    def c(batch):
        images = [b["pixel_values"] for b in batch]
        targets = [
            _labels_to_detection_target(b["labels"], b["pixel_values"].shape, label_offset=1)
            for b in batch
        ]
        return {"images": images, "targets": targets}

    return c


def make_retinanet_collate_fn():
    def c(batch):
        images = [b["pixel_values"] for b in batch]
        targets = [
            _labels_to_detection_target(b["labels"], b["pixel_values"].shape, label_offset=0)
            for b in batch
        ]
        return {"images": images, "targets": targets}

    return c
