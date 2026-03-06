from typing import Optional

import torch
from transformers import RTDetrForObjectDetection

from .detector_base import DetectorBase


class RTDetrDetector(DetectorBase):
    detector_type = "rtdetr"

    def __init__(
        self,
        *,
        device: torch.device,
        model: Optional[RTDetrForObjectDetection] = None,
        backbone: Optional[str] = None,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        init_weights_dir: Optional[str] = None,
        no_object_weight: Optional[float] = None,
    ):
        super().__init__()

        if model is None:
            if init_weights_dir is None:
                if backbone is None or id2label is None or label2id is None:
                    raise ValueError("backbone/id2label/label2id required for fresh RT-DETR init.")
                model = RTDetrForObjectDetection.from_pretrained(
                    backbone,
                    num_labels=len(id2label),
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                    num_queries=100,
                )
            else:
                model = RTDetrForObjectDetection.from_pretrained(init_weights_dir)
                if id2label is not None:
                    model.config.id2label = id2label
                if label2id is not None:
                    model.config.label2id = label2id

        if no_object_weight is not None:
            model.config.no_object_weight = no_object_weight

        self.model = model
        self.device = device
        self.model.to(device)

    def forward_train(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)
        labels = [{k: v.to(self.device) for k, v in lab.items()} for lab in batch["labels"]]

        out = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return out.loss

    def forward_eval(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: torch.device):
        model = RTDetrForObjectDetection.from_pretrained(path)
        return cls(model=model, device=device)
