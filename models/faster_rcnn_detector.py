import json
import os
from typing import Optional

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import (
    RESIZE_FIXED,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    RESIZE_SHORTEST_EDGE,
    RESIZE_LONGEST_EDGE,
)

from .detector_base import DetectorBase


class FasterRCNNDetector(DetectorBase):
    detector_type = "fasterrcnn"

    def __init__(
        self,
        *,
        num_classes: int,
        device: torch.device,
        init_weights_dir: Optional[str] = None,
        model=None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.device = device

        if model is None:
            if init_weights_dir:
                model = self._build_model(self.num_classes, pretrained=False)
                state_path = os.path.join(init_weights_dir, "fasterrcnn_model.pth")
                state = torch.load(state_path, map_location=device)
                model.load_state_dict(state)
            else:
                model = self._build_model(self.num_classes, pretrained=True)

        self.model = model
        self.model.to(device)

    @staticmethod
    def _build_model(num_classes: int, pretrained: bool):
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        except TypeError:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            model = fasterrcnn_resnet50_fpn(weights=weights)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Inputs are already resized/normalized by the RT-DETR processor.
        model.transform.image_mean = [0.0, 0.0, 0.0]
        model.transform.image_std = [1.0, 1.0, 1.0]

        if RESIZE_FIXED:
            model.transform.min_size = (int(RESIZE_HEIGHT),)
            model.transform.max_size = int(RESIZE_WIDTH)
        else:
            model.transform.min_size = (int(RESIZE_SHORTEST_EDGE),)
            model.transform.max_size = int(RESIZE_LONGEST_EDGE)

        return model

    def forward_train(self, batch):
        images = [img.to(self.device) for img in batch["images"]]
        targets = []
        for tgt in batch["targets"]:
            moved = {}
            for key, val in tgt.items():
                moved[key] = val.to(self.device) if isinstance(val, torch.Tensor) else val
            targets.append(moved)

        was_training = self.model.training
        if not was_training:
            self.model.train()

        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        if not was_training:
            self.model.eval()

        return loss

    def forward_eval(self, batch):
        images = [img.to(self.device) for img in batch["images"]]
        return self.model(images)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "fasterrcnn_model.pth"))
        meta = {"num_classes": self.num_classes}
        with open(os.path.join(path, "fasterrcnn_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str, device: torch.device):
        meta_path = os.path.join(path, "fasterrcnn_meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        num_classes = int(meta["num_classes"])
        model = cls._build_model(num_classes, pretrained=False)
        state = torch.load(os.path.join(path, "fasterrcnn_model.pth"), map_location=device)
        model.load_state_dict(state)
        return cls(num_classes=num_classes, device=device, model=model)
