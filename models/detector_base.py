from abc import ABC, abstractmethod

import torch


class DetectorBase(torch.nn.Module, ABC):
    detector_type = "base"

    @abstractmethod
    def forward_train(self, batch):
        """Return a scalar training loss tensor."""
        raise NotImplementedError

    @abstractmethod
    def forward_eval(self, batch):
        """Return model outputs for evaluation/inference."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to a directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: torch.device):
        """Load model from a directory onto device."""
        raise NotImplementedError
