"""Utilities."""

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2

__all__ = ["ReIDEncoder", "cos_sim"]

BORDER_MODE = cv2.BORDER_REPLICATE
# BORDER_MODE = cv2.BORDER_CONSTANT

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

IM_SIZE = 224


# TODO: This is a bad practice, but I have to get over my code OCD.
class ReIDEncoder(nn.Module):
    """Convenience class for ReID encoder."""

    # TODO: Debug torchscript not working when device is different.
    def __init__(self, model_path: str, device="cpu"):
        """Initialize ReIDEncoder."""
        super(ReIDEncoder, self).__init__()
        assert "torchscript" in model_path, "I am lazy."
        traced = torch.jit.load(model_path)
        self.encoder = torch.jit.optimize_for_inference(traced)
        self.normalize = A.Compose(
            [
                A.LongestMaxSize(max_size=IM_SIZE, interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(
                    min_height=IM_SIZE, min_width=IM_SIZE, border_mode=BORDER_MODE
                ),
                A.Normalize(mean=RGB_MEAN, std=RGB_STD),
                ToTensorV2(),
            ]
        )
        self.device = device
        self.to(device)

    def forward(self, ims: np.ndarray) -> torch.Tensor:
        """Forward pass."""
        x = torch.stack([self.normalize(image=im)["image"] for im in ims])
        x = x.to(self.device)
        return self.encoder(x).numpy(force=True)


def cos_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
