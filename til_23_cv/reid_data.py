"""Suspect Recognition data module."""

import os
from typing import List, Tuple

import albumentations as A
import cv2
import lightning.pytorch as pl
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

__all__ = ["LitImClsDataModule"]

BORDER_MODE = cv2.BORDER_REPLICATE
# BORDER_MODE = cv2.BORDER_CONSTANT

# Using similar augments as Object Detection.
DEFAULT_TRANSFORMS = [
    A.Affine(
        p=0.5,
        rotate=(-90, 90),
        shear=(-8, 8),
        interpolation=cv2.INTER_CUBIC,
        mode=BORDER_MODE,
        fit_output=False,
    ),
    # Train images have 0.5 padding.
    A.CropAndPad(
        p=0.9,
        percent=(-1 / 6, -0.08),
        pad_mode=BORDER_MODE,
        sample_independently=True,
        interpolation=cv2.INTER_CUBIC,
    ),
    A.Flip(p=0.6),
    A.AdvancedBlur(
        p=0.1, blur_limit=(5, 11), noise_limit=(0.0, 2.0), beta_limit=(0.0, 4.0)
    ),
    A.CLAHE(p=0.1),
    A.MotionBlur(p=0.1, blur_limit=(5, 11)),
    A.ColorJitter(p=0.7, brightness=0.4, contrast=0.0, saturation=0.7, hue=0.01),
    A.ImageCompression(p=0.1, quality_lower=20, quality_upper=50),
    A.GaussNoise(p=0.2, per_channel=True, var_limit=(500.0, 3000.0)),
    A.ISONoise(p=0.2, intensity=(0.1, 0.5), color_shift=(0.03, 0.06)),
]


class LitImClsDataModule(pl.LightningDataModule):
    """LitImClsDataModule."""

    def __init__(
        self,
        data_dir: str,
        im_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize LitImClsDataModule."""
        super(LitImClsDataModule, self).__init__()

        self.im_size = im_size
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.test_dir = os.path.join(self.data_dir, "test")

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.loader_cfg = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.eval_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=self.im_size, interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(
                    min_height=self.im_size,
                    min_width=self.im_size,
                    border_mode=BORDER_MODE,
                ),
            ]
        )
        self.to_tensor = A.Compose(
            [
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        self.train_transform = A.Compose(DEFAULT_TRANSFORMS)

    def _transform(self, im, is_train):
        im = np.array(im)
        a = self.eval_transform(image=im)
        if is_train:
            a = self.train_transform(**a)
        a = self.to_tensor(**a)
        return a["image"]

    def _eval_transform(self, im):
        return self._transform(im, is_train=False)

    def _train_transform(self, im):
        return self._transform(im, is_train=True)

    def preview_transform(self, num_samples: int = 1) -> List[Image.Image]:
        """Preview augmentations used."""
        ds = ImageFolder(self.train_dir)
        loader = DataLoader(ds, shuffle=True, batch_size=None)
        ims = []
        for im, _ in loader:
            im = np.array(im)
            a = self.eval_transform(image=im)
            a = self.train_transform(**a)
            ims.append(Image.fromarray(a["image"]))
            if len(ims) >= num_samples:
                break
        return ims

    def setup(self, stage: str):
        """Setup data module."""
        if os.path.isdir(self.train_dir) and stage == "fit":
            self.train_ds = ImageFolder(self.train_dir, transform=self._train_transform)
        if os.path.isdir(self.val_dir) and stage == "fit":
            self.val_ds = ImageFolder(self.val_dir, transform=self._eval_transform)
        if os.path.isdir(self.test_dir) and stage != "fit":
            self.test_ds = ImageFolder(self.test_dir, transform=self._eval_transform)

    def train_dataloader(self):
        """Train dataloader."""
        assert self.train_ds is not None
        return DataLoader(self.train_ds, shuffle=True, **self.loader_cfg)

    def val_dataloader(self):
        """Validation dataloader."""
        assert self.val_ds is not None
        return DataLoader(self.val_ds, shuffle=False, **self.loader_cfg)

    def test_dataloader(self):
        """Test dataloader."""
        assert self.test_ds is not None
        return DataLoader(self.test_ds, shuffle=False, **self.loader_cfg)
