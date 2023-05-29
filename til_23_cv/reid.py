"""Suspect Recognition model."""

from typing import List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# from torchvision.ops import sigmoid_focal_loss

__all__ = ["LitImEncoder"]


class LitImEncoder(pl.LightningModule):
    """LitImEncoder."""

    def __init__(
        self,
        model_name: str,
        head: Optional[nn.Module] = None,
        nclasses: int = -1,
        pretrained: bool = True,
        im_size: int = 224,
    ):
        """Initialize LitImEncoder.

        ``head`` and ``nclasses`` are only required during training. ``head`` should
        accept BC features and BN class targets (e.g., `til_23_cv.ArcMarginProduct`).

        Args:
            model_name (str): Backbone model from timm.
            head (nn.Module, optional): Training head. Defaults to None.
            nclasses (int, optional): Number of classes during training. Defaults to -1.
            pretrained (bool, optional): Use pretrained backbone. Defaults to True.
            im_size (int, optional): Image size. Defaults to 224.
        """
        super(LitImEncoder, self).__init__()

        # See https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/vision_transformer.py#L387
        # for config options.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=-1,
            global_pool="token",
            # NOTE: timm only supports pos encoding interpolation at init.
            img_size=im_size,
        )

        self.head = head
        self.nclasses = nclasses
        # TODO: Calculate class balance and provide `weight` to `nn.CrossEntropyLoss`.
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        # TODO: Try focal Loss to deal with class imbalance.
        # loss = sigmoid_focal_loss(logits, tgts, reduction="mean")

        # (embeddings, label) produced during testing.
        self._eval_embeds: List[np.ndarray] = []
        self._eval_lbls: List[int] = []

    def forward(self, ims: torch.Tensor):
        """Forward pass."""
        return self.model(ims)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step.

        ``batch`` is a tuple of images and class ids. Images should be BCHW and
        class ids should be B.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of images and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        assert self.head is not None and self.nclasses > 0
        ims, lbls = batch

        # BCHW -> BC
        feats = self.forward(ims)
        # TODO: Consider label smoothing.
        # B -> BN
        tgts = F.one_hot(lbls.long(), self.nclasses).type_as(feats)
        logits = self.head(feats, tgts)

        loss = self.loss(logits, lbls)
        acc = (logits.argmax(dim=1) == lbls).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        # TODO: What is Lightning's hparam system?
        lr = 5e-5
        # See https://github.com/Lightning-AI/lightning/issues/3095 on how to
        # change optimizer/scheduler midtraining for multi-stage finetune.
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = OneCycleLR(optimizer, lr, total_steps=10000)
        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step.

        ``batch`` is a tuple of images and class ids. Images should be BCHW and
        class ids should be B.

        Due to the nature of cluster-based evaluation, the actual validation logic
        is in ``on_validation_epoch_end``.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of images and labels.
            batch_idx (int): Batch index.
        """
        assert self.head is not None and self.nclasses > 0
        ims, lbls = batch
        # BCHW -> BC
        feats = self.forward(ims)
        for feat, lbl in zip(feats, lbls):
            self._eval_embeds.append(feat.numpy(force=True))
            self._eval_lbls.append(int(lbl))

    def on_validation_epoch_end(self):
        """Actual Validation Logic."""
        x = np.stack(self._eval_embeds)
        y = np.array(self._eval_lbls)
        if len(np.unique(y)) > 1:
            score: float = silhouette_score(x, y, metric="cosine")  # type: ignore
            self.log("val_sil_score", score, prog_bar=True)
        self._eval_embeds.clear()
        self._eval_lbls.clear()
