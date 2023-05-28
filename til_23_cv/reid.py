"""Suspect Recognition model."""

from typing import Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# from torchvision.ops import sigmoid_focal_loss

__all__ = ["LitImEncoder"]


class LitImEncoder(pl.LightningModule):
    """LitImEncoder."""

    def __init__(
        self, model: nn.Module, head: Optional[nn.Module] = None, nclasses: int = -1
    ):
        """Initialize LitImEncoder.

        ``head`` and ``nclasses`` are only required during training. ``head`` should
        accept BC features and BN class targets (e.g., `til_23_cv.ArcMarginProduct`).

        Args:
            model (nn.Module): Backbone model.
            head (nn.Module, optional): Training head. Defaults to None.
            nclasses (int, optional): Number of classes during training. Defaults to -1.
        """
        super(LitImEncoder, self).__init__()

        self.model = model
        self.head = head
        self.nclasses = nclasses
        # TODO: Calculate class balance and provide `weight` to `nn.CrossEntropyLoss`.
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        # TODO: Try focal Loss to deal with class imbalance.
        # loss = sigmoid_focal_loss(logits, tgts, reduction="mean")

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
        feats = self.model(ims)
        # TODO: Consider label smoothing.
        # B -> BN
        tgts = F.one_hot(lbls.long(), self.nclasses).type_as(feats)
        logits = self.head(feats, tgts)

        loss = self.loss(logits, lbls)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        # TODO: Make this actually configurable.
        lr = 1e-3
        # See https://github.com/Lightning-AI/lightning/issues/3095 on how to
        # change optimizer/scheduler midtraining for multi-stage finetune.
        optimizer = Adam(self.parameters(), lr=lr)
        return optimizer
