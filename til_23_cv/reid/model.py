"""Suspect Recognition model."""

from typing import List, Mapping, Tuple

import lightning.pytorch as pl
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import BaseFinetuning
from sklearn.metrics import silhouette_score
from timm.optim import MADGRAD
from torch.optim.lr_scheduler import OneCycleLR

from .arcface import ArcMarginProduct

__all__ = ["LitArcEncoder", "UnfreezeCallback"]


class LitArcEncoder(pl.LightningModule):
    """LitArcEncoder."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        im_size: int = 224,
        nclasses: int = -1,
        arc_s: float = 32.0,
        arc_m: float = 0.5,
        lr: float = 5e-5,
        momentum: float = 0.9,
        decay: float = 0.0,
        sched_steps: int = -1,
    ):
        """Initialize LitArcEncoder.

        Args:
            model_name (str, optional): Backbone model from timm. Defaults to "vit_small_patch14_dinov2.lvd142m".
            pretrained (bool, optional): Use pretrained backbone. Defaults to True.
            im_size (int, optional): Image size. Defaults to 224.
            nclasses (int, optional): Number of classes during training. Defaults to -1.
            arc_s (float, optional): ArcFace logit scaling factor. Defaults to 32.0.
            arc_m (float, optional): ArcFace angular margin. Defaults to 0.5.
            lr (float, optional): Max learning rate. Defaults to 5e-5.
            momentum (float, optional): Optimizer momentum. Defaults to 0.9.
            decay (float, optional): Optimizer weight decay. Defaults to 0.0.
            sched_steps (int, optional): Number of steps for OneCycleLR. Defaults to -1.
        """
        super(LitArcEncoder, self).__init__()
        self.save_hyperparameters()

        self.nclasses = nclasses
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.sched_steps = sched_steps
        self.best_score = -1.0

        # See https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/vision_transformer.py#L387
        # for config options.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=-1,  # Needs to be -1 to disable FC layer.
            global_pool="token",  # Use cls token.
            # NOTE: timm only supports pos encoding interpolation at init.
            img_size=im_size,
            scriptable=True,
        )
        self.head = ArcMarginProduct(
            self.model.num_features, nclasses, s=arc_s, m=arc_m
        )
        # TODO: Calculate class balance and provide `weight` to `nn.CrossEntropyLoss`.
        self.loss = nn.CrossEntropyLoss(reduction="mean")

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
        assert self.nclasses > 0
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
        # See https://github.com/Lightning-AI/lightning/issues/3095 on how to
        # change optimizer/scheduler midtraining for multi-stage finetune.
        optimizer = MADGRAD(self.parameters(), self.lr, self.momentum, self.decay)
        if self.sched_steps == -1:
            return optimizer
        scheduler = OneCycleLR(optimizer, self.lr, total_steps=self.sched_steps)
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
        assert self.nclasses > 0
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
            # For tensorboard.
            self.log("hp_metric", score)
            self.best_score = max(self.best_score, score)
        self._eval_embeds.clear()
        self._eval_lbls.clear()


class UnfreezeCallback(BaseFinetuning):
    """Callback to freeze all layers at start and unfreeze during training."""

    def __init__(self, milestones: Mapping[int, List[int]]):
        """Initialize UnfreezeCallback.

        Args:
            milestones (Mapping[int, Union[int, List[int]]]): Mapping from epoch to layer indices to unfreeze.
        """
        super(UnfreezeCallback, self).__init__()
        self.milestones = milestones

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        excluded = [pl_module.model.blocks[i] for i in self.milestones.get(0, [])]
        self.freeze([l for l in pl_module.model.blocks if l not in excluded])

    def finetune_function(self, pl_module, epoch: int, optimizer):
        """Unfreeze layers during training."""
        if epoch == 0:
            return
        if epoch in self.milestones:
            layers = [pl_module.model.blocks[i] for i in self.milestones[epoch]]
            self.unfreeze_and_add_param_group(layers, optimizer)
