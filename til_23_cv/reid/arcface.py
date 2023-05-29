"""Implementation of ArcFace.

Inspired by
https://www.kaggle.com/code/zeta1996/pytorch-lightning-arcface-focal-loss.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ArcMarginProduct"]


class ArcMarginProduct(nn.Module):
    """ArcMarginProduct logits.

    Based on "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
    see https://arxiv.org/abs/1801.07698.
    """

    def __init__(self, feat_dim: int, nclasses: int, s: float = 32.0, m: float = 0.5):
        """Initialize ArcMarginProduct.

        Args:
            feat_dim (int): Size of features.
            nclasses (int): Number of classes.
            s (float, optional): Logit scaling factor. Defaults to 32.0.
            m (float, optional): Angular margin. Defaults to 0.5.
        """
        super(ArcMarginProduct, self).__init__()

        self.feat_dim = feat_dim
        self.nclasses = nclasses
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(nclasses, feat_dim))
        )

        # Some precalculations.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, feats: torch.Tensor, tgts: torch.Tensor, eps=1e-8
    ) -> torch.Tensor:
        """Calculate ArcMarginProduct logits.

        Args:
            feats (torch.Tensor): BC features.
            tgts (torch.Tensor): BN class targets.
            eps (float, optional): Epsilon. Defaults to 1e-8.

        Returns:
            torch.Tensor: BN logits.
        """
        # BC -> BN
        cosine = F.linear(
            F.normalize(feats, eps=eps), F.normalize(self.weight, eps=eps)
        )
        sine = (1.0 - cosine.square()).clamp(eps).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.where(cosine > self.th, cosine - self.mm)
        logits = self.s * (tgts * phi + (1.0 - tgts) * cosine)
        return logits
