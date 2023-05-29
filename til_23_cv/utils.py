"""Utilities."""

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2

__all__ = [
    "ReIDEncoder",
    "cos_sim",
    "thres_strategy_A",
    "thres_strategy_naive",
    "thres_strategy_softmax",
    "evaluate_threshold_function",
]

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


def thres_strategy_A(scores: list, accept_thres=0.3, vote_thres=0.0, sd_thres=4.4):
    """Strategy A.

    If any in ``scores`` is greater than ``accept_thres``, return the index of the
    max score. If any in ``scores`` is greater than ``vote_thres``, return the index
    of the max score if it is greater than ``sd`` standard deviations away from
    the mean of ``scores`` (excluding the max score). Otherwise, return -1.

    I am not aware of this strategy in any literature, so it is essentially my own
    delusion.

    Args:
        scores (List[float]): List of scores.
        accept_thres (float, optional): Threshold for accepting a prediction. Defaults to 0.7.
        vote_thres (float, optional): Threshold for voting. Defaults to 0.1.
        sd_thres (float, optional): Number of standard deviations away from the mean. Defaults to 5.0.
    """
    if np.max(scores) > accept_thres:
        return np.argmax(scores)
    elif np.max(scores) > vote_thres:
        scores = np.array(scores).clip(0.0)  # type: ignore
        mean = np.mean(scores[scores < np.max(scores)])
        std = np.std(scores[scores < np.max(scores)])
        if np.max(scores) - mean > sd_thres * std:
            return np.argmax(scores)
    return -1


def thres_strategy_naive(scores: list, thres=0.3):
    """Naive thresholding strategy."""
    if np.max(scores) > thres:
        return np.argmax(scores)
    return -1


def thres_strategy_softmax(scores: list, temp=0.8, ratio=1.4):
    """Threshold using softmax."""
    x = np.array(scores) / temp  # type: ignore
    ex = np.exp(x - np.max(x))
    ex /= ex.sum() + 1e-12
    # TODO: Figure out proper solution to sensitivity.
    if np.max(ex) > ratio / (len(ex) + 1):
        return np.argmax(ex)
    return -1


def evaluate_threshold_function(ds, sus_cls, func, x_axis, suspect_dropout, fixed=None):
    """Used in data.ipynb to evaluate threshold functions."""
    from tqdm import tqdm

    np.random.seed(42)
    acc_axis = []
    p_axis = []
    r_axis = []
    f_axis = []
    all_logits = [np.array(s["logits"]).copy() for s in ds]
    all_gts = [sus_cls.index(s["ground_truth"].label) for s in ds]

    for thres in tqdm(x_axis):
        tp, fp, tn, fn = 0, 0, 0, 0
        if fixed is not None:
            suspect_dropout = thres
        for logits, gt in zip(all_logits, all_gts):  # type: ignore
            logits = np.array(logits).copy()
            no_suspect = np.random.rand() < suspect_dropout

            if no_suspect:
                logits[gt] = np.delete(logits, gt).mean()

            pred = func(logits, thres if fixed is None else fixed)
            if no_suspect and pred == -1:
                tn += 1
            elif not no_suspect and pred == gt:
                tp += 1
            elif no_suspect and pred != -1:
                fp += 1
            elif not no_suspect and pred == -1:
                fn += 1
            elif not no_suspect and pred != gt:
                # We don't count false predictions for now.
                # fp += 1
                pass

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-6)

        acc_axis.append(acc)
        p_axis.append(p)
        r_axis.append(r)
        f_axis.append(f)

    return acc_axis, p_axis, r_axis, f_axis
