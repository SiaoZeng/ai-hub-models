# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.metrics import ACCURACY_TOP1, MetricMetadata


class VideoClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for video classification with multi-view (clip x crop) aggregation.

    Each call to ``add_batch`` receives:

    * ``output``  - per-view class scores of shape ``[B, num_classes]`` (the
                    model forward returns summed softmax probabilities).
    * ``gt``      - a tuple ``(label_tensor, video_id_tensor)`` where
                    ``label_tensor`` has shape ``[B]`` (integer class indices)
                    and ``video_id_tensor`` has shape ``[B]`` (integer video IDs).

    Scores for the same ``video_id`` are **summed** across all views.  Top-1 /
    top-5 accuracy is computed from the aggregated scores at query time.

    Parameters
    ----------
    num_classes
        Number of output classes (400 for Kinetics-400).
    """

    def __init__(self, num_classes: int = 400) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        # video_id -> aggregated [num_classes] score tensor.
        self._score_sums: dict[int, torch.Tensor] = {}
        # video_id -> ground-truth label index.
        self._labels: dict[int, int] = {}

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Accumulate per-view scores keyed by ``video_id``.

        Parameters
        ----------
        output
            Per-view class scores, shape ``[B, num_classes]``.
        gt
            ``(labels, video_ids)`` each of shape ``[B]``; labels are class
            indices, video_ids are shared across all views of the same video.
        """
        labels, video_ids = gt
        scores = output.detach().cpu().float()
        labels_list = labels.cpu().to(torch.int64).tolist()
        vid_list = video_ids.cpu().to(torch.int64).tolist()

        for i, vid in enumerate(vid_list):
            if vid not in self._score_sums:
                self._score_sums[vid] = torch.zeros_like(scores[i])
                self._labels[vid] = labels_list[i]
            self._score_sums[vid] += scores[i]

    def _topk_correct(self, k: int) -> float:
        total = len(self._score_sums)
        if total == 0:
            return 0.0
        scores: list[torch.Tensor] = []
        labels: list[int] = []
        for vid, row in self._score_sums.items():
            scores.append(row)
            labels.append(self._labels[vid])
        score_tensor = torch.stack(scores)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64)
        topk = torch.topk(score_tensor, k, dim=1).indices
        correct = (topk == label_tensor.view(-1, 1)).any(dim=1).sum().item()
        return float(correct) / total * 100

    def top1(self) -> float:
        return self._topk_correct(1)

    def top5(self) -> float:
        return self._topk_correct(5)

    def get_accuracy_score(self) -> float:
        return self.top1()

    def formatted_accuracy(self) -> str:
        return f"{self.top1():.1f}% (Top 1), {self.top5():.1f}% (Top 5)"

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY_TOP1
