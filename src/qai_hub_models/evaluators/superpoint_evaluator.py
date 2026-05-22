# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from qai_hub_models.evaluators.metrics import HOMOGRAPHY_ACCURACY, MetricMetadata
from qai_hub_models.utils.base_evaluator import BaseEvaluator


class SuperPointEvaluator(BaseEvaluator):
    """Evaluates SuperPoint on HPatches using homography estimation @3px."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        keep_k_points: int = 1000,
        ransac_reproj_threshold: float = 3.0,
        correctness_thresh: float = 3.0,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.keep_k_points = keep_k_points
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.correctness_thresh = correctness_thresh
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.reset()

    def reset(self) -> None:
        self._num_total = 0
        self._homo_correct = 0

    def get_accuracy_score(self) -> float:
        if self._num_total == 0:
            return 0.0
        return 100.0 * self._homo_correct / self._num_total

    def formatted_accuracy(self) -> str:
        return f"Homography@3 {self.get_accuracy_score():.1f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return HOMOGRAPHY_ACCURACY

    def add_batch(
        self,
        output: Sequence[torch.Tensor],
        gt: torch.Tensor | Sequence[torch.Tensor],
    ) -> None:
        """
        Accumulate homography estimation results for a batch.

        Parameters
        ----------
        output
            Model outputs as a sequence of three tensors:

            - ``output[0]`` — keypoints, shape ``(B, 2, K, 2)``, float32,
              (x, y) pixel coordinates sorted by score descending.
            - ``output[1]`` — scores, shape ``(B, 2, K)``, float32 in ``[0, 1]``.
              Zero-padded slots indicate no keypoint.
            - ``output[2]`` — descriptors, shape ``(B, 2, K, 256)``, float32,
              L2-normalized.

        gt
            Ground truth homography ``H_gt`` of shape ``(B, 3, 3)``, float32,
            mapping image 0 pixel coords to image 1 pixel coords.
        """
        keypoints_batch = output[0]
        scores_batch = output[1]
        descriptors_batch = output[2]

        H_gt_batch: torch.Tensor = gt if isinstance(gt, torch.Tensor) else gt[0]

        B = H_gt_batch.shape[0]

        for i in range(B):
            kp0_i = keypoints_batch[i, 0].cpu().numpy()
            kp1_i = keypoints_batch[i, 1].cpu().numpy()
            scores0_i = scores_batch[i, 0].cpu().numpy()
            scores1_i = scores_batch[i, 1].cpu().numpy()
            desc0_i = descriptors_batch[i, 0].cpu().numpy()
            desc1_i = descriptors_batch[i, 1].cpu().numpy()

            valid_mask0 = scores0_i > 0
            valid_mask1 = scores1_i > 0
            kp0_i = kp0_i[valid_mask0]
            kp1_i = kp1_i[valid_mask1]
            desc0_i = desc0_i[valid_mask0]
            desc1_i = desc1_i[valid_mask1]

            H_gt = np.asarray(H_gt_batch[i].cpu().numpy(), dtype=np.float64).reshape(
                3, 3
            )

            image_hw = (self.image_height, self.image_width)

            kp0_i, desc0_i = self._keep_shared_points(
                kp0_i, desc0_i, H_gt, image_hw, keep_k=self.keep_k_points
            )
            kp1_i, desc1_i = self._keep_shared_points(
                kp1_i,
                desc1_i,
                np.linalg.inv(H_gt),
                image_hw,
                keep_k=self.keep_k_points,
            )

            correct = self._evaluate_homography(
                kp0_i, desc0_i, kp1_i, desc1_i, H_gt, image_hw
            )
            self._homo_correct += int(correct)
            self._num_total += 1

    @staticmethod
    def _keep_shared_points(
        kp: np.ndarray,
        desc: np.ndarray,
        H: np.ndarray,
        image_hw: tuple[int, int],
        keep_k: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter keypoints that warp inside the target image and keep top-k."""
        h, w = image_hw
        if len(kp) == 0:
            return kp, desc
        n = kp.shape[0]
        pts_h = np.concatenate([kp, np.ones((n, 1))], axis=1)
        warped = (H @ pts_h.T).T
        warped_xy = warped[:, :2] / warped[:, 2:3]
        mask = (
            (warped_xy[:, 0] >= 0)
            & (warped_xy[:, 0] < w)
            & (warped_xy[:, 1] >= 0)
            & (warped_xy[:, 1] < h)
        )
        kp, desc = kp[mask], desc[mask]
        if len(kp) > keep_k:
            kp, desc = kp[:keep_k], desc[:keep_k]
        return kp, desc

    def _evaluate_homography(
        self,
        kp0: np.ndarray,
        desc0: np.ndarray,
        kp1: np.ndarray,
        desc1: np.ndarray,
        H_gt: np.ndarray,
        image_hw: tuple[int, int],
    ) -> bool:
        """Match descriptors, estimate homography via RANSAC, and check correctness."""
        if len(kp0) < 4 or len(kp1) < 4:
            return False
        matches = self._bf.match(desc0.astype(np.float32), desc1.astype(np.float32))
        if len(matches) < 4:
            return False
        query_idx = np.array([m.queryIdx for m in matches], dtype=np.int32)
        train_idx = np.array([m.trainIdx for m in matches], dtype=np.int32)
        src_pts = kp0[query_idx].astype(np.float32)
        dst_pts = kp1[train_idx].astype(np.float32)
        H_est, inliers = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold
        )
        if H_est is None or inliers is None or inliers.sum() < 4:
            return False
        return self._corner_error(H_est, H_gt, image_hw) < self.correctness_thresh

    @staticmethod
    def _corner_error(
        H_est: np.ndarray,
        H_gt: np.ndarray,
        image_hw: tuple[int, int],
    ) -> float:
        """Compute mean corner reprojection error between estimated and ground truth homography."""
        h, w = image_hw
        H_est = np.asarray(H_est, dtype=np.float64).reshape(3, 3)
        H_gt = np.asarray(H_gt, dtype=np.float64).reshape(3, 3)
        corners = np.array(
            [[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]],
            dtype=np.float64,
        )
        real_warped = corners @ H_gt.T
        real_warped = real_warped[:, :2] / real_warped[:, 2:3]
        est_warped = corners @ H_est.T
        est_warped = est_warped[:, :2] / est_warped[:, 2:3]
        return float(np.mean(np.linalg.norm(real_warped - est_warped, axis=1)))
