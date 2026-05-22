# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.image_processing import pre_process_with_affine
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.printing import suppress_stdout


class CocoCenterNetDataset(CocoBodyDataset):
    """Whole-image COCO dataset for bottom-up pose estimation."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        with suppress_stdout():
            super().__init__(
                split=split, input_spec=input_spec, num_samples=num_samples
            )
        # Override img_ids to use all images (not just single-person crops)
        self.img_ids = sorted(self.cocoGt.getImgIds())

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Preprocessed full image. Shape (3, H, W), float32, RGB.
        gt : tuple[int, np.ndarray, np.ndarray]
            image_id
                COCO image ID.
            center
                Image center [w/2, h/2], shape (2,).
            scale
                Scale [max(h,w), max(h,w)], shape (2,).
        """
        img_id = self.img_ids[index]
        img_info = self.cocoGt.loadImgs(img_id)[0]

        raw = cv2.imread(
            str(self.image_dir / img_info["file_name"]),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        assert raw is not None, f"Image not found: {img_info['file_name']}"
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        h, w = raw.shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = np.array([float(max(h, w))] * 2, dtype=np.float32)

        image = pre_process_with_affine(
            raw, center, scale, 0, (self.target_h, self.target_w)
        ).squeeze(0)  # (3, H, W)

        return image, (img_id, center, scale)

    def __len__(self) -> int:
        return len(self.img_ids)

    @staticmethod
    def default_samples_per_job() -> int:
        return 300
