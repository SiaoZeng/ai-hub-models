# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.eg1800 import EG1800SegmentationDataset
from qai_hub_models.models._shared.selfie_segmentation.model import SelfieSegmentor
from qai_hub_models.models.sinet.external_repos.ext_portrait_segmentation.models.SINet import (
    SINet as SINetModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_dataset import BaseDataset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "SINet.pth"
NUM_CLASSES = 2
INPUT_IMAGE_LOCAL_PATH = "sinet_demo.png"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, INPUT_IMAGE_LOCAL_PATH
)


class SINet(SelfieSegmentor):
    MASK_THRESHOLD = 0
    DEFAULT_HW = (224, 224)

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        sinet_model = _load_sinet_source_model_from_weights(weights)
        return cls(sinet_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run SINet on `image`, and produce a tensor of classes for segmentation

        Parameters
        ----------
        image
            Pixel values pre-processed for model consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        class_logits : torch.Tensor
            1x2xHxW tensor of class logits per pixel
        """
        image = image[:, [2, 1, 0]]  # RGB -> BGR
        # Mean / STD from https://github.com/clovaai/ext_portrait_segmentation/blob/master/etc/Visualize_video.py#L232
        mean = torch.Tensor([107.304565, 115.69884, 132.35703]) / 255
        std = torch.Tensor([63.97182, 65.1337, 68.29726])
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        image = (image - mean) / std
        return self.model(image)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [EG1800SegmentationDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return EG1800SegmentationDataset

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10


def _get_weightsfile_from_name(
    weights_name: str = DEFAULT_WEIGHTS,
) -> CachedWebModelAsset:
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == DEFAULT_WEIGHTS:
        return CachedWebModelAsset(
            "https://github.com/clovaai/ext_portrait_segmentation/raw/master/result/SINet/SINet.pth",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            "SINet.pth",
        )
    raise NotImplementedError(f"Cannot get weights file from name {weights_name}")


def _load_sinet_source_model_from_weights(
    weights_name_or_path: str,
) -> torch.nn.Module:
    if os.path.exists(os.path.expanduser(weights_name_or_path)):
        weights_path = os.path.expanduser(weights_name_or_path)
    elif not os.path.exists(weights_name_or_path):
        # Load SINet model from the source repository using the given weights.
        weights_path = _get_weightsfile_from_name(weights_name_or_path).fetch()
    else:
        weights_path = None
    weights = load_torch(weights_path or weights_name_or_path)

    # This config is copied from the main function in Sinet.py:
    # https://github.com/clovaai/ext_portrait_segmentation/blob/9bc1bada1cb7bd17a3a80a2964980f4b4befef5b/models/SINet.py#L557
    config = [
        [[3, 1], [5, 1]],
        [[3, 1], [3, 1]],
        [[3, 1], [5, 1]],
        [[3, 1], [3, 1]],
        [[5, 1], [3, 2]],
        [[5, 2], [3, 4]],
        [[3, 1], [3, 1]],
        [[5, 1], [5, 1]],
        [[3, 2], [3, 4]],
        [[3, 1], [5, 2]],
    ]

    sinet_model = SINetModel(classes=2, p=2, q=8, config=config, chnn=1)
    sinet_model.load_state_dict(weights, strict=True)

    return sinet_model
