# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.rangenet_plus_plus.external_repos.lidar_bonnetal.train.tasks.semantic.modules.segmentator import (
    Segmentator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

NUM_CLASSES = 20
INPUT_HEIGHT = 64
INPUT_WIDTH = 2048
INPUT_CHANNELS = 5

SAMPLE_POINT_CLOUD_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000.bin"
)

OUTPUT_MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "rangenet_mask.npy"
)

# Source: https://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz
DARKNET53_MODEL_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "darknet53.tar.gz"
)


class RangeNetPlusPlus(BaseModel):
    """RangeNet++ LiDAR semantic segmentation model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path | None = None,
    ) -> Self:
        """
        Load pretrained RangeNet++.

        Parameters
        ----------
        model_dir
            Path to the darknet53 model folder. Downloaded automatically if None.

        Returns
        -------
        Self
            Loaded model instance.
        """
        if model_dir is None:
            model_dir = DARKNET53_MODEL_ASSET.fetch(extract=True)

        config_path = Path(model_dir) / "arch_cfg.yaml"
        with open(config_path) as f:
            arch_cfg = YAML(typ="safe", pure=True).load(f)

        model = Segmentator(arch_cfg, NUM_CLASSES, path=str(model_dir))
        model.eval()
        return cls(model)

    def forward(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic labels for a LiDAR range image.

        Parameters
        ----------
        range_image
            float32 tensor of shape [1, 5, H, W] with channels ordered
            [depth, x, y, z, intensity]. Values must be pre-normalised
            per-channel using the SemanticKITTI mean/std statistics
            (see ``project_points_to_range_image`` in app.py). The
            expected input range after normalisation is approximately
            [-3, 3] for each channel.

        Returns
        -------
        torch.Tensor
            uint8 class-index mask of shape [1, H, W]. Each value is a
            class index in [0, NUM_CLASSES - 1] corresponding to the
            SemanticKITTI 20-class label set.
        """
        logits = self.model(range_image)
        return torch.argmax(logits, dim=1).byte()

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = INPUT_HEIGHT,
        width: int = INPUT_WIDTH,
    ) -> InputSpec:
        return {"range_image": ((batch_size, INPUT_CHANNELS, height, width), "float32")}

    def get_output_names(self) -> list[str]:
        return ["mask"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["range_image"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["mask"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        from qai_hub_models.models.rangenet_plus_plus.app import (
            project_points_to_range_image,
        )

        points = np.fromfile(
            str(SAMPLE_POINT_CLOUD_ADDRESS.fetch()), dtype=np.float32
        ).reshape(-1, 4)
        arr, _, _ = project_points_to_range_image(points)
        return {"range_image": [arr]}
