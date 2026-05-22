# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.evaluators.posenet_mobilenet_evaluator import (
    PosenetMobilenetEvaluator,
)
from qai_hub_models.models.posenet_mobilenet.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.posenet_mobilenet.external_repos.posenet_pytorch import (
    posenet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_numpy,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "posenet_inputs.npy"
)
DEFAULT_MODEL_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "mobilenet_v1_101.pth"
)
OUTPUT_STRIDE = 16

_POSENET_REPO_ROOT = EXTERNAL_REPO_PATHS["posenet_pytorch"]


class PosenetMobilenet(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_id: int = 101,
    ) -> Self:
        # Built in weights downloading is sometimes flaky.
        # Download default weights from Qualcomm AWS
        ckpt_path = _POSENET_REPO_ROOT / "_models" / DEFAULT_MODEL_WEIGHTS.path.name
        if not ckpt_path.exists():
            DEFAULT_MODEL_WEIGHTS.fetch()
            os.makedirs(ckpt_path.parent, exist_ok=True)
            os.symlink(DEFAULT_MODEL_WEIGHTS.path, ckpt_path)

        model = posenet.load_model(model_id)

        return cls(model)

    # Caution: adding typehints to this method's parameter or return will trigger a
    # bug in torch that leads to the following exception:
    # AttributeError: 'str' object has no attribute '__name__'. Did you mean: '__ne__'?
    def forward(
        self, image: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Image inputs are expected to be in RGB format in the range [0, 1]."""
        raw_output = self.model(image * 2.0 - 1.0)
        max_vals = F.max_pool2d(raw_output[0], 3, stride=1, padding=1)
        return (*raw_output, max_vals)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 513,
        width: int = 257,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_output_names(self) -> list[str]:
        return [
            "heatmaps_result",
            "offsets_result",
            "displacement_fwd_result",
            "displacement_bwd_result",
            "max_vals",
        ]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [load_numpy(SAMPLE_INPUTS)]}

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        h, w = self.get_input_spec()["image"][0][2:]
        return PosenetMobilenetEvaluator(h, w)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoBodyDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoBodyDataset

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.

        For this model, the value is fixed to 10.0 based on internal experimentation
        that showed it provides a good trade-off between accuracy and performance.

        """
        return 10
