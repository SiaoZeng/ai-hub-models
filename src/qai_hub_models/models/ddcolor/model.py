# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.datasets.imagenet_colorization import ImagenetColorizationDataset
from qai_hub_models.datasets.imagenette_colorization import (
    ImagenetteColorizationDataset,
)
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.colorization_evaluator import ColorizationEvaluator
from qai_hub_models.models.common import Precision
from qai_hub_models.models.ddcolor.external_repos.ddcolor.infer_hf import DDColorHF
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHT = "piddnad/ddcolor_paper_tiny"


class DDColor(BaseModel):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHT) -> Self:
        model = DDColorHF.from_pretrained(weights)
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Stores the input image and passes it through the colorization model to obtain AB channel.

        Args :
            image (torch.Tensor) : Input tensor of shape (1, 3, H, W) representing a grayscale RGB image with range [0, 1].

        Returns :
            torch.Tensor: Output tensor of shape (1, 2, 256, 256) representing the predicted AB color channels.
        """
        return self.model(image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
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
        return ["output"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return ColorizationEvaluator()

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetColorizationDataset, ImagenetteColorizationDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return ImagenetteColorizationDataset

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for mixed precision quantization."""
        return 10.0

    @classmethod
    def get_labels_file_name(cls) -> str | None:
        return "imagenet_labels.txt"
