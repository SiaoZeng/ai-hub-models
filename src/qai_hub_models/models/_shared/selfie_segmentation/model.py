# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)


class SelfieSegmentor(BaseModel):
    MASK_THRESHOLD: float  # Threshold above which a pixel is classified as foreground in the binary mask output by the model.
    DEFAULT_HW: tuple[int, int] = (256, 256)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_HW[0],
        width: int = DEFAULT_HW[1],
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
            )
        }

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return {
            "mask": TensorSpec(
                io_type=IoType.TENSOR,
                description="Binary segmentation mask",
            ),
        }

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["mask"]

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(2, mask_threshold=self.MASK_THRESHOLD)
