# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.datasets.nyuv2 import NYUV2Dataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.depth_evaluator import DepthEvaluator
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec


class DepthEstimationModel(BaseModel):
    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["depth_estimates"]

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return {
            "depth_estimates": TensorSpec(
                io_type=IoType.TENSOR,
                description="Monocular depth map",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "midas", 3, "test_input_image.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    def get_evaluator(self) -> BaseEvaluator:
        return DepthEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [NYUV2Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NYUV2Dataset
