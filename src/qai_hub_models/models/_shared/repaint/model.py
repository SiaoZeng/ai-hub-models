# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.celebahq import CelebAHQDataset
from qai_hub_models.evaluators.inpaint_evaluator import InpaintEvaluator
from qai_hub_models.models._shared.repaint.utils import preprocess_inputs
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
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

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    "aotgan", 2, "test_images/test_input_image.png"
)
MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    "aotgan", 2, "test_images/test_input_mask.png"
)


class RepaintModel(BaseModel):
    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
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
            "mask": TensorSpec(
                shape=(batch_size, 1, height, width),
                dtype="float32",
            ),
        }

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return {
            "painted_image": TensorSpec(
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_channel_last_inputs(self) -> list[str]:
        return ["image", "mask"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["painted_image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """Provides an example image of a man with a mask over the glasses."""
        image = load_image(IMAGE_ADDRESS)
        mask = load_image(MASK_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
            mask = mask.resize((w, h))
        torch_inputs = preprocess_inputs(image, mask)
        return {k: [v.detach().numpy()] for k, v in torch_inputs.items()}

    def get_evaluator(self) -> BaseEvaluator:
        return InpaintEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CelebAHQDataset]
