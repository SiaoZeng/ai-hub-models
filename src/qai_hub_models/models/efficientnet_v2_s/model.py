# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torchvision.models as tv_models
import torchvision.transforms as T
from typing_extensions import Self

from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.imagenette import ImagenetteDataset
from qai_hub_models.models._shared.imagenet_classifier.model import (
    TEST_IMAGENET_IMAGE,
    ImagenetClassifier,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.image_processing import make_imagenet_transform
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "IMAGENET1K_V1"
EFFICIENTNET_V2_S_DIM = 384

# Official torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1:
# resize=384 (no crop), BILINEAR, antialias=True
EFFICIENTNET_V2_S_TRANSFORM = make_imagenet_transform(
    crop_size=EFFICIENTNET_V2_S_DIM,
    resize_size=EFFICIENTNET_V2_S_DIM,
    interpolation=T.InterpolationMode.BILINEAR,
    antialias=True,
)


class EfficientNetV2s(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.efficientnet_v2_s(weights=weights)
        return cls(net)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    @staticmethod
    def calibration_dataset_name() -> str:
        return "imagenette_efficientnet_v2_s"

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["imagenet_efficientnet_v2_s", "imagenette"]

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, EFFICIENTNET_V2_S_DIM, EFFICIENTNET_V2_S_DIM),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = EFFICIENTNET_V2_S_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])

    @classmethod
    def get_dataset_class(cls) -> type[ImagenetDataset]:
        class ImagenetEfficientNetV2SDataset(ImagenetDataset):
            def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
                super().__init__(split=split, transform=EFFICIENTNET_V2_S_TRANSFORM)

            @classmethod
            def dataset_name(cls) -> str:
                return "imagenet_efficientnet_v2_s"

        return ImagenetEfficientNetV2SDataset

    @classmethod
    def get_imagenette_dataset_class(cls) -> type[ImagenetteDataset]:
        class ImagenetteEfficientNetV2SDataset(ImagenetteDataset):
            def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
                super().__init__(split=split, transform=EFFICIENTNET_V2_S_TRANSFORM)

            @classmethod
            def dataset_name(cls) -> str:
                return "imagenette_efficientnet_v2_s"

        return ImagenetteEfficientNetV2SDataset


DATASET_NAME_MAP["imagenet_efficientnet_v2_s"] = EfficientNetV2s.get_dataset_class()
DATASET_NAME_MAP["imagenette_efficientnet_v2_s"] = (
    EfficientNetV2s.get_imagenette_dataset_class()
)
