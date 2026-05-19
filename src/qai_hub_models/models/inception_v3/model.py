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
DEFAULT_WEIGHTS = "IMAGENET1K_V1"
INCEPTION_V3_DIM = 299

# Official torchvision Inception_V3_Weights.IMAGENET1K_V1:
# resize=342, crop=299, BILINEAR, antialias=True
INCEPTION_V3_TRANSFORM = make_imagenet_transform(
    crop_size=INCEPTION_V3_DIM,
    interpolation=T.InterpolationMode.BILINEAR,
    antialias=True,
)


class InceptionNetV3(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.inception_v3(weights=weights, transform_input=False)
        return cls(net, transform_input=True)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "imagenette_inception_v3"

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["imagenet_inception_v3", "imagenette"]

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, INCEPTION_V3_DIM, INCEPTION_V3_DIM),
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
        tensor = INCEPTION_V3_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])

    @classmethod
    def get_dataset_class(cls) -> type[ImagenetDataset]:
        class ImagenetInceptionV3Dataset(ImagenetDataset):
            def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
                super().__init__(split=split, transform=INCEPTION_V3_TRANSFORM)

            @classmethod
            def dataset_name(cls) -> str:
                return "imagenet_inception_v3"

        return ImagenetInceptionV3Dataset

    @classmethod
    def get_imagenette_dataset_class(cls) -> type[ImagenetteDataset]:
        class ImagenetteInceptionV3Dataset(ImagenetteDataset):
            def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
                super().__init__(split=split, transform=INCEPTION_V3_TRANSFORM)

            @classmethod
            def dataset_name(cls) -> str:
                return "imagenette_inception_v3"

        return ImagenetteInceptionV3Dataset


DATASET_NAME_MAP["imagenet_inception_v3"] = InceptionNetV3.get_dataset_class()
DATASET_NAME_MAP["imagenette_inception_v3"] = (
    InceptionNetV3.get_imagenette_dataset_class()
)
