# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models.fomm.external_repos.first_order_model.demo import (
    load_checkpoints,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_path,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = CachedWebModelAsset(
    "https://github.com/AliaksandrSiarohin/first-order-model/raw/master/config/vox-256.yaml",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-256.yaml",
)
DEFAULT_WEIGHTS_GDRIVE = CachedWebModelAsset.from_google_drive(
    "1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-cpk.pth.tar",
)
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-cpk.pth.tar",
)


class FOMMDetector(BaseModel):
    """Keypoint detector that finds keypoints over source and driving images"""

    def __init__(self, kp_detector: torch.nn.Module) -> None:
        super().__init__()
        self.model = kp_detector

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run FOMM keypoint detector on an image.

        Parameters
        ----------
        image
            Input image tensor of shape BxCxHxW.
            RGB, range [0 - 1].

        Returns
        -------
        keypoints : torch.Tensor
            Keypoints detected in the image.
            Shape: B x Num keypoints x 2.
        jacobian : torch.Tensor
            Jacobian matrix around each keypoint.
            Shape: B x Num keypoints x 2 x 2.
        """
        result = self.model(image)
        keypoints = result["value"]
        jacobian = result["jacobian"]
        return keypoints, jacobian

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(FOMM.get_fomm_model()[1])

    @staticmethod
    def get_input_spec(
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

    @staticmethod
    def get_output_names() -> list[str]:
        return ["keypoints", "jacobian"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


class FOMMGenerator(BaseModel):
    """Given keypoints from a source image, a target image, and the norm of the keypoints from the target,
    generates the new target image
    """

    def __init__(self, generator: torch.nn.Module) -> None:
        super().__init__()
        self.model = generator

    def forward(
        self,
        image: torch.Tensor,
        source_keypoint_values: torch.Tensor,
        source_keypoint_jacobians: torch.Tensor,
        kp_norm_values: torch.Tensor,
        kp_norm_jacobians: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the new target image based on the source keypoints and target keypoints.

        Parameters
        ----------
        image
            The source image tensor of shape BxCxHxW.
            RGB, range [0 - 1].
        source_keypoint_values
            Keypoints detected in source image.
            Shape: B x num keypoints x 2.
        source_keypoint_jacobians
            Jacobians around source keypoints.
            Shape: B x num keypoints x 2 x 2.
        kp_norm_values
            Normalised keypoints detected in driving image.
            Shape: B x num keypoints x 2.
        kp_norm_jacobians
            Jacobians around driving keypoints.
            Shape: B x num keypoints x 2 x 2.

        Returns
        -------
        output_image : torch.Tensor
            Predicted output image for the given driving frame keypoints.
            Shape: BxCxHxW.
        """
        # run generator. The underlying model takes in dictionaries
        source_kp = dict(
            value=source_keypoint_values, jacobian=source_keypoint_jacobians
        )
        kp_norm = dict(value=kp_norm_values, jacobian=kp_norm_jacobians)
        out = self.model(image, kp_source=source_kp, kp_driving=kp_norm)
        return out["prediction"]
        # For the purposes of tracing we return only the prediction element of the dictionary
        # as this is the only part that the app uses

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(FOMM.get_fomm_model()[0])

    @staticmethod
    def get_input_spec(
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
            "source_keypoint_values": TensorSpec(
                shape=(batch_size, 10, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "source_keypoint_jacobians": TensorSpec(
                shape=(batch_size, 10, 2, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "kp_norm_values": TensorSpec(
                shape=(batch_size, 10, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "kp_norm_jacobians": TensorSpec(
                shape=(batch_size, 10, 2, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_image"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


@CollectionModel.add_component(FOMMDetector, "detector")
@CollectionModel.add_component(FOMMGenerator, "generator")
class FOMM(PretrainedCollectionModel):
    """Exportable FOMM for Image Editing"""

    def __init__(self, detector: FOMMDetector, generator: FOMMGenerator) -> None:
        super().__init__(detector, generator)
        self.detector = detector
        self.generator = generator

    @classmethod
    def get_fomm_model(
        cls, weights_url: str | None = None
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        # Download default config
        fomm_config = DEFAULT_CONFIG.fetch()

        # Download weights
        with qaihm_temp_dir() as tmpdir:
            weights_path = load_path(weights_url or DEFAULT_WEIGHTS, tmpdir)

            generator, kp_detector = load_checkpoints(
                config_path=fomm_config, checkpoint_path=weights_path, cpu=True
            )
        return generator, kp_detector

    @classmethod
    def from_pretrained(cls, weights_url: str | None = None) -> Self:
        generator, kp_detector = cls.get_fomm_model(weights_url=weights_url)
        generator_model = FOMMGenerator(generator)
        kp_detector_model = FOMMDetector(kp_detector)
        return cls(kp_detector_model, generator_model)
