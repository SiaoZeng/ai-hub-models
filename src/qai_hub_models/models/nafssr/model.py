# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms
from typing_extensions import Self

from qai_hub_models.datasets.flickr1024 import Flickr1024Dataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.stereo_evaluator import StereoEvaluator
from qai_hub_models.models._shared.nafnet.model import (
    NAFNetModel,
    _load_nafnet_source_model,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

LR_IMAGE_L = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/lr_img_l.png"
)
LR_IMAGE_R = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/lr_img_r.png"
)
SCALING_FACTOR = 2
DEFAULT_WEIGHTS = "NAFSSR-S_2x.pth"
DEFAULT_CONFIG = "NAFSSR/NAFSSR-S_2x.yml"


class NAFSSR(NAFNetModel):
    """Exportable NAFNet stereo super resolution model for image upscaling, end-to-end."""

    def __init__(
        self,
        nafssr_model: torch.nn.Module,
    ) -> None:
        super().__init__(nafssr_model)

    @classmethod
    def from_pretrained(
        cls,
        nafnet_weights: str | Path | None = None,
        yaml_path_nafnet: str | Path | None = None,
    ) -> Self:
        """Load NAFSSR from a weightfile created by the source NAFNet repository.

        Parameters
        ----------
        nafnet_weights
            Local path to a ``.pth`` weights file.  When ``None`` the default
            pretrained weights are downloaded from the asset store.
        yaml_path_nafnet
            Local path to a ``.yml`` config file.  When ``None`` the default
            config bundled with the NAFNet repository is used.

        Returns
        -------
        Self
            Instantiated NAFSSR model with pretrained weights loaded.
        """
        nafnet_model = _load_nafnet_source_model(
            nafnet_weights=Path(nafnet_weights)
            if nafnet_weights is not None
            else DEFAULT_WEIGHTS,
            yaml_path_nafnet=Path(yaml_path_nafnet)
            if yaml_path_nafnet is not None
            else DEFAULT_CONFIG,
            MODEL_ID=MODEL_ID,
            MODEL_ASSET_VERSION=MODEL_ASSET_VERSION,
        )

        return cls(nafnet_model)

    def get_evaluator(self) -> BaseEvaluator:
        return StereoEvaluator()

    def forward(
        self, l_image: torch.Tensor, r_image: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Run NAFNet Stereo Resolution on left and right view images, and produce upscaled image(s).

        Parameters
        ----------
        l_image
            Left view of input image to be processed
            Pixel values in range [0, 1], RGB color space.


        r_image
            Right view of input image to be processed
            Pixel values in range [0, 1], RGB color space.

        Returns
        -------
        upscaled_image(s) : list[torch.Tensor]
            Upscaled images in list [left,right]
            Pixel values in range [0, 1], RGB color space.
        """
        return self.model(l_image, r_image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        return {
            "l_image": ((batch_size, 3, height, width), "float32"),
            "r_image": ((batch_size, 3, height, width), "float32"),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        l_img = load_image(LR_IMAGE_L)
        r_img = load_image(LR_IMAGE_R)
        if input_spec is not None:
            h, w = input_spec["l_image"][0][2:]
            l_img = l_img.resize((w, h))
            r_img = r_img.resize((w, h))
        img_l = transforms.ToTensor()(l_img).unsqueeze(0)
        img_r = transforms.ToTensor()(r_img).unsqueeze(0)
        return {"l_image": [img_l.numpy()], "r_image": [img_r.numpy()]}

    def get_output_names(self) -> list[str]:
        return ["upscaled_left", "upscaled_right"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["l_image", "r_image"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["upscaled_left", "upscaled_right"]

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        return [Flickr1024Dataset]

    def get_calibration_dataset_cls(self) -> type:
        return Flickr1024Dataset
