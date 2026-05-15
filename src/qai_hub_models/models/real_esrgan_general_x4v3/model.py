# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from typing_extensions import Self

from qai_hub_models.extern.basicsr.archs.srvgg_arch import SRVGGNetCompact
from qai_hub_models.models._shared.super_resolution.model import SuperResolutionModel
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "realesr-general-x4v3"
DEFAULT_WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
PRE_PAD = 10
SCALING_FACTOR = 4


class Real_ESRGAN_General_x4v3(SuperResolutionModel):
    """Exportable RealESRGAN upscaler, end-to-end."""

    def __init__(
        self,
        realesrgan_model: torch.nn.Module,
    ) -> None:
        super().__init__(realesrgan_model, scale_factor=SCALING_FACTOR)

    @classmethod
    def from_pretrained(
        cls,
        weight_path: str = DEFAULT_WEIGHTS,
    ) -> Self:
        """Load Real_ESRGAN_General_x4v3 from a weightfile created by the source RealESRGAN repository."""
        # Load PyTorch model from disk
        realesrgan_model = _load_realesrgan_source_model_from_weights(weight_path)

        return cls(realesrgan_model)


def _load_realesrgan_source_model_from_weights(
    weights_name_or_path: str,
) -> torch.nn.Module:
    if os.path.exists(os.path.expanduser(weights_name_or_path)):
        weights_path = os.path.expanduser(weights_name_or_path)
    else:
        weights_asset = CachedWebModelAsset(
            DEFAULT_WEIGHTS_URL,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            "realesr-general-x4v3.pth",
        )
        weights_path = weights_asset.fetch()

    realesrgan_model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type="prelu",
    )
    pretrained_dict = torch.load(
        weights_path, map_location=torch.device("cpu"), weights_only=False
    )

    keyname = "params_ema" if "params_ema" in pretrained_dict else "params"
    realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

    return realesrgan_model
