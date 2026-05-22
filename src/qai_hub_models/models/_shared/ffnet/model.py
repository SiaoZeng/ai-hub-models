# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import torch
from typing_extensions import Self

from qai_hub_models.datasets.cityscapes_lowres import CityscapesLowResDataset
from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.models._shared.ffnet.external_repos.ffnet.models.model_registry import (
    model_entrypoint,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
FFNET_WEIGHTS_URL_ROOT = (
    "https://github.com/quic/aimet-model-zoo/releases/download/torch_segmentation_ffnet"
)
FFNET_SUBPATH_NAME_LOOKUP = {
    # Variant name (in FFNet repo) to (subpath, src_name, dst_name)
    "segmentation_ffnet40S_dBBB_mobile": (
        "ffnet40S",
        "ffnet40S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet40S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet54S_dBBB_mobile": (
        "ffnet54S",
        "ffnet54S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet54S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet78S_dBBB_mobile": (
        "ffnet78S",
        "ffnet78S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet78S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet78S_BCC_mobile_pre_down": (
        "ffnet78S",
        "ffnet78S_BCC_cityscapes_state_dict_quarts_pre_down.pth",
        "ffnet78S_BCC_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet122NS_CCC_mobile_pre_down": (
        "ffnet122NS",
        "ffnet122NS_CCC_cityscapes_state_dict_quarts_pre_down.pth",
        "ffnet122NS_CCC_cityscapes_state_dict_quarts.pth",
    ),
}
FFNetType = TypeVar("FFNetType", bound="FFNet")


class FFNet(CityscapesSegmentor):
    """Exportable FFNet fuss-free Cityscapes segmentation model."""

    model: torch.nn.Module  # narrows BaseModel's Tensor | Module to nn.Module

    @classmethod
    def from_pretrained(cls, variant_name: str) -> Self:
        model = _load_ffnet_source_model(variant_name)
        return cls(model)


def _load_ffnet_source_model(variant_name: str) -> torch.nn.Module:
    subpath, src_name, dst_name = FFNET_SUBPATH_NAME_LOOKUP[variant_name]

    weights_path = CachedWebModelAsset(
        f"{FFNET_WEIGHTS_URL_ROOT.rstrip('/')}/{src_name.lstrip('/')}",
        MODEL_ID,
        MODEL_ASSET_VERSION,
        Path(subpath) / dst_name,
    ).fetch()
    root_weights_path = os.path.dirname(os.path.dirname(weights_path))

    os.environ["FFNET_WEIGHTS_PATH"] = root_weights_path

    return model_entrypoint(variant_name)()


class FFNetLowRes(FFNet):
    @classmethod
    def from_pretrained(cls, variant_name: str) -> Self:
        instance = super().from_pretrained(variant_name)
        # The _pre_down variants were trained with an internal Gaussian
        # downsample (1024x2048 -> 512x1024). For deployment we feed
        # 512x1024 directly, so disable the pre-downsampling layer so
        # the input is not halved a second time.
        instance.model.pre_downsampling = False  # type: ignore[assignment]
        return instance

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 1024,
    ) -> InputSpec:
        return super().get_input_spec(batch_size, height, width)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CityscapesLowResDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CityscapesLowResDataset
