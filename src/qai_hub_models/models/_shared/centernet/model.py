# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.models._shared.centernet.external_repos.centernet.src.lib.models.networks.DCNv2.dcn_v2 import (
    DCN,
)
from qai_hub_models.models._shared.centernet.external_repos.centernet.src.lib.models.networks.pose_dla_dcn import (
    get_pose_net,
)
from qai_hub_models.models._shared.centernet.model_patches import custom_dcn_forward
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)


class CenterNet(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_pretrained(cls, ckpt_path: str, heads: dict) -> Self:
        DCN.forward = custom_dcn_forward
        model = get_pose_net(
            num_layers=34,
            heads=heads,
            head_conv=256,
            down_ratio=4,
        )
        model = cast(Self, load_model(model, ckpt_path))
        model.eval()
        return model

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"

        return compile_options

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
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


def load_model(model: nn.Module, model_path: str) -> nn.Module:
    checkpoint = torch.load(model_path)
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model.load_state_dict(state_dict, strict=False)
    return model
