# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

import qai_hub_models.models.cvt.external_repos.cross_view_transformers as cvt_repo
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.datasets.nuscenes_bev import NuscenesBevCVTDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.nuscenes_bev_evaluator import (
    NuscenesBevSegmentationEvaluator,
)
from qai_hub_models.models.common import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.models.cvt.external_repos.cross_view_transformers.cross_view_transformer.common import (
    remove_prefix,
    setup_network,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.external_repo import rewrite_hydra_targets
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

MODEL_ID = __name__.split(".")[-2]
CKPT_NAME = "vehicles_50k"  # Try road_75k for road predictions
MODEL_ASSET_VERSION = 2


class CVT(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = CKPT_NAME) -> Self:
        WEIGHTS_URL = CachedWebModelAsset(
            f"https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_{ckpt_name}.ckpt",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"cvt_nuscenes_{ckpt_name}.ckpt",
        )
        checkpoint = load_torch(WEIGHTS_URL)
        cfg: Any = DictConfig(checkpoint["hyper_parameters"])

        cfg = OmegaConf.to_object(checkpoint["hyper_parameters"])
        cfg = DictConfig(cfg)

        state_dict = remove_prefix(checkpoint["state_dict"], "backbone")

        rewrite_hydra_targets(cfg, cvt_repo.__name__)
        model = setup_network(cfg)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model)

    def forward(
        self,
        image: torch.Tensor,
        inv_intrinsics: torch.Tensor,
        inv_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Cross-View Transformer model.

        Parameters
        ----------
        image
            Input image tensor for 6 cameras, with 3 color channels, shape [1, 6, 3, H, W].
        inv_intrinsics
            Inverse intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays, shape [1, 6, 3, 3].
        inv_extrinsics
            Inverse extrinsics tensor mapping world coordinates to camera coordinates, shape [1, 6, 4, 4].

        Returns
        -------
        bev : torch.Tensor
            BEV heatmap tensor with predictions, shape [1, 1, 200, 200].
        """
        out = self.model(
            {"image": image, "intrinsics": inv_intrinsics, "extrinsics": inv_extrinsics}
        )
        return out["bev"]

    def get_output_names(self) -> list[str]:
        return ["bev"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def get_evaluator(self) -> BaseEvaluator:
        return NuscenesBevSegmentationEvaluator()

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 4

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [NuscenesBevCVTDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NuscenesBevCVTDataset

    def get_input_spec(
        self,
        num_frames: int = 6,
        height: int = 224,
        width: int = 480,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, num_frames, 3, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "intrinsics": TensorSpec(
                shape=(1, num_frames, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "extrinsics": TensorSpec(
                shape=(1, num_frames, 4, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }
