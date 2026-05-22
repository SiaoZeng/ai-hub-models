# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from hydra import compose, initialize_config_dir
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.nuscenes_bev import NuscenesBevGKTDataset
from qai_hub_models.evaluators.nuscenes_bev_evaluator import (
    NuscenesBevSegmentationEvaluator,
)
from qai_hub_models.models.gkt.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.gkt.external_repos.gkt.segmentation.cross_view_transformer.common import (
    remove_prefix,
    setup_network,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Checkpoint is sourced from
GKT_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "map_segmentation_gkt_7x1_conv_setting2.ckpt"
)


class GKT(BaseModel):
    """GKT BEV Object Detection"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = model
        self.encoder: Any = self.model.encoder
        self.decoder: Any = self.model.decoder

    @classmethod
    def from_pretrained(cls, ckpt_name: str | None = None) -> Self:
        if ckpt_name:
            checkpoint = load_torch(ckpt_name)
        else:
            checkpoint = load_torch(GKT_CKPT.fetch())

        config_path = EXTERNAL_REPO_PATHS["gkt"] / "segmentation" / "config"
        with initialize_config_dir(version_base=None, config_dir=str(config_path)):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=gkt_nuscenes_vehicle_kernel_7x1"],
            )
        state_dict = remove_prefix(checkpoint["state_dict"], "backbone")
        model = setup_network(cfg)
        model.load_state_dict(state_dict)
        return cls(model)

    def forward(
        self,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        inv_intrinsics: torch.Tensor,
        inv_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for GKT model.

        Parameters
        ----------
        image
            Input image tensor for 6 cameras with 3 color channels, shape [B, N, 3, H, W].
        intrinsics
            Intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays,
            shape [B, N, 3, 3].
        extrinsics
            Extrinsics tensor mapping world coordinates to camera coordinates,
            shape [B, N, 4, 4].
        inv_intrinsics
            Inverse intrinsics tensor mapping 2D pixel coordinates to 3D camera-space
            rays, shape [B, N, 3, 3].
        inv_extrinsics
            Inverse extrinsics tensor mapping world coordinates to camera coordinates,
            shape [B, N, 4, 4].

        Returns
        -------
        bev : torch.Tensor
            BEV heatmap tensor with predictions, shape [B, 1, 200, 200].
        """
        image = image.flatten(0, 1)

        features = [
            self.encoder.down(y)
            for y in self.encoder.backbone(self.encoder.norm(image))
        ]

        x = self.encoder.bev_embedding.get_prior()

        for cross_view, feature, layer in zip(
            self.encoder.cross_views, features, self.encoder.layers, strict=False
        ):
            x = cross_view(
                x,
                self.encoder.bev_embedding.grid,
                feature,
                inv_intrinsics,
                inv_extrinsics,
                intrinsics,
                extrinsics,
            )
            x = layer(x)
        y = self.decoder(x)
        z = self.model.to_logits(y)
        return z.split(1, dim=1)[0]

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_cams: int = 6,
        height: int = 224,
        width: int = 480,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, num_cams, 3, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "intrinsics": TensorSpec(
                shape=(batch_size, num_cams, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "extrinsics": TensorSpec(
                shape=(batch_size, num_cams, 4, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "inv_intrinsics": TensorSpec(
                shape=(batch_size, num_cams, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "inv_extrinsics": TensorSpec(
                shape=(batch_size, num_cams, 4, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_names(self) -> list[str]:
        return ["bev"]

    def get_evaluator(self) -> BaseEvaluator:
        return NuscenesBevSegmentationEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [NuscenesBevGKTDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NuscenesBevGKTDataset

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
