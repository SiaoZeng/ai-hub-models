# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from timm.models.helpers import load_checkpoint
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.gpunet.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.gpunet.external_repos.deeplearningexamples.PyTorch.Classification.GPUNet.configs.model_hub import (
    get_configs,
)
from qai_hub_models.models.gpunet.external_repos.deeplearningexamples.PyTorch.Classification.GPUNet.models.gpunet_builder import (
    GPUNet_Builder,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

_GPUNET_ROOT = (
    EXTERNAL_REPO_PATHS["deeplearningexamples"]
    / "PyTorch"
    / "Classification"
    / "GPUNet"
)


class GPUNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls) -> Self:
        modelJSON, ckpt_path = get_configs(
            batch=1,
            latency="0.65ms",
            gpuType="GV100",
            config_root_dir=str(_GPUNET_ROOT / "configs"),
        )
        builder = GPUNet_Builder()
        model = builder.get_model(modelJSON)
        load_checkpoint(model, ckpt_path)

        return cls(model).eval()
