# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.datasets.mpiigaze import MPIIGazeDataset
from qai_hub_models.evaluators.mpigaze_evaluator import MPIIGazeEvaluator
from qai_hub_models.models.eyegaze.external_repos.gaze_estimation.models.eyenet import (
    EyeNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "default"

DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    f"{DEFAULT_WEIGHTS}.pt",
)


class EyeGaze(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS) -> Self:
        weights_file = weights_name
        if weights_name == DEFAULT_WEIGHTS:
            weights_file = DEFAULT_WEIGHTS_FILE
        checkpoint = load_torch(weights_file)

        nstack = checkpoint["nstack"]
        nfeatures = checkpoint["nfeatures"]
        nlandmarks = checkpoint["nlandmarks"]
        eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks)
        eyenet.load_state_dict(checkpoint["model_state_dict"])
        return cls(eyenet)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(image)

    def get_output_names(self) -> list[str]:
        return ["heatmaps", "landmarks", "gaze_pitchyaw"]

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
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    def get_evaluator(self) -> BaseEvaluator:
        return MPIIGazeEvaluator()

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        options = " --compute_unit cpu"  # Accuracy no regained on NPU
        return profile_options + options

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [MPIIGazeDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return MPIIGazeDataset

    def get_input_spec(
        self,
        height: int = 96,
        width: int = 160,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }
