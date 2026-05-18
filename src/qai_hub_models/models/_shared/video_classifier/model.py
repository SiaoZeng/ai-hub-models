# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.video_classification_evaluator import (
    VideoClassificationEvaluator,
)
from qai_hub_models.models._shared.video_classifier.utils import (
    preprocess_video_kinetics_400,
    read_video_at_fps,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = "video_classifier"
MODEL_ASSET_VERSION = 1

INPUT_VIDEO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "surfing_cutback.mp4"
)

# 5 temporal clips x 1 center crop matches the torchvision reference eval
# protocol for R3D / R2+1D / MC3 and is the default single-crop setting
# for VideoMAE.
DEFAULT_NUM_CLIPS = 5
DEFAULT_NUM_CROPS = 1
DEFAULT_NUM_VIEWS = DEFAULT_NUM_CLIPS * DEFAULT_NUM_CROPS


class SimpleAvgPool(torch.nn.Module):
    """
    Replacement for Global Average Pool that's numerically equivalent to the one used
    in the torchvision models. It operates in rank 3 instead of rank 5 which makes it
    more NPU friendly.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return tensor.reshape(shape[0], shape[1], -1).mean(dim=2, keepdim=False)


class KineticsClassifier(BaseModel):
    """Base class for all Kinetics Classifier models within QAI Hub Models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        # TODO: rename input_mean/input_std back to mean/std when
        # #https://github.com/pytorch/pytorch/issues/168211 is fixed
        self.input_mean = torch.Tensor([0.43216, 0.394666, 0.37645]).reshape(
            1, 3, 1, 1, 1
        )
        self.input_std = torch.Tensor([0.22803, 0.22145, 0.216989]).reshape(
            1, 3, 1, 1, 1
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for a single multi-view video.

        Parameters
        ----------
        video
            Shape ``[B, V*3, T, H, W]`` where ``V = num_clips * num_crops``.
            Assumes video has been resized and normalized to range [0, 1]
            3-channel Color Space: RGB.

        Returns
        -------
        class_probs : torch.Tensor
            Shape ``[B, 400]`` — summed softmax probabilities across all views.
        """
        # [B, V*3, T, H, W] -> [B*V, 3, T, H, W] with explicit sizes so the
        # torchscript tracer keeps the output rank for ONNX export.
        B, VC, T, H, W = video.shape
        V = VC // 3
        flat = video.view(B * V, 3, T, H, W)
        flat = (flat - self.input_mean) / self.input_std
        logits = self.model(flat)
        probs = torch.softmax(logits, dim=1)
        return probs.view(B, V, -1).sum(dim=1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_frames: int = 16,
        num_views: int = DEFAULT_NUM_VIEWS,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.

        Parameters
        ----------
        batch_size
            Batch dimension of the input tensor.
        num_frames
            Number of frames per clip.
        num_views
            Number of views (temporal clips x spatial crops) per video,
            packed along the channel dim with ``C=3``.

        Returns
        -------
        input_spec : InputSpec
            Input specification mapping input names to (shape, type) tuples.
        """
        return {
            "video": TensorSpec(
                shape=(batch_size, num_views * 3, num_frames, 112, 112),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        input_tensor = read_video_at_fps(str(INPUT_VIDEO_PATH.fetch()), target_fps=15)
        input_tensor = preprocess_video_kinetics_400(input_tensor)
        num_views = DEFAULT_NUM_VIEWS
        if input_spec:
            num_frames = input_spec["video"][0][2]
            num_views = input_spec["video"][0][1] // 3
            input_tensor = input_tensor[:, :num_frames]
        # Replicate the single clip `num_views` times packed along channel.
        C, T, H, W = input_tensor.shape
        return {
            "video": [
                input_tensor.unsqueeze(0)
                .expand(num_views, -1, -1, -1, -1)
                .reshape(num_views * C, T, H, W)
                .unsqueeze(0)
                .numpy()
            ]
        }

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return {
            "class_probs": TensorSpec(
                io_type=IoType.TENSOR,
                softmax_applied=True,
            ),
        }

    def get_channel_last_inputs(self) -> list[str]:
        return ["video"]

    def get_evaluator(self) -> BaseEvaluator:
        return VideoClassificationEvaluator(num_classes=400)

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["kinetics400"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "kinetics400"
