# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import SuperPointForKeypointDetection
from typing_extensions import Self

from qai_hub_models.datasets.hpatches import HPatchesDataset
from qai_hub_models.evaluators.superpoint_evaluator import SuperPointEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
HF_MODEL_ID = "magic-leap-community/superpoint"

# Input images are sourced from the hpatch dataset
INPUT_IMAGE_ADDRESS_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "melon_1.ppm"
)
INPUT_IMAGE_ADDRESS_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "melon_2.ppm"
)


class SuperPoint(BaseModel):
    """Exportable SuperPoint interest-point detector and descriptor."""

    CELL_SIZE = 8
    DESCRIPTOR_DIM = 256

    def __init__(
        self,
        model: SuperPointForKeypointDetection,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = 1000,
        nms_radius: int = 8,
        border_removal_distance: int = 4,
    ) -> None:
        super().__init__(model)
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str = HF_MODEL_ID,
        max_keypoints: int = 1000,
        keypoint_threshold: float = 0.005,
        nms_radius: int = 8,
        border_removal_distance: int = 4,
    ) -> Self:
        """Load SuperPoint from a HuggingFace repo ID or local path."""
        hf_model = SuperPointForKeypointDetection.from_pretrained(weights_path)
        return cls(
            hf_model,
            keypoint_threshold=keypoint_threshold,
            max_keypoints=max_keypoints,
            nms_radius=nms_radius,
            border_removal_distance=border_removal_distance,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and compute descriptors for a batch of image pairs.

        Parameters
        ----------
        image
            Shape ``(B, N, 1, H, W)``, float32 in ``[0, 1]``.
            B is batch size, N is number of images per sample (typically 2),
            1 is the grayscale channel, H and W are height and width.
            The 5D input is reshaped to 4D ``(B*N, 1, H, W)`` before encoder,
            so the core CNN operations run on standard 4D tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            keypoints
                Shape ``(B, N, K, 2)``, float32. Keypoint (x, y) pixel coordinates,
                sorted by score descending. K is max_keypoints.
            scores
                Shape ``(B, N, K)``, float32. Keypoint confidence scores in ``[0, 1]``.
                Zero-padded for positions below keypoint_threshold.
            descriptors
                Shape ``(B, N, K, 256)``, float32. L2-normalized descriptors.
        """
        b, num_images, c, h, w = image.shape
        bn = b * num_images
        flat_image = image.reshape(bn, c, h, w)

        last_hidden_state = self.model.encoder(  # type: ignore[operator]
            flat_image, output_hidden_states=False, return_dict=False
        )[0]

        scores = self.model.keypoint_decoder.relu(  # type: ignore[call-arg,operator]
            self.model.keypoint_decoder.conv_score_a(last_hidden_state)  # type: ignore[operator,union-attr]
        )
        scores = self.model.keypoint_decoder.conv_score_b(scores)  # type: ignore[operator,union-attr]
        scores = F.softmax(scores, 1)[:, :-1]
        _, _, fh, fw = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(
            bn, fh, fw, self.CELL_SIZE, self.CELL_SIZE
        )
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            bn, fh * self.CELL_SIZE, fw * self.CELL_SIZE
        )

        descriptors = self.model.descriptor_decoder.relu(  # type: ignore[call-arg,operator]
            self.model.descriptor_decoder.conv_descriptor_a(last_hidden_state)  # type: ignore[operator,union-attr]
        )
        descriptors = self.model.descriptor_decoder.conv_descriptor_b(descriptors)  # type: ignore[operator,union-attr]
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Custom NMS: HF uses score-based dynamic indexing which is not export-friendly.
        # We replace it with a fixed-shape max-pool dilation + two iterative suppression passes.
        nms_kernel = self.nms_radius * 2 + 1
        nms_pad = self.nms_radius
        zeros = torch.zeros_like(scores)
        scores_4d = scores.unsqueeze(1)

        max_mask = scores == F.max_pool2d(
            scores_4d, kernel_size=nms_kernel, stride=1, padding=nms_pad
        ).reshape(bn, h, w)

        # Two passes of iterative suppression to catch near-maxima that survive a single dilation round.
        for _ in range(2):
            supp = (
                F.max_pool2d(
                    max_mask.float().unsqueeze(1),
                    kernel_size=nms_kernel,
                    stride=1,
                    padding=nms_pad,
                ).reshape(bn, h, w)
                > 0
            )
            supp_scores = torch.where(supp, zeros, scores)
            new_max = supp_scores == F.max_pool2d(
                supp_scores.unsqueeze(1),
                kernel_size=nms_kernel,
                stride=1,
                padding=nms_pad,
            ).reshape(bn, h, w)
            max_mask = max_mask | (new_max & ~supp)

        scores = torch.where(max_mask, scores, zeros)

        # Custom threshold + border removal: HF uses boolean indexing which produces a
        # variable-length tensor. We zero out invalid positions to preserve a fixed shape.
        scores = torch.where(scores > self.keypoint_threshold, scores, zeros)
        border = self.border_removal_distance
        border_mask = torch.zeros_like(scores, dtype=torch.bool)
        border_mask[:, border : h - border, border : w - border] = True
        scores = torch.where(border_mask, scores, zeros)

        # Custom top-k selection: HF returns a variable number of keypoints per image.
        # We select the top ``max_keypoints`` scores to produce a fixed output shape.
        scores_flat = scores.reshape(bn, h * w)
        k = min(self.max_keypoints, h * w) if self.max_keypoints > 0 else h * w
        top_scores, top_indices = torch.topk(scores_flat, k=k, dim=1)

        # Custom coordinate conversion: replace ``torch.div(..., rounding_mode='floor')``
        # and modulo with subtraction-based arithmetic for broader export compatibility.
        rows = top_indices // w
        cols = top_indices - rows * w
        keypoints = torch.stack([cols, rows], dim=2).to(top_scores.dtype)

        sampled = self.model.descriptor_decoder._sample_descriptors(  # type: ignore[operator,union-attr]
            keypoints.unsqueeze(1), descriptors, scale=self.CELL_SIZE
        )
        sampled = F.normalize(sampled.permute(0, 2, 1), p=2, dim=2)

        return (
            keypoints.reshape(b, num_images, k, 2),
            top_scores.reshape(b, num_images, k),
            sampled.reshape(b, num_images, k, self.DESCRIPTOR_DIM),
        )

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_images: int = 2,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        cell_size = SuperPoint.CELL_SIZE
        assert height % cell_size == 0 and width % cell_size == 0, (
            f"Input H,W must be divisible by {cell_size}, got {height}x{width}"
        )
        return {"image": ((batch_size, num_images, 1, height, width), "float32")}

    def get_output_names(self) -> list[str]:
        return ["keypoints", "scores", "descriptors"]

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [HPatchesDataset]

    def get_evaluator(self) -> BaseEvaluator:
        spec = self.get_input_spec()
        _, _, _, h, w = spec["image"][0]
        return SuperPointEvaluator(image_height=h, image_width=w)
