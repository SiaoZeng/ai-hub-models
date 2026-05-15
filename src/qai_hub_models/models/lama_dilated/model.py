# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models.models._shared.repaint.model import RepaintModel
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.lama_dilated.external_repos.lama.saicinpainting.training.trainers.default import (
    DefaultInpaintingTrainingModule,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_json,
    load_torch,
    set_log_level,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "lama-dilated_celeba-hq"
MODEL_ASSET_VERSION = 2


class LamaDilated(RepaintModel):
    """Exportable LamaDilated inpainting algorithm by Samsung Research."""

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime == TargetRuntime.QNN_DLC:
            other_compile_options += " -O2"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS) -> Self:
        """Load LamaDilated from a weights file created by the source LaMa repository."""
        # Load PyTorch model from disk
        lama_dilated_model = _load_lama_dilated_source_model_from_weights(weights_name)
        return cls(lama_dilated_model)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run LamaDilated on `image` and `mask`, and produce an image with mask area inpainted.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB
        mask
            Pixel values pre-processed to have have mask values either 0. or 1.
            Range: float[0, 1] and only values of 0. or 1.
            1-channel binary image.

        Returns
        -------
        inpainted_image : torch.Tensor
            Pixel values
            Range: float[0, 1]
            3-channel Color Space: RGB
        """
        masked_img = image * (1 - mask)

        if self.model.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        predicted_image = self.model.generator(masked_img)  # type: ignore[operator]
        return mask * predicted_image + (1 - mask) * image


def _get_weightsfile_from_name(weights_name: str) -> CachedWebModelAsset:
    """Convert from names of weights files to the url for the weights file"""
    return CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, f"checkpoints/{weights_name}.ckpt"
    )


def _get_config_url() -> CachedWebModelAsset:
    """Get the url for the config file"""
    return CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "checkpoints/training_config.json"
    )


def _load_lama_dilated_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    weights_url = _get_weightsfile_from_name(weights_name)
    config_url = _get_config_url()

    # Pass config as needed to create the module for tracing.
    config_json = load_json(config_url)
    config = OmegaConf.create(config_json)
    kwargs = dict(config.training_model)
    kwargs.pop("kind")
    kwargs["use_ddp"] = True
    state = load_torch(weights_url)
    with set_log_level(logging.WARN):
        lama_dilated_model = DefaultInpaintingTrainingModule(config, **kwargs)
        # Needed for pytorch-lightning to script the module appropriately.
        lama_dilated_model._jit_is_scripting = True

    lama_dilated_model.load_state_dict(state["state_dict"], strict=False)
    lama_dilated_model.on_load_checkpoint(state)
    lama_dilated_model.freeze()

    return lama_dilated_model
