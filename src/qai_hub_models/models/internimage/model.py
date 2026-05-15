# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from types import SimpleNamespace

import torch
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.internimage.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.internimage.external_repos.internimage.classification.config import (
    get_config,
)
from qai_hub_models.models.internimage.external_repos.internimage.classification.models import (
    build_model,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "internimage_t_1k_224.pth"
DEFAULT_CONFIG_PATH = "internimage_t_1k_224.yaml"
MODEL_ASSET_VERSION = 1
NUM_CLASSES = 1000
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "cupcake.jpg"
)
INTERNIMAGE_REPO_PATH = EXTERNAL_REPO_PATHS["internimage"]


class InternImageClassifier(ImagenetClassifier):
    """Exportable InternImage classifier."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(net=model, transform_input=False, normalize_input=True)

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | None = None, config_path: str | None = None
    ) -> Self:
        """
        Load InternImage classifier from pretrained weights.

        Parameters
        ----------
        checkpoint_path
            Path to a pretrained model checkpoint. If None, the default checkpoint
            will be fetched from the asset store.
        config_path
            Path to a config file. If None, the default config will be used.

        Returns
        -------
        model : Self
            An instance of the classifier with the model loaded and ready for inference.
        """
        if not config_path:
            config_path = str(
                INTERNIMAGE_REPO_PATH
                / "classification"
                / "configs"
                / DEFAULT_CONFIG_PATH
            )

        args = SimpleNamespace(cfg=config_path)
        config = get_config(args)
        model = build_model(config)

        if not checkpoint_path:
            checkpoint_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)

        return cls(model)
