# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.models._shared.efficientvit.external_repos.efficientvit.efficientvit.seg_model_zoo import (
    create_seg_model,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "l2.pt"
MODEL_ASSET_VERSION = 1


class EfficientViT(CityscapesSegmentor):
    """Exportable EfficientViT Image segmentation, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str | None = None) -> Self:
        """Load EfficientViT from a weightfile created by the source repository."""
        if not weights:
            weights = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        efficientvit_model = create_seg_model(
            name="l2", dataset="cityscapes", weight_url=weights
        )
        efficientvit_model.to(torch.device("cpu"))
        efficientvit_model.eval()
        return cls(efficientvit_model)
