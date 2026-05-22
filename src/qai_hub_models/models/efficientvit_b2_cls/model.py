# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models._shared.efficientvit.external_repos.efficientvit.efficientvit.cls_model_zoo import (
    create_cls_model,
)
from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "b2-r288.pt"
MODEL_ASSET_VERSION = 1


class EfficientViT(ImagenetClassifier):
    """Exportable EfficientViT Image classifier, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str | None = None) -> Self:
        """Load EfficientViT from a weightfile created by the source repository."""
        if not weights:
            weights = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        efficientvit_model = create_cls_model(name="b2", weight_url=weights)
        efficientvit_model.to(torch.device("cpu"))
        efficientvit_model.eval()
        return cls(efficientvit_model)

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
