# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models._shared.deeplab.model import NUM_CLASSES, DeepLabV3Model
from qai_hub_models.models.deeplabv3_plus_mobilenet.external_repos.pytorch_deeplab_xception.modeling.deeplab import (
    DeepLab,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
# Weights downloaded from https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/deeplab-mobilenet.pth.tar
DEEPLABV3_WEIGHTS = "deeplab-mobilenet.pth.tar"
BACKBONE = "mobilenet"


class DeepLabV3PlusMobilenet(DeepLabV3Model):
    """Exportable DeepLabV3_Plus_MobileNet image segmentation applications, end-to-end."""

    @classmethod
    def from_pretrained(cls, normalize_input: bool = True) -> Self:
        model = _load_deeplabv3_source_model()
        dst = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEEPLABV3_WEIGHTS
        ).fetch()
        checkpoint = torch.load(
            dst, map_location=torch.device("cpu"), weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        return cls(model, normalize_input)


def _load_deeplabv3_source_model() -> torch.nn.Module:
    # Load DeepLabV3 model from the source repository using the given weights.
    # Returns <source repository>.modeling.deeplab.DeepLab
    return DeepLab(backbone=BACKBONE, sync_bn=False, num_classes=NUM_CLASSES)
