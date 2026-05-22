# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from qai_hub_models.evaluators.denoising_evaluator import DenoisingEvaluator
from qai_hub_models.models._shared.nafnet.model_patches import (
    AutoLayerNorm2d,
    NAFLocal_Base,
    ssrforward,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_yaml,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

NAFNET_SOURCE_REPOSITORY = "https://github.com/megvii-research/NAFNet.git"
NAFNET_SOURCE_REPO_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
NAFNetAsRoot = partial(
    SourceAsRoot,
    NAFNET_SOURCE_REPOSITORY,
    NAFNET_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
)


class NAFNetModel(BaseModel):
    """Base Model for NAFNet."""

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run NAFNet on image, and produce denoised/deblurred image.

        Parameters
        ----------
        image
            Input image to be processed
            Pixel values in range [0, 1], RGB color space.

        Returns
        -------
        restored_image : torch.Tensor
            Restored image
            Pixel values in range [0, 1], RGB color space.
        """
        return self.model(image)

    def get_evaluator(self) -> BaseEvaluator:
        return DenoisingEvaluator()


def _load_nafnet_source_model(
    nafnet_weights: str | Path,
    yaml_path_nafnet: str | Path,
    MODEL_ID: str,
    MODEL_ASSET_VERSION: int,
) -> torch.nn.Module:
    """Load the NAFNet source model.

    Parameters
    ----------
    nafnet_weights
        Remote asset filename (``str``) fetched from the asset store, or a
        local ``Path`` to an existing ``.pth`` weights file.
    yaml_path_nafnet
        Repo-relative config path under ``options/test/`` (``str``), or a
        local ``Path`` to an existing ``.yml`` config file.
    MODEL_ID
        Model identifier used for asset store.
    MODEL_ASSET_VERSION
        Asset version number used for asset store.


    Returns
    -------
    torch.nn.Module
        The ``net_g`` submodule of the loaded NAFNet model.
    """
    if isinstance(nafnet_weights, Path):
        weights_path_nafnet = nafnet_weights
        if not weights_path_nafnet.exists():
            raise FileNotFoundError(f"Local weights file not found: {nafnet_weights}")
    else:
        weights_path_nafnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, nafnet_weights
        ).fetch()

    with NAFNetAsRoot() as repo_path:
        import basicsr.models.archs.NAFNet_arch as NAFarch
        import basicsr.models.archs.NAFSSR_arch as NAFSSRarch

        NAFarch.LayerNorm2d = AutoLayerNorm2d
        NAFSSRarch.LayerNorm2d = AutoLayerNorm2d

        NAFarch.Local_Base.convert = NAFLocal_Base.convert
        NAFSSRarch.Local_Base.convert = NAFLocal_Base.convert
        NAFSSRarch.NAFNetSR.forward = ssrforward

        from basicsr.models import create_model

        # Handle config path
        if isinstance(yaml_path_nafnet, Path):
            yaml_path = yaml_path_nafnet.resolve()
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"Local config file not found: {yaml_path_nafnet}"
                )
        else:
            # Load YAML from the cloned repo
            yaml_path = Path(repo_path, "options/test", yaml_path_nafnet).resolve()

        opt = load_yaml(yaml_path)

        opt["is_train"] = False
        opt["dist"] = False
        opt["num_gpu"] = 0
        opt["path"]["pretrain_network_g"] = str(weights_path_nafnet)

        model = create_model(opt)

    return model.net_g
