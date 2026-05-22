# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from qai_hub_models_cli.proto import release_assets_pb2

from qai_hub_models import Precision
from qai_hub_models.configs.proto_helpers import precision_to_proto, runtime_to_proto
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


class QAIHMModelReleaseAssets(BaseQAIHMConfig):
    """Schema for model release_assets.yaml files."""

    # Version of QAIHM (only present in manifest.yaml, not release-assets.yaml)
    version: str | None = None

    class AssetDetails(BaseQAIHMConfig):
        # Key for object in the AI Hub Models S3 Bucket
        # (see qai_hub_models/utils/aws.py)
        # Optional for manifest.yaml (uses download_url instead)
        s3_key: str | None = None

        # Tool versions used to generate this asset.
        tool_versions: ToolVersions | None = None

        # Optional: Public download URL (populated for manifest generation)
        download_url: str | None = None

    class PrecisionDetails(BaseQAIHMConfig):
        universal_assets: dict[
            ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails
        ] = Field(default_factory=dict)
        chipset_assets: dict[
            str,
            dict[ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails],
        ] = Field(default_factory=dict)

    precisions: dict[Precision, QAIHMModelReleaseAssets.PrecisionDetails] = Field(
        default_factory=dict
    )

    @property
    def empty(self) -> bool:
        return not self.precisions

    def add_asset(
        self,
        details: QAIHMModelReleaseAssets.AssetDetails,
        precision: Precision,
        chipset: str | None,
        path: ScorecardProfilePath,
    ) -> None:
        if precision not in self.precisions:
            self.precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails()
        if chipset:
            if chipset not in self.precisions[precision].chipset_assets:
                self.precisions[precision].chipset_assets[chipset] = {}
            self.precisions[precision].chipset_assets[chipset][path] = details
        else:
            self.precisions[precision].universal_assets[path] = details

    def get_asset(
        self,
        precision: Precision,
        chipset: str | None,
        path: ScorecardProfilePath,
    ) -> QAIHMModelReleaseAssets.AssetDetails | None:
        if precision not in self.precisions:
            return None
        if chipset:
            if chipset not in self.precisions[precision].chipset_assets:
                return None
            return self.precisions[precision].chipset_assets[chipset].get(path)
        return self.precisions[precision].universal_assets.get(path)

    def to_proto(
        self, aihm_version: str, model_id: str
    ) -> release_assets_pb2.ModelReleaseAssets:
        assets: list[release_assets_pb2.ModelReleaseAssets.AssetDetails] = []
        for precision, prec_details in self.precisions.items():
            for path, asset in prec_details.universal_assets.items():
                if asset.download_url:
                    assets.append(
                        release_assets_pb2.ModelReleaseAssets.AssetDetails(
                            precision=precision_to_proto(precision),
                            runtime=runtime_to_proto(path.runtime),
                            download_url=asset.download_url,
                            tool_versions=asset.tool_versions.to_proto()
                            if asset.tool_versions
                            else None,
                        )
                    )
            for chipset, path_dict in prec_details.chipset_assets.items():
                for path, asset in path_dict.items():
                    if asset.download_url:
                        assets.append(
                            release_assets_pb2.ModelReleaseAssets.AssetDetails(
                                precision=precision_to_proto(precision),
                                runtime=runtime_to_proto(path.runtime),
                                chipset=chipset,
                                download_url=asset.download_url,
                                tool_versions=asset.tool_versions.to_proto()
                                if asset.tool_versions
                                else None,
                            )
                        )

        return release_assets_pb2.ModelReleaseAssets(
            aihm_version=aihm_version,
            model_id=model_id,
            assets=assets,
        )

    @classmethod
    def from_model(
        cls: type[QAIHMModelReleaseAssets], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelReleaseAssets:
        assets_path = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        if not_exists_ok and not os.path.exists(assets_path):
            return QAIHMModelReleaseAssets()
        return cls.from_yaml(assets_path)

    def to_model_yaml(self, model_id: str) -> Path:
        out = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        self.to_yaml(out)
        return out
