# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto import info_pb2
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry
from qai_hub_models_cli.proto_helpers.platform import (
    get_runtime_info,
    precision_proto_to_str,
    precision_str_to_proto,
    runtime_proto_to_str,
    runtime_str_to_proto,
)
from qai_hub_models_cli.versions import CURRENT_VERSION


class AssetNotFoundError(FileNotFoundError):
    def __init__(self, *args: object, model_sharing_restricted: bool = False) -> None:
        self.model_sharing_restricted = model_sharing_restricted
        super().__init__(*args)


@functools.lru_cache(maxsize=1)
def get_model_release_assets(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelReleaseAssets:
    """
    Fetch and cache the model release assets protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local release assets protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelReleaseAssets
        Parsed release assets protobuf containing download URLs for
        each available runtime, precision, and chipset combination.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    proto = fetch_model_proto(
        model,
        version,
        ModelReleaseAssets,
        cache_filename="release_assets.pb",
        manifest_url_field="release_assets",
        source_getter="get_release_assets_proto",
        local_path=local_path,
    )

    if not proto.assets:
        entry = get_manifest_entry(model, version)
        info_proto = get_model_info(model, version)

        if info_proto.HasField("llm_details"):
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_COMING_SOON
            ):
                raise AssetNotFoundError(
                    f"No pre-compiled model files are available for {entry.display_name}, but assets are coming soon. Stay tuned!"
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_US
            ):
                raise AssetNotFoundError(
                    f"If you have interest in downloading {entry.display_name}, reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_DOWNLOAD
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for download. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_PURCHASE
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for purchase. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )

        if info_proto.restrict_model_sharing:
            raise AssetNotFoundError(
                (
                    f"No pre-compiled model files for {entry.display_name} are available due to licensing"
                    " restrictions. You can use the AI Hub Models package to manually"
                    " export the model. See"
                    " https://github.com/qualcomm/ai-hub-models/tree/"
                )
                + (f"v{version}" if not version.is_devrelease else "main")
                + f"/src/qai_hub_models/models/{entry.id} for export instructions.",
                model_sharing_restricted=True,
            )

        raise AssetNotFoundError(
            f"No pre-compiled model files are available for {entry.display_name}. Reach out to us at"
            " qai-hub-support@qti.qualcomm.com."
        )

    return proto


def get_model_asset_details(
    model: str,
    runtime: Runtime.ValueType | str,
    precision: Precision.ValueType | str,
    chipset: str | None = None,
    version: Version = CURRENT_VERSION,
) -> ModelReleaseAssets.AssetDetails:
    """
    Look up a specific asset from a model's release assets.

    If the runtime is AOT-compiled (per the platform registry), *chipset*
    is required and a chipset-specific asset is returned. Otherwise
    *chipset* is ignored and a universal asset is returned.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name.
    runtime
        Runtime enum value (e.g. ``RUNTIME_TFLITE``) or string
        (e.g. ``"tflite"``).
    precision
        Precision enum value (e.g. ``PRECISION_FLOAT``) or string
        (e.g. ``"float"``).
    chipset
        Chipset name. Required for AOT-compiled runtimes, ignored otherwise.
    version
        AI Hub Models release version. Defaults to the installed CLI version.

    Returns
    -------
    ModelReleaseAssets.AssetDetails
        Matching asset entry with download URL, tool versions, etc.

    Raises
    ------
    KeyError
        If no matching asset is found, or if *chipset* is missing for
        an AOT-compiled runtime.
    """
    runtime_val = runtime_str_to_proto(runtime)
    precision_val = precision_str_to_proto(precision)
    rt_info = get_runtime_info(runtime_val, version)
    release_assets = get_model_release_assets(model, version)

    if rt_info.is_aot_compiled:
        if chipset is None:
            raise KeyError(f"Chipset is required for AOT-compiled runtime {runtime!r}.")
        for asset in release_assets.assets:
            if (
                asset.runtime == runtime_val
                and asset.precision == precision_val
                and asset.chipset == chipset
            ):
                return asset
    else:
        for asset in release_assets.assets:
            if asset.runtime == runtime_val and asset.precision == precision_val:
                return asset

    available = ", ".join(
        f"{runtime_proto_to_str(a.runtime)}/{precision_proto_to_str(a.precision)}"
        + (f"/{a.chipset}" if a.HasField("chipset") else "")
        for a in release_assets.assets
    )
    raise AssetNotFoundError(
        f"No asset found for model={model!r} with runtime={runtime!r}, "
        f"precision={precision!r}, chipset={chipset!r} (version={version}).\n"
        f"Available: {available}"
    )
