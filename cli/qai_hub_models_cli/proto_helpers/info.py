# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto.info_pb2 import ModelInfo
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.versions import CURRENT_VERSION


@functools.lru_cache(maxsize=1)
def get_model_info(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelInfo:
    """
    Fetch and cache the model info protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local info protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelInfo
        Parsed model info protobuf containing metadata such as name,
        description, domain, use case, tags, and license.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    return fetch_model_proto(
        model,
        version,
        ModelInfo,
        cache_filename="info.pb",
        manifest_url_field="info",
        source_getter="get_info_proto",
        local_path=local_path,
    )
