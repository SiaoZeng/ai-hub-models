# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.versions import CURRENT_VERSION


@functools.lru_cache(maxsize=1)
def get_model_perf(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelPerf:
    """
    Fetch and cache the model perf protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local perf protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelPerf
        Parsed model perf protobuf containing per-device performance
        metrics such as inference time, memory usage, and layer counts.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    return fetch_model_proto(
        model,
        version,
        ModelPerf,
        cache_filename="perf.pb",
        manifest_url_field="perf",
        source_getter="get_perf_proto",
        local_path=local_path,
    )
