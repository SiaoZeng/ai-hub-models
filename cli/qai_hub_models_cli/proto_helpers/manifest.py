# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import os
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.common import STORE_URL
from qai_hub_models_cli.proto.manifest_pb2 import (
    ManifestModelEntry,
    ReleaseManifest,
)
from qai_hub_models_cli.proto_helpers._common import fetch_release_proto
from qai_hub_models_cli.versions import CURRENT_VERSION


def get_manifest_entry(
    model: str, version: Version = CURRENT_VERSION
) -> ManifestModelEntry:
    manifest = get_manifest(version)
    model_lower = model.lower()
    for entry in manifest.models:
        if model_lower in (entry.id.lower(), entry.display_name.lower()):
            return entry
    raise KeyError(
        f"No model exists with the name or ID: {model!r}. "
        "Run qai-hub-models models to see all Model IDs."
    )


@functools.lru_cache(maxsize=1)
def get_manifest(
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ReleaseManifest:
    """
    Fetch and cache the release manifest for a given version.

    The manifest lists every model in the release along with URLs
    for its info, perf, numerics, and release-assets protobufs.

    Parameters
    ----------
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local manifest protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ReleaseManifest
        Parsed manifest protobuf.

    Raises
    ------
    UnsupportedVersionError
        If *version* is not a supported release (when *local_path* is None).
    """
    url = os.path.join(
        STORE_URL, "qai-hub-models", "releases", f"v{version}", "manifest.pb"
    )
    return fetch_release_proto(
        version,
        ReleaseManifest,
        cache_filename="manifest.pb",
        source_getter="get_manifest_proto",
        url=url,
        local_path=local_path,
    )
