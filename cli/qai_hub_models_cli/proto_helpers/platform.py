# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto.info_pb2 import (
    ModelDomain,
    ModelLicense,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo, RuntimeInfo
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers._common import fetch_release_proto
from qai_hub_models_cli.proto_helpers.manifest import get_manifest
from qai_hub_models_cli.versions import CURRENT_VERSION


@functools.lru_cache(maxsize=1)
def get_platform(
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> PlatformInfo:
    """
    Fetch and cache the platform info protobuf for a given version.

    Contains the registry of devices, chipsets, form factors, and
    runtimes supported by AI Hub.

    Parameters
    ----------
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local platform protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    PlatformInfo
        Parsed platform info protobuf.

    Raises
    ------
    UnsupportedVersionError
        If *version* is not a supported release (when *local_path* is None).
    """
    if local_path is not None:
        url = None
    else:
        manifest = get_manifest(version)
        url = manifest.platform_url
    return fetch_release_proto(
        version,
        PlatformInfo,
        cache_filename="platform.pb",
        source_getter="get_platform_proto",
        url=url,
        local_path=local_path,
    )


def get_runtime_info(
    runtime: Runtime.ValueType | str,
    version: Version = CURRENT_VERSION,
) -> RuntimeInfo:
    """
    Look up the ``RuntimeInfo`` for a given runtime.

    Parameters
    ----------
    runtime
        Runtime enum value (e.g. ``RUNTIME_TFLITE``) or string
        (e.g. ``"tflite"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.

    Returns
    -------
    RuntimeInfo
        Platform runtime entry containing ``is_aot_compiled``,
        ``file_extension``, etc.

    Raises
    ------
    KeyError
        If *runtime* is not found in the platform registry.
    """
    runtime_val = runtime_str_to_proto(runtime)
    platform = get_platform(version)
    for rt in platform.runtimes:
        if rt.runtime == runtime_val:
            return rt
    runtime_name = (
        runtime if isinstance(runtime, str) else runtime_proto_to_str(runtime)
    )
    raise KeyError(f"Runtime {runtime_name!r} not found in platform registry.")


def precision_proto_to_str(precision: Precision.ValueType) -> str:
    """
    Convert a Precision proto enum value to its lowercase string name.

    Parameters
    ----------
    precision
        ``Precision`` enum value (e.g. ``PRECISION_FLOAT``).

    Returns
    -------
    str
        Lowercase name without the ``PRECISION_`` prefix (e.g. ``"float"``).

    Raises
    ------
    KeyError
        If *precision* is not a valid enum value.
    """
    name = Precision.Name(precision)
    if not name.startswith("PRECISION_"):
        raise KeyError(f"Unknown precision value: {precision!r}")
    return name.removeprefix("PRECISION_").lower()


def precision_str_to_proto(precision: str | Precision.ValueType) -> Precision.ValueType:
    """
    Convert a precision string to its proto enum value.

    Parameters
    ----------
    precision
        Precision name (e.g. ``"float"``, ``"w8a8"``, ``"PRECISION_MXFP4"``).
        Case-insensitive. The ``PRECISION_`` prefix is optional.

    Returns
    -------
    Precision.ValueType
        Corresponding ``Precision`` enum value.

    Raises
    ------
    KeyError
        If *precision* does not match any known precision.
    """
    if not isinstance(precision, str):
        return precision

    key = precision.upper()
    if not key.startswith("PRECISION_"):
        key = f"PRECISION_{key}"
    try:
        return Precision.Value(key)
    except ValueError:
        valid = ", ".join(
            name.removeprefix("PRECISION_").lower()
            for name in Precision.DESCRIPTOR.values_by_name
            if name != "PRECISION_UNSPECIFIED"
        )
        raise KeyError(
            f"Unknown precision: {precision!r}. Valid precisions: {valid}"
        ) from None


def runtime_proto_to_str(runtime: Runtime.ValueType) -> str:
    """
    Convert a Runtime proto enum value to its lowercase string name.

    Parameters
    ----------
    runtime
        ``Runtime`` enum value (e.g. ``RUNTIME_TFLITE``).

    Returns
    -------
    str
        Lowercase name without the ``RUNTIME_`` prefix (e.g. ``"tflite"``).

    Raises
    ------
    KeyError
        If *runtime* is not a valid enum value.
    """
    name = Runtime.Name(runtime)
    if not name.startswith("RUNTIME_"):
        raise KeyError(f"Unknown runtime value: {runtime!r}")
    return name.removeprefix("RUNTIME_").lower()


def runtime_str_to_proto(runtime: str | Runtime.ValueType) -> Runtime.ValueType:
    """
    Convert a runtime string to its proto enum value.

    Parameters
    ----------
    runtime
        Runtime name (e.g. ``"tflite"``, ``"qnn_dlc"``, ``"RUNTIME_ONNX"``).
        Case-insensitive. The ``RUNTIME_`` prefix is optional.

    Returns
    -------
    Runtime.ValueType
        Corresponding ``Runtime`` enum value.

    Raises
    ------
    KeyError
        If *runtime* does not match any known runtime.
    """
    if not isinstance(runtime, str):
        return runtime

    key = runtime.upper()
    if not key.startswith("RUNTIME_"):
        key = f"RUNTIME_{key}"
    try:
        return Runtime.Value(key)
    except ValueError:
        valid = ", ".join(
            name.removeprefix("RUNTIME_").lower()
            for name in Runtime.DESCRIPTOR.values_by_name
            if name != "RUNTIME_UNSPECIFIED"
        )
        raise KeyError(
            f"Unknown runtime: {runtime!r}. Valid runtimes: {valid}"
        ) from None


def domain_proto_to_str(domain: int) -> str:
    """Convert a ModelDomain enum value to a human-readable string."""
    name = ModelDomain.Name(domain)  # type: ignore[arg-type]
    return (
        name.removeprefix("MODEL_DOMAIN_").replace("_", " ").title().replace("Ai", "AI")
    )


def use_case_proto_to_str(use_case: int) -> str:
    """Convert a ModelUseCase enum value to a human-readable string."""
    name = ModelUseCase.Name(use_case)  # type: ignore[arg-type]
    return (
        name.removeprefix("MODEL_USE_CASE_")
        .replace("_", " ")
        .title()
        .replace("Ai", "AI")
    )


def tag_proto_to_str(tag: int) -> str:
    """Convert a ModelTag enum value to a human-readable string."""
    name = ModelTag.Name(tag)  # type: ignore[arg-type]
    return name.removeprefix("MODEL_TAG_").replace("_", " ").title().replace("Ai", "AI")


_LICENSE_DISPLAY_NAMES: dict[str, str] = {
    "UNLICENSED": "Unlicensed",
    "COMMERCIAL": "Commercial",
    "AI_HUB_MODELS_LICENSE": "AI Hub Models License",
    "APACHE_2_0": "Apache-2.0",
    "MIT": "MIT",
    "BSD_3_CLAUSE": "BSD-3-Clause",
    "CC_BY_4_0": "CC-BY-4.0",
    "AGPL_3_0": "AGPL-3.0",
    "GPL_3_0": "GPL-3.0",
    "CREATIVEML_OPENRAIL_M": "CreativeML OpenRAIL-M",
    "CC_BY_NON_COMMERCIAL_4_0": "CC-BY-NC-4.0",
    "OTHER_NON_COMMERCIAL": "Other (Non-Commercial)",
    "LLAMA2": "Llama 2",
    "LLAMA3": "Llama 3",
    "TAIDE": "TAIDE",
    "FALCON3": "Falcon 3",
    "GEMMA": "Gemma",
    "LFM1_0": "LFM-1.0",
    "AIMET_MODEL_ZOO": "AIMET Model Zoo",
    "SAM3": "SAM3",
}


def license_proto_to_str(license_val: int) -> str:
    """Convert a ModelLicense enum value to a human-readable string."""
    name = ModelLicense.Name(license_val)  # type: ignore[arg-type]
    key = name.removeprefix("MODEL_LICENSE_")
    return _LICENSE_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
