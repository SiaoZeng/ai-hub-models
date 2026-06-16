# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""If set, forces the CLI to act as if it was running as this AI Hub Models version."""

FORCE_VERSION_ENVVAR = "QAIHM_CLI_FORCE_VERSION"

"""
If set, FOR ANY RELEASE, forces the CLI to load manifest files from this path.
In practice, this means all releases will display the same information.
"""
FORCE_MANIFEST_ROOT_ENVVAR = "QAIHM_CLI_MANIFEST_ROOT"

"""
If set to "1", enables verbose exceptions. By default ("0"), the CLI will swallow the traceback.
"""
VERBOSE_EXCEPTIONS_ENVVAR = "QAIHM_CLI_VERBOSE_EXCEPTIONS"
