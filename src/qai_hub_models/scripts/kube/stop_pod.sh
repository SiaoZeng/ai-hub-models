#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

NAMESPACE="${NAMESPACE:-aihub-ci}"

if [ "$1" = "-l" ]; then
  argo list -n "$NAMESPACE"
  exit 0
fi

if [ "$1" = "-a" ]; then
  argo terminate -n "$NAMESPACE" --all
  exit 0
fi

if [ -z "$1" ]; then
  echo "Usage: $0 <workflow-name>" >&2
  exit 1
fi

echo "Terminating workflow '$1' in namespace '$NAMESPACE'..."
argo terminate -n "$NAMESPACE" "$1" || true
echo "Done."
