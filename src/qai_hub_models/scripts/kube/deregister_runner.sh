#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

IFS=',' read -ra IDS <<< "$RUNNER_IDS"
for id in "${IDS[@]}"; do
  id=$(echo "$id" | xargs)
  if [ -z "$id" ]; then
    continue
  fi

  # Clean up the runner-token secret in case it's still around
  kubectl delete secret "runner-token-$id" -n "$ARGO_NS" --ignore-not-found || true

  RUNNER_ID=$(curl -s \
    -H "Authorization: token $PAT_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "$GITHUB_API_URL/repos/$GITHUB_REPO/actions/runners?per_page=100" \
    | jq -r --arg name "$id" '.runners[] | select(.name==$name) | .id')

  if [ -n "$RUNNER_ID" ] && [ "$RUNNER_ID" != "null" ]; then
    echo "De-registering runner '$id' (ID: $RUNNER_ID)..."
    curl -s -X DELETE \
      -H "Authorization: token $PAT_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      "$GITHUB_API_URL/repos/$GITHUB_REPO/actions/runners/$RUNNER_ID" || true
  else
    echo "Runner '$id' not found on GitHub (already de-registered)"
  fi
done
echo "Cleanup complete"
