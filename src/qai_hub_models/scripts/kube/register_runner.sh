#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

RESPONSE=$(curl -s -X POST \
  -H "Authorization: token $PAT_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "$GITHUB_API_URL/repos/$GITHUB_REPO/actions/runners/registration-token")
TOKEN=$(echo "$RESPONSE" | jq -r '.token')
if [ "$TOKEN" = "null" ] || [ -z "$TOKEN" ]; then
  echo "Failed to generate runner registration token"
  echo "$RESPONSE"
  exit 1
fi
echo "::add-mask::$TOKEN"

kubectl create secret generic "runner-token-$RUNNER_ID" \
  -n "$ARGO_NS" \
  --from-literal=token="$TOKEN"

echo "Waiting for runner '$RUNNER_ID' to register with GitHub..."

NUM_TRIES=20
for i in $(seq 1 $NUM_TRIES); do
  WF_STATUS=$(argo get "$RUNNER_ID" -n "$ARGO_NS" -o json 2>/dev/null | jq -r '.status.phase // empty')
  if [ "$WF_STATUS" = "Failed" ] || [ "$WF_STATUS" = "Error" ]; then
    echo "Argo workflow entered terminal state: $WF_STATUS"
    argo get "$RUNNER_ID" -n "$ARGO_NS" || true
    exit 1
  fi

  STATUS=$(curl -s \
    -H "Authorization: token $PAT_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "$GITHUB_API_URL/repos/$GITHUB_REPO/actions/runners?per_page=100" \
    | jq -r --arg name "$RUNNER_ID" '.runners[] | select(.name==$name) | .status')

  if [ "$STATUS" = "online" ]; then
    echo "Runner '$RUNNER_ID' is online!"
    kubectl delete secret "runner-token-$RUNNER_ID" \
      -n "$ARGO_NS" --ignore-not-found || true
    exit 0
  fi
  echo "Attempt $i/$NUM_TRIES - runner not yet online (status: ${STATUS:-not found})"
  sleep 30
done

echo "Runner did not come online within 10 minutes after pod started"
exit 1
