#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

NAMESPACE="${NAMESPACE:-aihub-ci}"

TEMPLATE="github-actions-runner"
WF_NAME=""
WF_LABELS=""
WAIT_STEP=""
CPU_REQUEST=""
GPU_REQUEST=""
MEMORY_REQUEST=""
DOCKER_IMAGE=""
EXTRA_PARAMS=()

usage() {
  echo "Usage: $0 [options]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --template <name>       Argo WorkflowTemplate (default: github-actions-runner)" >&2
  echo "  --name <wf-name>        Explicit Argo workflow name" >&2
  echo "  --labels <string>       Argo workflow labels (key=value,...)" >&2
  echo "  --wait-step <name>      Poll for this specific step to be Running" >&2
  echo "  --docker-image <image>  Docker image to use for the pod" >&2
  echo "  -p <key=value>          Extra Argo parameter (can be repeated)" >&2
  echo "  -c <cpu>                CPU request" >&2
  echo "  -g <gpu>                GPU request" >&2
  echo "  -m <memory>             Memory request" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --template)       TEMPLATE="$2"; shift 2 ;;
    --name)           WF_NAME="$2"; shift 2 ;;
    --labels)         WF_LABELS="$2"; shift 2 ;;
    --wait-step)      WAIT_STEP="$2"; shift 2 ;;
    --docker-image)   DOCKER_IMAGE="$2"; shift 2 ;;
    -p)               EXTRA_PARAMS+=("$2"); shift 2 ;;
    -c)               CPU_REQUEST="$2"; shift 2 ;;
    -g)               GPU_REQUEST="$2"; shift 2 ;;
    -m)               MEMORY_REQUEST="$2"; shift 2 ;;
    -h|--help)        usage ;;
    *)                echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

ARGO_ARGS=()
ARGO_ARGS+=(--from "workflowtemplate/$TEMPLATE")
ARGO_ARGS+=(-n "$NAMESPACE")

if [ -n "$WF_NAME" ]; then
  ARGO_ARGS+=(--name "$WF_NAME")
fi

if [ -n "$WF_LABELS" ]; then
  ARGO_ARGS+=(--labels "$WF_LABELS")
fi

[ -n "$CPU_REQUEST" ]    && ARGO_ARGS+=(-p "cpu-request=$CPU_REQUEST")
[ -n "$GPU_REQUEST" ]    && ARGO_ARGS+=(-p "gpu-request=$GPU_REQUEST")
[ -n "$MEMORY_REQUEST" ] && ARGO_ARGS+=(-p "memory-request=$MEMORY_REQUEST")
[ -n "$DOCKER_IMAGE" ]   && ARGO_ARGS+=(-p "docker-image=$DOCKER_IMAGE")

for param in "${EXTRA_PARAMS[@]}"; do
  ARGO_ARGS+=(-p "$param")
done

ARGO_ARGS+=(-o name)

echo "Submitting workflow (template=$TEMPLATE)..." >&2
SUBMITTED_WF=$(argo submit "${ARGO_ARGS[@]}")
echo "Workflow submitted: $SUBMITTED_WF" >&2

if [ -z "$WAIT_STEP" ]; then
  echo "$SUBMITTED_WF"
  exit 0
fi

echo "Waiting for step '$WAIT_STEP' to be Running..." >&2

MAX_ATTEMPTS=40
for i in $(seq 1 $MAX_ATTEMPTS); do
  WF_JSON=$(argo get "$SUBMITTED_WF" -n "$NAMESPACE" -o json 2>/dev/null || echo '{}')
  WF_STATUS=$(echo "$WF_JSON" | jq -r '.status.phase // empty')

  if [ "$WF_STATUS" = "Failed" ] || [ "$WF_STATUS" = "Error" ]; then
    echo "ERROR: Workflow $SUBMITTED_WF $WF_STATUS" >&2
    argo get "$SUBMITTED_WF" -n "$NAMESPACE" >&2
    exit 1
  fi

  STEP_PHASE=$(echo "$WF_JSON" | jq -r --arg step "$WAIT_STEP" '
    [.status.nodes // {} | to_entries[] | select(.value.displayName == $step)]
    | first | .value.phase // "NotStarted"')

  case "$STEP_PHASE" in
    Running)
      echo "Step '$WAIT_STEP' is running!" >&2
      echo "$SUBMITTED_WF"
      exit 0
      ;;
    Failed|Error)
      echo "Step '$WAIT_STEP' failed with phase: $STEP_PHASE" >&2
      argo get "$SUBMITTED_WF" -n "$NAMESPACE" >&2
      exit 1
      ;;
    *)
      echo "Attempt $i/$MAX_ATTEMPTS - step phase: ${STEP_PHASE} (workflow: ${WF_STATUS}) -- waiting..." >&2
      sleep 30
      ;;
  esac
done

echo "Step '$WAIT_STEP' did not reach Running within 20 minutes" >&2
exit 1
