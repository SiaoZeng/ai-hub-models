#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

set -e

# Redirect all output to log file for QDC collection
mkdir -p /data/local/tmp/QDC_logs
exec > /data/local/tmp/QDC_logs/script.log 2>&1

mount -o rw,remount /

cd /data/local/tmp/TestContent/genie_bundle

# Download QAIRT SDK
curl -L -J --output /data/local/tmp/qairt.zip \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip

unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp || {
    echo "unzip failed, retrying once" >&2
    rm -rf /data/local/tmp/qairt
    unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp
}

export QAIRT_HOME=/data/local/tmp/qairt/{QAIRT_VERSION}
export PATH=$QAIRT_HOME/bin/aarch64-oe-linux-gcc11.2:$PATH
export LD_LIBRARY_PATH=$QAIRT_HOME/lib/aarch64-oe-linux-gcc11.2
export ADSP_LIBRARY_PATH=$QAIRT_HOME/lib/hexagon-{HEXAGON_VERSION}/unsigned

# genie-t2t-run fails randomly on QDC devices; give each invocation one retry
# before letting the failure propagate.
genie_retry() {
    "$@" || {
        echo "genie_retry: command failed, retrying once: $*" >&2
        "$@"
    }
}

# Run genie (capture initial output, including stderr)
genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt > /data/local/tmp/QDC_logs/genie.log 2>&1

# Run profiling iterations
for i in $(seq 1 {NUM_TRIALS}); do
    sed -i "s/\"seed\": [0-9]*/\"seed\": $i/" genie_config.json
    genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt \
      --profile /data/local/tmp/QDC_logs/profile${i}.txt
done

# Run evaluation over all prompt files
PROMPT_DIR=/data/local/tmp/TestContent/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/QDC_logs/eval_outputs.txt

if [ -d "$PROMPT_DIR" ]; then
    true > "$EVAL_OUTPUT_FILE"
    for prompt_file in "$PROMPT_DIR"/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\([0-9]*\)\.txt/\1/')
        echo "===EVAL_IDX_${idx}===" >> "$EVAL_OUTPUT_FILE"
        genie_retry genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" >> "$EVAL_OUTPUT_FILE" 2>&1
    done
fi

mount -o rw,remount /
