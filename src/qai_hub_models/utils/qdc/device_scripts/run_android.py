# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import re
import subprocess
import sys

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

# adb shell does not propagate the remote command's exit status on most
# Android versions, so we tee an explicit marker and parse it out.
_EXIT_MARKER = "__QDC_EXIT__"

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))


class TestGenie:
    @pytest.fixture
    def driver(self) -> webdriver.Remote:
        return webdriver.Remote(
            command_executor="http://127.0.0.1:4723/wd/hub", options=options
        )

    def test_genie(self, driver: webdriver.Remote) -> None:
        # download qairt sdk via curl on device
        # script to set environment variables
        # run genie-t2t-run on the device
        num_trials = int("<<NUM_TRIALS>>")
        trial_commands = []
        for i in range(num_trials):
            trial_commands.append(
                f'sed -i \'s/"seed": [0-9]*/"seed": {i}/\' genie_config.json'
            )
            trial_commands.append(
                f"genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile /data/local/tmp/QDC_logs/profile{i}.txt"
            )
        full_genie_command = " && ".join(trial_commands)
        qairt_path = "/data/local/tmp/qairt/<<QAIRT_VERSION>>"
        # The EXIT trap runs on every exit path (clean, set -e abort, signal),
        # so $? captures the real rc and the host can parse it from stdout.
        # Needed because adb shell always returns 0, hiding on-device failures.
        genie_script = f"""trap 'rc=$?; echo {_EXIT_MARKER}$rc' EXIT
set -e
# genie-t2t-run fails randomly on QDC devices; give each invocation one retry
# before letting the failure (and set -e) abort the whole job.
genie_retry() {{
    "$@" || {{
        echo "genie_retry: command failed, retrying once: $*" >&2
        "$@"
    }}
}}
cd /data/local/tmp/genie_bundle
echo "=== Pre-download connectivity check ==="
echo "Pinging google.com before QAIRT SDK download..."
ping -c 1 google.com && echo "Pre-download ping: SUCCESS" || echo "Pre-download ping: FAILED"
curl -L -J --fail --max-time 300 --retry 3 --retry-delay 5 --output /data/local/tmp/qairt.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<<QAIRT_VERSION>>/v<<QAIRT_VERSION>>.zip
echo "=== Post-download connectivity check ==="
echo "Pinging google.com after QAIRT SDK download..."
ping -c 1 google.com && echo "Post-download ping: SUCCESS" || echo "Post-download ping: FAILED"
unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp || {{
    echo "unzip failed, retrying once" >&2
    rm -rf /data/local/tmp/qairt
    unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp
}}
export QAIRT_HOME={qairt_path}
export PATH={qairt_path}/bin/aarch64-android:${{PATH}}
export LD_LIBRARY_PATH={qairt_path}/lib/aarch64-android
export ADSP_LIBRARY_PATH={qairt_path}/lib/hexagon-<<HEXAGON_VERSION>>/unsigned

mkdir -p /data/local/tmp/QDC_logs
genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt > /data/local/tmp/QDC_logs/genie.log
{full_genie_command}

PROMPT_DIR=/data/local/tmp/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/QDC_logs/eval_outputs.txt
if [ -d "$PROMPT_DIR" ]; then
    > "$EVAL_OUTPUT_FILE"
    for prompt_file in $PROMPT_DIR/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\\([0-9]*\\)\\.txt/\\1/')
        echo "===EVAL_IDX_${{idx}}===" >> "$EVAL_OUTPUT_FILE"
        genie_retry genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" >> "$EVAL_OUTPUT_FILE" 2>&1
    done
fi
"""
        # Push the genie_bundle directory to the device
        subprocess.run(
            ["adb", "push", "/qdc/appium/genie_bundle/", "/data/local/tmp"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Preflight: bail fast if the device can't reach the QAIRT download
        # host. We've seen QDC SM8750 QRD boot with wifi degraded (logcat
        # shows WifiHAL fatal_event + ENETDOWN), in which case the curl below
        # would hang for ~20 minutes and the test would silently "pass".
        preflight = subprocess.run(
            [
                "adb",
                "shell",
                "curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "
                "https://softwarecenter.qualcomm.com/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        http_code = preflight.stdout.strip()
        if preflight.returncode != 0 or not http_code.startswith(("2", "3")):
            pytest.fail(
                "Device cannot reach softwarecenter.qualcomm.com "
                f"(rc={preflight.returncode}, http_code={http_code!r}, "
                f"stderr={preflight.stderr!r}). Likely QDC device-side wifi "
                "failure — file a QDC infra ticket and re-run."
            )

        # Run the shell script on the device. adb shell does not propagate
        # the remote exit code; the script's own EXIT trap echoes a marker
        # we can parse out of stdout instead.
        proc = subprocess.run(
            ["adb", "shell", "sh", "-c", genie_script],
            capture_output=True,
            text=True,
            check=True,  # only catches adb-side failures, not on-device ones
        )
        match = re.search(rf"{_EXIT_MARKER}(\d+)\s*$", proc.stdout)
        if match is None:
            pytest.fail(
                "adb shell did not report an exit code.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        rc = int(match.group(1))
        if rc != 0:
            pytest.fail(
                f"On-device genie script exited with rc={rc}.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

        # Confirm the on-device script actually produced its outputs. A green
        # pytest with no genie.log was the failure mode on QDC job 613912.
        expected = ["/data/local/tmp/QDC_logs/genie.log"] + [
            f"/data/local/tmp/QDC_logs/profile{i}.txt" for i in range(num_trials)
        ]
        ls = subprocess.run(
            ["adb", "shell", "ls", "-l", *expected],
            check=False,
            capture_output=True,
            text=True,
        )
        if ls.returncode != 0:
            pytest.fail(
                "Expected on-device outputs are missing:\n"
                f"--- stdout ---\n{ls.stdout}\n--- stderr ---\n{ls.stderr}"
            )


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))
