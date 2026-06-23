# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import subprocess
import sys

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

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
                f"genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile /data/local/tmp/QDC_logs/profile{i}.txt 2>>/data/local/tmp/QDC_logs/genie_stderr.log"
            )
        full_genie_command = " && ".join(trial_commands)
        qairt_path = "/data/local/tmp/qairt/<<QAIRT_VERSION>>"
        genie_script = f"""set -e
# We pipe genie output through `tee` (below) so it shows up on adb stdout
# (and thus in the captured proc.stdout) even when a failed QDC job never
# makes the on-device log files available. pipefail keeps the pipeline's
# exit status tied to genie rather than to tee, which always succeeds.
set -o pipefail
# genie-t2t-run fails randomly on QDC devices; give each invocation one retry
# before letting the failure (and set -e) abort the whole job. Redirect stderr
# to a log file: QDC flags jobs Unsuccessful on any stderr output (PR #3641).
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
genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt 2>>/data/local/tmp/QDC_logs/genie_stderr.log | tee /data/local/tmp/QDC_logs/genie.log
{full_genie_command}

PROMPT_DIR=/data/local/tmp/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/QDC_logs/eval_outputs.txt
if [ -d "$PROMPT_DIR" ]; then
    # Switch to power_saver perf_profile: sustained burst thermal-throttles and kills the eval loop on QDC SM8750.
    sed -i 's/"perf_profile": "[^"]*"/"perf_profile": "power_saver"/' htp_backend_ext_config.json
    > "$EVAL_OUTPUT_FILE"
    for prompt_file in $PROMPT_DIR/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\\([0-9]*\\)\\.txt/\\1/')
        echo "===EVAL_IDX_${{idx}}===" | tee -a "$EVAL_OUTPUT_FILE"
        genie_retry genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" 2>&1 | tee -a "$EVAL_OUTPUT_FILE"
        # Short inter-prompt cooldown to keep the HTP from thermal-throttling.
        sleep 3
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

        # Run the shell script on the device. adb shell does not propagate the
        # remote exit code, so on-device failures can't be detected here; the
        # output-existence check below is what catches them.
        proc = subprocess.run(
            ["adb", "shell", "sh", "-c", genie_script],
            capture_output=True,
            text=True,
            check=True,  # only catches adb-side failures, not on-device ones
        )

        # Since adb shell hides the on-device exit code, confirm the script
        # actually produced its outputs. A green pytest with no genie.log was
        # the failure mode on QDC job 613912.
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
                "Expected on-device outputs are missing — the genie script "
                "likely failed on device.\n"
                f"--- ls stdout ---\n{ls.stdout}\n--- ls stderr ---\n{ls.stderr}\n"
                f"--- script stdout ---\n{proc.stdout}\n"
                f"--- script stderr ---\n{proc.stderr}"
            )


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))
