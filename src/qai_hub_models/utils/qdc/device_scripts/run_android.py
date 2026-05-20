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
        genie_command = [
            f"genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile /data/local/tmp/QDC_logs/profile{i}.txt"
            for i in range(10)
        ]
        full_genie_command = " && ".join(genie_command)
        qairt_path = "/data/local/tmp/qairt/<<QAIRT_VERSION>>"
        genie_script = f"""set -e
cd /data/local/tmp/genie_bundle
curl -L -J --output /data/local/tmp/qairt.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<<QAIRT_VERSION>>/v<<QAIRT_VERSION>>.zip
unzip /data/local/tmp/qairt.zip -d /data/local/tmp
export QAIRT_HOME={qairt_path}
export PATH={qairt_path}/bin/aarch64-android:${{PATH}}
export LD_LIBRARY_PATH={qairt_path}/lib/aarch64-android
export ADSP_LIBRARY_PATH={qairt_path}/lib/hexagon-<<HEXAGON_VERSION>>/unsigned

mkdir -p /data/local/tmp/QDC_logs
genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt > /data/local/tmp/QDC_logs/genie.log
{full_genie_command}

PROMPT_DIR=/data/local/tmp/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/QDC_logs/eval_outputs.txt
if [ -d "$PROMPT_DIR" ]; then
    > "$EVAL_OUTPUT_FILE"
    for prompt_file in $PROMPT_DIR/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\\([0-9]*\\)\\.txt/\\1/')
        echo "===EVAL_IDX_${{idx}}===" >> "$EVAL_OUTPUT_FILE"
        genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" >> "$EVAL_OUTPUT_FILE" 2>&1
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

        # Run the shell script on the device
        subprocess.run(
            [
                "adb",
                "shell",
                "sh",
                "-c",
                genie_script,
            ],
            capture_output=True,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))
