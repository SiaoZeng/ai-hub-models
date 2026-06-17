# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
from pathlib import Path

from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_download,
)

# S3 key / local filename for QAIRT auto SDK
QAIRT_AUTO_SDK_S3_KEY = "qai-hub-models/qairt/2.42.0-auto-20260106/artifact.zip"
QAIRT_AUTO_SDK_FILENAME = "qairt_auto_sdk.zip"


def download_qairt_auto_sdk(local_path: str) -> None:
    """Download the QAIRT SDK for automotive devices from S3."""
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)
    s3_download(bucket, QAIRT_AUTO_SDK_S3_KEY, local_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.environ.get("QAIRT_SDK_PATH")
        or str(Path.home() / QAIRT_AUTO_SDK_FILENAME),
        help="Local path to write the downloaded SDK zip. Defaults to "
        "$QAIRT_SDK_PATH if set, else $HOME/qairt_auto_sdk.zip.",
    )
    args = parser.parse_args()
    download_qairt_auto_sdk(args.output)


if __name__ == "__main__":
    main()
