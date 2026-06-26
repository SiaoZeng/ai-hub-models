# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.vlm.evaluate import vlm_evaluate
from qai_hub_models.models.qwen3_vl_4b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    SUPPORTED_PRECISIONS,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_VL_4B_PreSplit,
    Qwen3_VL_4B_QuantizablePreSplit,
    Qwen3_VL_4B_VisionEncoder,
)

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")
    vlm_evaluate(
        quantized_model_cls=Qwen3_VL_4B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Qwen3_VL_4B_PreSplit if use_presplit else FPSplitModelWrapper,
        supported_precisions=SUPPORTED_PRECISIONS,
        vision_encoder_cls=Qwen3_VL_4B_VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )
