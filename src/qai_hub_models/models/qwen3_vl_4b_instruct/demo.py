# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models._shared.qwen3_vl.model import DEFAULT_USER_PROMPT, END_TOKENS
from qai_hub_models.models.qwen3_vl_4b_instruct import MODEL_ID
from qai_hub_models.models.qwen3_vl_4b_instruct.model import (
    HF_REPO_NAME,
    HF_REPO_URL,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_VL_4B_PreSplit,
    Qwen3_VL_4B_QuantizablePreSplit,
    Qwen3_VL_4B_VisionEncoder,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec


def qwen3_vl_4b_instruct_chat_demo(
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")

    quantized_cls = (
        Qwen3_VL_4B_QuantizablePreSplit if use_presplit else QuantizedSplitModelWrapper
    )
    fp_cls = Qwen3_VL_4B_PreSplit if use_presplit else FPSplitModelWrapper

    llm_chat_demo(
        model_cls=quantized_cls,
        fp_model_cls=fp_cls,
        qnn_model_cls=LLM_QNN,
        model_id=MODEL_ID,
        end_tokens=END_TOKENS,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_url=HF_REPO_URL,
        default_prompt=DEFAULT_USER_PROMPT,
        test_checkpoint=test_checkpoint,
        vision_encoder_cls=Qwen3_VL_4B_VisionEncoder,
    )


def main() -> None:
    qwen3_vl_4b_instruct_chat_demo()


if __name__ == "__main__":
    main()
