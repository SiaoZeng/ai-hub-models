# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.models._shared.llm.demo import llm_chat_demo


def vlm_chat_demo(
    *,
    model_cls: type,
    fp_model_cls: type,
    vision_encoder_cls: type,
    end_tokens: set[str],
    default_prompt: str,
    **kwargs: Any,
) -> None:
    """Run the VLM chat demo via the shared ``llm_chat_demo``.

    The per-model shim resolves ``--use-presplit`` (PreSplit vs Split-wrapper)
    and passes the chosen ``model_cls`` / ``fp_model_cls`` here.
    """
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        vision_encoder_cls=vision_encoder_cls,
        end_tokens=end_tokens,
        default_prompt=default_prompt,
        **kwargs,
    )
