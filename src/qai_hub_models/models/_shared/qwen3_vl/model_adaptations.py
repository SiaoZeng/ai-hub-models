# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
)

from qai_hub_models.models._shared.llm.model_adaptations import ConvInplaceLinear
from qai_hub_models.models._shared.qwen3.model_adaptations import SHAQwen3Attention


class SHAQwen3VLTextAttention(Qwen3VLTextAttention):
    """Split-Head Attention for Qwen3-VL text model.

    Reuses prepare_conv, prepare_sha, and forward_sha from SHAQwen3Attention.
    Must inherit from Qwen3VLTextAttention so the monkey-patch works.
    """

    prepare_conv = SHAQwen3Attention.prepare_conv
    prepare_sha = SHAQwen3Attention.prepare_sha
    forward_sha = SHAQwen3Attention.forward_sha


class QCQwen3VLTextMLP(Qwen3VLTextMLP):
    """Qwen3-VL text MLP with Conv2d adaptation for HTP backend."""

    def prepare_conv(self) -> None:
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[has-type, arg-type, unused-ignore]


class QCQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def prepare_conv(self) -> None:
        self.lm_head = ConvInplaceLinear(self.lm_head)  # type: ignore[has-type, arg-type, unused-ignore]
