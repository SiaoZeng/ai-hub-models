# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
)

from qai_hub_models.models._shared.qwen2_vl.vision_encoder_adaptations import (
    Qwen2_5_VLVisionAttentionAdaptation,
)


class Qwen3VLVisionAttentionAdaptation(Qwen2_5_VLVisionAttentionAdaptation):
    """
    Adapted vision attention with split Q/K/V Conv2d projections.

    Replaces the fused QKV linear layer with separate Q, K, V Conv2d(1x1)
    projections and uses explicit attention masks instead of dynamic cu_seqlens.
    """


class Qwen3VLVisionBlockAdaptation(nn.Module):
    """Adapted vision block that accepts pre-computed attention masks and RoPE."""

    def __init__(self, block: Qwen3VLVisionBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.attn = block.attn
        self.mlp = block.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


# Utility functions for replacing modules


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    pre, _, post = attr.rpartition(".")
    setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def replace_visual_attention_with_adaptation(
    model: nn.Module,
) -> nn.Module:
    """Replace all VisionBlock and VisionAttention modules with adapted versions."""
    # Replace blocks first
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLVisionBlock):
            _rsetattr(model, name, Qwen3VLVisionBlockAdaptation(module))

    # Then replace attention modules
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLVisionAttention):
            _rsetattr(model, name, Qwen3VLVisionAttentionAdaptation(module))  # type: ignore[arg-type, unused-ignore]

    return model
