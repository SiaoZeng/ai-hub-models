# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import types
from collections import OrderedDict
from typing import Any, cast

import torch
import torch.nn.functional as F
from sam3.model.decoder import TransformerDecoder, gen_sineembed_for_position
from sam3.model.model_misc import inverse_sigmoid
from sam3.model.text_encoder_ve import ResidualAttentionBlock
from sam3.model.vitdet import concat_rel_pos
from torch import nn

from qai_hub_models.models._shared.sam.model_patches import Conv2DInplaceLinear


class SAM3Normalize(nn.Module):
    """
    Input normalizer for SAM3.

    Maps images in [0, 1] to [-1, 1] using mean=0.5, std=0.5 on all channels,
    matching the preprocessing used by the pretrained SAM3 weights.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        return (x - mean) / std


def _copy_qkv_weights(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor | None,
    q: Conv2DInplaceLinear,
    k: Conv2DInplaceLinear,
    v: Conv2DInplaceLinear,
    per_proj_dim: int,
) -> None:
    """
    Split and copy QKV weights from a fused projection into separate Q, K, V layers.

    Parameters
    ----------
    src_weight
        Fused QKV weight tensor of shape [3*per_proj_dim, per_proj_dim].
    src_bias
        Fused QKV bias tensor of shape [3*per_proj_dim], or None.
    q
        Target Conv2DInplaceLinear for query projection.
    k
        Target Conv2DInplaceLinear for key projection.
    v
        Target Conv2DInplaceLinear for value projection.
    per_proj_dim
        Embedding dim of each of Q/K/V projections (i.e. the full
        ``embed_dim``, not ``head_dim``).
    """
    for i, proj in enumerate([q, k, v]):
        proj.conv2d.weight.data.copy_(
            src_weight[i * per_proj_dim : (i + 1) * per_proj_dim, :, None, None]
        )
        if src_bias is not None and proj.conv2d.bias is not None:
            proj.conv2d.bias.data.copy_(
                src_bias[i * per_proj_dim : (i + 1) * per_proj_dim]
            )


class SplitHeadResidualAttentionBlock(nn.Module):
    """
    Wrapper for ResidualAttentionBlock with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
        * Linear layers are replaced with Conv2DInplaceLinear for better QNN performance.
    """

    def __init__(self, attention_block: ResidualAttentionBlock) -> None:
        super().__init__()

        # Get the MultiheadAttention module
        mha = attention_block.attn
        d_model = mha.embed_dim
        n_head = mha.num_heads
        head_dim = d_model // n_head

        # Copy basic attributes
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim

        # Get device from the original block
        device = next(attention_block.parameters()).device

        # Extract Q, K, V weights from MultiheadAttention
        # MultiheadAttention stores weights as [3*embed_dim, embed_dim] for in_proj_weight
        # or separate q/k/v weights if _qkv_same_embed_dim is False
        if mha._qkv_same_embed_dim:
            # Weights are combined in in_proj_weight [3*embed_dim, embed_dim]
            in_proj_weight = mha.in_proj_weight
            in_proj_bias = mha.in_proj_bias

            # Split into Q, K, V
            q_weight = in_proj_weight[:d_model, :]
            k_weight = in_proj_weight[d_model : 2 * d_model, :]
            v_weight = in_proj_weight[2 * d_model :, :]

            if in_proj_bias is not None:
                q_bias = in_proj_bias[:d_model]
                k_bias = in_proj_bias[d_model : 2 * d_model]
                v_bias = in_proj_bias[2 * d_model :]
            else:
                q_bias = k_bias = v_bias = None
        else:
            # Separate Q, K, V weights
            q_weight = mha.q_proj_weight
            k_weight = mha.k_proj_weight
            v_weight = mha.v_proj_weight
            q_bias = (
                mha.in_proj_bias[:d_model] if mha.in_proj_bias is not None else None
            )
            k_bias = (
                mha.in_proj_bias[d_model : 2 * d_model]
                if mha.in_proj_bias is not None
                else None
            )
            v_bias = (
                mha.in_proj_bias[2 * d_model :]
                if mha.in_proj_bias is not None
                else None
            )

        # Create Conv2DInplaceLinear layers for Q, K, V
        self.q_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=q_bias is not None, device=device
        )
        self.k_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=k_bias is not None, device=device
        )
        self.v_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=v_bias is not None, device=device
        )

        # Q/K/V are already split above; copy directly without round-tripping
        # through a fused tensor.
        for proj, w, b in (
            (self.q_proj, q_weight, q_bias),
            (self.k_proj, k_weight, k_bias),
            (self.v_proj, v_weight, v_bias),
        ):
            proj.conv2d.weight.data.copy_(w[:, :, None, None])
            if b is not None and proj.conv2d.bias is not None:
                proj.conv2d.bias.data.copy_(b)

        self.out_proj = Conv2DInplaceLinear.from_linear(mha.out_proj)
        self.out_proj.to(device)

        self.ln_1 = attention_block.ln_1
        self.ln_2 = attention_block.ln_2
        self.ls_1 = attention_block.ls_1
        self.ls_2 = attention_block.ls_2

        c_fc_conv = Conv2DInplaceLinear.from_linear(attention_block.mlp[0])
        c_proj_conv = Conv2DInplaceLinear.from_linear(attention_block.mlp[2])
        c_fc_conv.to(device)
        c_proj_conv.to(device)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", c_fc_conv),
                    ("gelu", attention_block.mlp[1]),
                    ("c_proj", c_proj_conv),
                ]
            )
        )

        if hasattr(attention_block, "ln_1_kv"):
            self.ln_1_kv = attention_block.ln_1_kv

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor | None = None,
        v_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute multi-head self or cross attention.

        Parameters
        ----------
        q_x
            Query input tensor of shape (B, N, C).
        k_x
            Key input tensor of shape (B, N_k, C). Defaults to q_x if None.
        v_x
            Value input tensor of shape (B, N_v, C). Defaults to q_x if None.
        attn_mask
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, C).
        """
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        B, N, C = q_x.shape

        # Project Q, K, V using Conv2DInplaceLinear
        # Conv2DInplaceLinear expects (B, N, C) and handles the permutation internally
        q = self.q_proj(q_x)  # (B, N, C)
        k = self.k_proj(k_x)  # (B, N_k, C)
        v = self.v_proj(v_x)  # (B, N_v, C)

        # Reshape for multi-head attention: (B, N, C) -> (B, n_head, N, head_dim)
        q = q.reshape(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, k.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, v.shape[1], self.n_head, self.head_dim).transpose(1, 2)

        # Handle attention mask
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        # Scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Reshape back: (B, n_head, N, head_dim) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection
        return self.out_proj(x)

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor | None = None,
        v_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pre-LN residual attention + MLP block.

        Parameters
        ----------
        q_x
            Query input, shape ``(B, N, C)``.
        k_x
            Key input, shape ``(B, N_k, C)``. Uses ``q_x`` if None.
        v_x
            Value input, shape ``(B, N_v, C)``. Uses ``q_x`` if None.
        attn_mask
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, N, C)``.
        """
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class SplitHeadVitDetAttention(nn.Module):
    """
    Wrapper for VitDet Attention with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
        * Linear layers are replaced with Conv2DInplaceLinear for better QNN performance.
    """

    def __init__(self, attention_block: Any) -> None:
        super().__init__()

        # Copy basic attributes
        self.num_heads = cast(int, attention_block.num_heads)
        self.head_dim = cast(int, attention_block.head_dim)
        self.scale = attention_block.scale
        self.cls_token = attention_block.cls_token

        # Get dimensions
        dim = self.num_heads * self.head_dim

        # Get device from the original block
        device = next(attention_block.parameters()).device

        # Extract fused QKV weight from qkv linear layer (shape [3*dim, dim])
        qkv_weight = attention_block.qkv.weight
        qkv_bias = (
            attention_block.qkv.bias if attention_block.qkv.bias is not None else None
        )

        # Create Conv2DInplaceLinear layers for Q, K, V
        self.q = Conv2DInplaceLinear(
            dim, dim, has_bias=qkv_bias is not None, device=device
        )
        self.k = Conv2DInplaceLinear(
            dim, dim, has_bias=qkv_bias is not None, device=device
        )
        self.v = Conv2DInplaceLinear(
            dim, dim, has_bias=qkv_bias is not None, device=device
        )

        # Copy weights using shared helper
        _copy_qkv_weights(qkv_weight, qkv_bias, self.q, self.k, self.v, dim)

        # Create output projection
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.proj.to(device)

        # Copy rel_pos and rope attributes
        self.use_rel_pos = attention_block.use_rel_pos
        self.input_size = attention_block.input_size
        self.use_rope = attention_block.use_rope

        if self.use_rel_pos:
            self.rel_pos_h = attention_block.rel_pos_h
            self.rel_pos_w = attention_block.rel_pos_w
            if hasattr(attention_block, "relative_coords"):
                self.register_buffer(
                    "relative_coords",
                    cast(torch.Tensor, attention_block.relative_coords),
                )

        if self.use_rope:
            self.freqs_cis = attention_block.freqs_cis
            self.rope_theta = attention_block.rope_theta
            self.rope_pt_size = attention_block.rope_pt_size
            self.rope_interp = attention_block.rope_interp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        VitDet split-head attention with RoPE and optional relative-position bias.

        Parameters
        ----------
        x
            Input tokens, either ``(B, H, W, C)`` (spatial) or ``(B, L, C)``
            (flattened, where ``L`` is a perfect square plus the optional
            cls token).

        Returns
        -------
        torch.Tensor
            Output with the same layout as ``x``: ``(B, H, W, C)`` or ``(B, L, C)``.
        """
        s = 1 if self.cls_token else 0  # used to exclude cls_token
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0  # no cls_token
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = int(math.sqrt(L - s))

        # Project Q, K, V separately and reshape following SplitHeadSAMEncoderAttention pattern
        k = (
            self.k(x)
            .reshape(B, L, self.num_heads * self.head_dim // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(B, L, 2 * self.num_heads, self.head_dim // 2)
            .permute(0, 2, 1, 3)
        )
        v = self.v(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = (
            self.q(x)
            .reshape(B, L, self.num_heads * self.head_dim // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(B, L, 2 * self.num_heads, self.head_dim // 2)
            .permute(0, 2, 1, 3)
        )

        # Apply rope and rel pos embeddings (bound via types.MethodType at model init).
        q, k = self._apply_rope(q, k)  # type: ignore[operator]
        if self.use_rel_pos:
            q, k = concat_rel_pos(
                q.flatten(0, 1),
                k.flatten(0, 1),
                (H, W),
                (H, W),
                self.rel_pos_h,
                self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords
                if hasattr(self, "relative_coords")
                else None,
            )
            # sdpa expects [B, nheads, H*W, C] so we reshape back
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        # Scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v)

        # Reshape back to original format
        if ndim == 4:
            x = x.permute(0, 2, 1, 3).reshape(B, H, W, -1)
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        # Output projection
        return self.proj(x)


class SplitHeadMultiheadAttention(nn.Module):
    """
    Wrapper for nn.MultiheadAttention with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
        * Linear layers are replaced with Conv2DInplaceLinear for better QNN performance.
    """

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()

        d_model = mha.embed_dim
        n_head = mha.num_heads
        head_dim = d_model // n_head

        # Copy basic attributes
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim
        # Check if the original MHA has batch_first attribute (added in PyTorch 1.9+)
        # If not, default to False (sequence-first)
        if hasattr(mha, "batch_first"):
            self.batch_first = mha.batch_first
        else:
            # For older PyTorch versions or if not set, default to False
            self.batch_first = False

        # Get device from the original module
        device = next(mha.parameters()).device

        # Extract Q, K, V weights from MultiheadAttention
        if mha._qkv_same_embed_dim:
            # Weights are combined in in_proj_weight [3*embed_dim, embed_dim]
            in_proj_weight = mha.in_proj_weight
            in_proj_bias = mha.in_proj_bias

            # Split into Q, K, V
            q_weight = in_proj_weight[:d_model, :]
            k_weight = in_proj_weight[d_model : 2 * d_model, :]
            v_weight = in_proj_weight[2 * d_model :, :]

            if in_proj_bias is not None:
                q_bias = in_proj_bias[:d_model]
                k_bias = in_proj_bias[d_model : 2 * d_model]
                v_bias = in_proj_bias[2 * d_model :]
            else:
                q_bias = k_bias = v_bias = None
        else:
            # Separate Q, K, V weights
            q_weight = mha.q_proj_weight
            k_weight = mha.k_proj_weight
            v_weight = mha.v_proj_weight
            q_bias = (
                mha.in_proj_bias[:d_model] if mha.in_proj_bias is not None else None
            )
            k_bias = (
                mha.in_proj_bias[d_model : 2 * d_model]
                if mha.in_proj_bias is not None
                else None
            )
            v_bias = (
                mha.in_proj_bias[2 * d_model :]
                if mha.in_proj_bias is not None
                else None
            )

        # Create Conv2DInplaceLinear layers for Q, K, V
        self.q_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=q_bias is not None, device=device
        )
        self.k_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=k_bias is not None, device=device
        )
        self.v_proj = Conv2DInplaceLinear(
            d_model, d_model, has_bias=v_bias is not None, device=device
        )

        # Q/K/V are already split above; copy directly without round-tripping
        # through a fused tensor.
        for proj, w, b in (
            (self.q_proj, q_weight, q_bias),
            (self.k_proj, k_weight, k_bias),
            (self.v_proj, v_weight, v_bias),
        ):
            proj.conv2d.weight.data.copy_(w[:, :, None, None])
            if b is not None and proj.conv2d.bias is not None:
                proj.conv2d.bias.data.copy_(b)

        self.out_proj = Conv2DInplaceLinear.from_linear(mha.out_proj)
        self.out_proj.to(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """
        Forward pass for split-head multihead attention.

        Parameters
        ----------
        query
            Query tensor of shape (L, B, C) if sequence-first or (B, L, C) if batch-first
        key
            Key tensor of shape (S, B, C) if sequence-first or (B, S, C) if batch-first
        value
            Value tensor of shape (S, B, C) if sequence-first or (B, S, C) if batch-first
        attn_mask
            Attention mask
        key_padding_mask
            Key padding mask of shape (B, S)

        Returns
        -------
        tuple[torch.Tensor, None]
            Tuple of (output, None) to match nn.MultiheadAttention interface
        """
        # Handle both batch-first and sequence-first formats
        if self.batch_first:
            # Input is batch first: (B, L, C)
            B, _L, _C_in = query.shape
            query_bf = query
            key_bf = key
            value_bf = value
        else:
            # Input is sequence first: (L, B, C)
            _L, B, _C_in = query.shape
            # Convert to batch first for Conv2DInplaceLinear: (L, B, C) -> (B, L, C)
            query_bf = query.transpose(0, 1)
            key_bf = key.transpose(0, 1)
            value_bf = value.transpose(0, 1)

        # Project Q, K, V - Conv2DInplaceLinear expects (B, L, C)
        q = self.q_proj(query_bf)  # (B, L, C)
        k = self.k_proj(key_bf)  # (B, S, C)
        v = self.v_proj(value_bf)  # (B, S, C)

        # Get actual sequence lengths after projection
        L_out = q.shape[1]
        S_out = k.shape[1]
        C_out = q.shape[2]

        # Reshape for multi-head attention: (B, L, C) -> (B, n_head, L, head_dim)
        # Note: We use L_out and S_out which are the actual sequence lengths after projection
        q = q.reshape(B, L_out, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S_out, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S_out, self.n_head, self.head_dim).transpose(1, 2)

        # Handle attention mask
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(query.dtype)

        # Handle key padding mask - convert to attention mask format
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) where True means ignore/mask out
            # For F.scaled_dot_product_attention with boolean mask:
            #   - True means KEEP the position
            #   - False means MASK OUT the position
            # So we need to INVERT the key_padding_mask!

            # Expand to (B, 1, 1, S) for broadcasting across heads and queries
            # Invert: True (ignore) -> False (mask out), False (keep) -> True (keep)
            key_padding_mask_expanded = (~key_padding_mask).unsqueeze(1).unsqueeze(2)

            if attn_mask is None:
                attn_mask = key_padding_mask_expanded
            # Combine masks: both must be True to keep
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask & key_padding_mask_expanded
            else:
                # If attn_mask is float, convert key_padding_mask to float
                key_padding_mask_float = torch.zeros_like(
                    key_padding_mask_expanded, dtype=query.dtype
                )
                key_padding_mask_float.masked_fill_(
                    ~key_padding_mask_expanded, float("-inf")
                )
                attn_mask = attn_mask + key_padding_mask_float

        # Scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Reshape back: (B, n_head, L, head_dim) -> (B, L, C)
        x = x.transpose(1, 2).reshape(B, L_out, C_out)

        # Output projection - Conv2DInplaceLinear expects (B, L, C)
        x = self.out_proj(x)

        # Convert back to original format
        if not self.batch_first:
            # Convert back to sequence first: (B, L, C) -> (L, B, C)
            x = x.transpose(0, 1)

        # Return tuple to match nn.MultiheadAttention interface (output, attention_weights)
        # We don't return attention weights, so return None
        return x, None


def patch_decoder_last_layer_only(decoder: TransformerDecoder) -> None:
    """
    Replace ``TransformerDecoder.forward`` with a variant that only
    materializes the final layer's hidden state, reference boxes, and
    presence logit. SAM3's heads only consume ``[-1]``, so upstream's
    per-layer stacking is wasted work. Numerically equivalent; output
    keeps the 4-tuple shape with a singleton ``num_layers=1`` axis.
    """

    def _forward_last_only(
        self: TransformerDecoder,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        reference_boxes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
        memory_text: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        apply_dac: bool | None = None,
        is_instance_prompt: bool = False,
        decoder_extra_kwargs: dict | None = None,
        obj_roi_memory_feat: torch.Tensor | None = None,
        obj_roi_memory_mask: torch.Tensor | None = None,
        box_head_trk: torch.nn.Module | None = None,
    ) -> tuple:
        if memory_mask is not None:
            assert self.boxRPB == "none"

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            tgt = tgt.repeat(2, 1, 1)
            if reference_boxes is not None:
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs = tgt.shape[1]
        presence_feats = None

        if not self.box_refine:
            raise NotImplementedError(
                "box_refine=False not supported in last-only path"
            )
        if reference_boxes is None:
            reference_boxes = self.reference_points.weight.unsqueeze(1)
            reference_boxes = (
                reference_boxes.repeat(2, bs, 1)
                if apply_dac
                else reference_boxes.repeat(1, bs, 1)
            )
            reference_boxes = reference_boxes.sigmoid()

        # SAM3Head always passes these in; narrow for mypy so the
        # arithmetic/indexing below doesn't trip on Optional types.
        assert valid_ratios is not None
        assert spatial_shapes is not None
        assert decoder_extra_kwargs is None or isinstance(decoder_extra_kwargs, dict)

        output = tgt
        presence_out = None
        if self.presence_token is not None and not is_instance_prompt:
            presence_out = self.presence_token.weight[None].expand(1, bs, -1)

        box_head = self.bbox_embed
        if is_instance_prompt and self.instance_bbox_embed is not None:
            box_head = self.instance_bbox_embed
        out_norm = self.norm
        if is_instance_prompt and self.instance_norm is not None:
            out_norm = self.instance_norm

        # Upstream's intermediate_ref_boxes list starts with the initial
        # reference_boxes (before any layer runs) and appends new refs
        # for layers 0..N-2 — i.e., the refs going INTO each layer,
        # excluding the last layer's own refinement output. So the
        # "last ref" the upstream forward emits is the one going into
        # the final layer, which is produced by layer N-2's refinement.
        last_reference_boxes = (
            reference_boxes  # will be overwritten with input to final layer
        )
        for layer_idx, layer in enumerate(self.layers):
            # Capture the refs going INTO this layer. After the last
            # layer runs, last_reference_boxes holds the refs that went
            # into the final layer — matching upstream's ``[-1]``.
            if layer_idx == len(self.layers) - 1:
                last_reference_boxes = reference_boxes
            reference_points_input = (
                reference_boxes[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )
            query_pos = self.ref_point_head(query_sine_embed)

            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes.shape[0] == 1
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1)

            output, presence_out = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                presence_token=presence_out,
                **(decoder_extra_kwargs or {}),
            )

            # iter update: always produce new refs (layer N+1 needs them).
            reference_before_sigmoid = inverse_sigmoid(reference_boxes)
            if box_head_trk is None:
                if not self.use_normed_output_consistently:
                    delta_unsig = box_head(output)
                else:
                    delta_unsig = box_head(out_norm(output))
            else:
                assert decoder_extra_kwargs is not None
                Q_det = decoder_extra_kwargs["Q_det"]
                delta_unsig_det = self.bbox_embed(output[:Q_det])
                delta_unsig_trk = box_head_trk(output[Q_det:])
                delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)
            new_reference_points = (delta_unsig + reference_before_sigmoid).sigmoid()
            reference_boxes = new_reference_points.detach()

        # Final-layer outputs only.
        hs_last = out_norm(output)

        if self.presence_token is not None and not is_instance_prompt:
            assert presence_out is not None
            presence_last = self.presence_token_head(
                self.presence_token_out_norm(presence_out)
            ).squeeze(-1)
            if self.clamp_presence_logits:
                presence_last = presence_last.clamp(
                    min=-self.clamp_presence_logit_max_val,
                    max=self.clamp_presence_logit_max_val,
                )
            presence_feats = presence_out.clone()
            presence_stacked = presence_last.unsqueeze(0)
        else:
            presence_stacked = None

        return (
            hs_last.unsqueeze(0),
            last_reference_boxes.unsqueeze(0),
            presence_stacked,
            presence_feats,
        )

    decoder.forward = types.MethodType(_forward_last_only, decoder)


def patch_decoder_rpb_device(decoder: TransformerDecoder) -> None:
    """
    Monkey-patch TransformerDecoder._get_rpb_matrix to fix a device mismatch.

    The upstream decoder pre-caches coords_h / coords_w on CUDA (when a GPU is
    available at construction time) but inference is run on CPU.  This patch
    moves the cached coordinate tensors to the same device as reference_boxes
    before they are used, without touching the library source.

    Parameters
    ----------
    decoder
        The TransformerDecoder instance to patch in-place.
    """

    def _get_rpb_matrix_device_safe(
        self: TransformerDecoder,
        reference_boxes: torch.Tensor,
        feat_size: tuple[int, int],
    ) -> torch.Tensor:
        # Coerce feat_size to Python ints. Upstream may pass tensor scalars
        # (from spatial_shapes indexing) which break the cache-hit comparison
        # during JIT tracing verification.
        h, w = feat_size
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
        feat_size = (h, w)

        # Ensure the compilable coord cache lives on the same device as the
        # input tensors (upstream may have created it on CUDA at init time).
        if self.compilable_cord_cache is not None:
            coords_h, coords_w = self.compilable_cord_cache
            target_device = reference_boxes.device
            if coords_h.device != target_device or coords_w.device != target_device:
                self.compilable_cord_cache = (
                    coords_h.to(target_device),
                    coords_w.to(target_device),
                )
        # Also fix any entries already stored in the dict-based cache.
        if hasattr(self, "coord_cache"):
            for key, (ch, cw) in list(self.coord_cache.items()):
                target_device = reference_boxes.device
                if ch.device != target_device or cw.device != target_device:
                    self.coord_cache[key] = (
                        ch.to(target_device),
                        cw.to(target_device),
                    )
        # Delegate to the original implementation.
        return TransformerDecoder._get_rpb_matrix(self, reference_boxes, feat_size)

    decoder._get_rpb_matrix = types.MethodType(_get_rpb_matrix_device_safe, decoder)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position encodings to query and key tensors.

    Decomposes complex rotation into real/imaginary components to avoid
    complex number operations unsupported by QNN.

    Parameters
    ----------
    xq
        Query tensor.
    xk
        Key tensor.
    freqs_cos
        Cosine component of rotary frequencies.
    freqs_sin
        Sine component of rotary frequencies.
    repeat_freqs_k
        If True, repeat frequencies to match key sequence length.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    xq_parts = torch.split(xq, xq.shape[1] // 2, dim=1)
    xq_real = xq_parts[0]
    xq_imag = xq_parts[1]

    if xk.shape[-2] != 0:
        xk_parts = torch.split(xk, xk.shape[1] // 2, dim=1)
        xk_real = xk_parts[0]
        xk_imag = xk_parts[1]
    else:
        xk_real = None
        xk_imag = None

    ndim = xq_real.ndim
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(xq_real.shape)]
    freqs_cos = freqs_cos.view(*shape)
    freqs_sin = freqs_sin.view(*shape)

    # Apply rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    xq_out_real = xq_real * freqs_cos - xq_imag * freqs_sin
    xq_out_imag = xq_real * freqs_sin + xq_imag * freqs_cos
    xq_out = torch.cat([xq_out_real, xq_out_imag], dim=-1)

    if xk_real is None:
        return xq_out, xk

    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_cos = freqs_cos.repeat(*([1] * (freqs_cos.ndim - 2)), r, 1)
        freqs_sin = freqs_sin.repeat(*([1] * (freqs_sin.ndim - 2)), r, 1)

    # Apply rotation to keys
    assert xk_real is not None and xk_imag is not None
    xk_out_real = xk_real * freqs_cos - xk_imag * freqs_sin
    xk_out_imag = xk_real * freqs_sin + xk_imag * freqs_cos
    xk_out = torch.cat([xk_out_real, xk_out_imag], dim=-1)

    return xq_out, xk_out


def apply_rope(
    self: Any, q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings using pre-split cos/sin buffers.

    Bound as a method on VitDet attention blocks via types.MethodType
    to replace the original complex-number-based implementation.

    Parameters
    ----------
    self
        The attention block instance (bound at runtime).
    q
        Query tensor.
    k
        Key tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    if not self.use_rope:
        return q, k

    assert self.freqs_cos is not None and self.freqs_sin is not None
    return apply_rotary_enc(q, k, self.freqs_cos, self.freqs_sin)
