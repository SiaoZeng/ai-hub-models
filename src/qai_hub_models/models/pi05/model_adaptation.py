# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers.models.gemma.modeling_gemma import GemmaMLP


def apply_rope_direct(
    x: torch.Tensor,
    rope_emb_sin: torch.Tensor,
    rope_emb_cos: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE using precomputed sin/cos. Simplified version for direct application.

    Parameters
    ----------
    x
        [B, L, H, D] (D even)
    rope_emb_sin
        [B, L, 1, D/2] (float32)
    rope_emb_cos
        [B, L, 1, D/2] (float32)

    Returns
    -------
    rotated_x : torch.Tensor
        [B, L, H, D] (same dtype as input)
    """
    d_half = x.shape[-1] // 2
    x1, x2 = x.split(d_half, dim=-1)
    part1 = x1 * rope_emb_cos - x2 * rope_emb_sin
    part2 = x2 * rope_emb_cos + x1 * rope_emb_sin
    return torch.cat([part1, part2], dim=-1)


class SHAGemmaExpertAttention(torch.nn.Module):
    """
    Split-head attention replacement for GemmaAttention in expert layers.

    Replaces multi-head Q/K/V projections with per-head nn.Linear modules
    and computes attention one head at a time, reducing peak memory for
    on-device deployment.
    """

    def __init__(
        self,
        attn_module: torch.nn.Module,
        num_att_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_att_heads = num_att_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_att_heads // num_kv_heads

        q_proj = attn_module.q_proj
        k_proj = attn_module.k_proj
        v_proj = attn_module.v_proj
        assert isinstance(q_proj, nn.Linear)
        assert isinstance(k_proj, nn.Linear)
        assert isinstance(v_proj, nn.Linear)
        # Derive hidden_size from actual weights (expert hidden dim may
        # differ from num_att_heads * head_dim).
        hidden_size = q_proj.in_features
        device = q_proj.weight.device
        dtype = q_proj.weight.dtype

        # Per-head Q projections
        self.q_proj_sha = nn.ModuleList(
            [
                nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_att_heads)
            ]
        )
        # Per-KV-head K/V projections
        self.k_proj_sha = nn.ModuleList(
            [
                nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_kv_heads)
            ]
        )
        self.v_proj_sha = nn.ModuleList(
            [
                nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_kv_heads)
            ]
        )

        # Keep o_proj as single projection
        o_proj = attn_module.o_proj
        assert isinstance(o_proj, nn.Linear)
        self.o_proj: nn.Linear = o_proj

        # Copy weight slices from original projections
        with torch.no_grad():
            for i in range(num_att_heads):
                q_sha = self.q_proj_sha[i]
                assert isinstance(q_sha, nn.Linear)
                q_sha.weight.copy_(q_proj.weight[i * head_dim : (i + 1) * head_dim])
            for i in range(num_kv_heads):
                k_sha = self.k_proj_sha[i]
                v_sha = self.v_proj_sha[i]
                assert isinstance(k_sha, nn.Linear)
                assert isinstance(v_sha, nn.Linear)
                k_sha.weight.copy_(k_proj.weight[i * head_dim : (i + 1) * head_dim])
                v_sha.weight.copy_(v_proj.weight[i * head_dim : (i + 1) * head_dim])

    def forward(
        self,
        normed: torch.Tensor,
        rope_emb_sin: torch.Tensor,
        rope_emb_cos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        full_att_4d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute split-head attention for all heads.

        Parameters
        ----------
        normed
            [B, Ls, D] layer-normed hidden states.
        rope_emb_sin
            [B, Ls, 1, D/2] RoPE sine embeddings for suffix.
        rope_emb_cos
            [B, Ls, 1, D/2] RoPE cosine embeddings for suffix.
        k_cache
            [B, Lp, H_kv, D] prefix key cache (already RoPE-applied).
        v_cache
            [B, Lp, H_kv, D] prefix value cache.
        full_att_4d
            [B, 1, Ls, Lp+Ls] additive mask (0 allowed, -1e4 blocked).

        Returns
        -------
        output : torch.Tensor
            [B, Ls, hidden_size] attention output after o_proj.
        """
        bsize, suffix_len, _ = normed.shape

        # Per-head Q projections: each [B, Ls, 1, head_dim]
        q_list = [q(normed).unsqueeze(2) for q in self.q_proj_sha]

        # Per-KV-head K/V projections: each [B, Ls, 1, head_dim]
        k_list = [k(normed).unsqueeze(2) for k in self.k_proj_sha]
        v_list = [v(normed).unsqueeze(2) for v in self.v_proj_sha]

        # Apply RoPE per-head (H=1) to avoid materializing the full multi-head
        # tensor, which exceeds on-device memory at inference time.
        q_list = [apply_rope_direct(q, rope_emb_sin, rope_emb_cos) for q in q_list]
        k_list = [apply_rope_direct(k, rope_emb_sin, rope_emb_cos) for k in k_list]

        # Split prefix KV cache per KV head and concat with suffix
        k_full = [
            torch.cat([k_cache[:, :, i : i + 1, :], k_list[i]], dim=1)
            for i in range(self.num_kv_heads)
        ]
        v_full = [
            torch.cat([v_cache[:, :, i : i + 1, :], v_list[i]], dim=1)
            for i in range(self.num_kv_heads)
        ]

        # GQA expansion: repeat each KV head for its group of Q heads
        k_full = [k for k in k_full for _ in range(self.num_kv_groups)]
        v_full = [v for v in v_full for _ in range(self.num_kv_groups)]

        # Per-head attention
        scale = self.head_dim**-0.5
        att_outputs: list[torch.Tensor] = []
        for q, k, v in zip(q_list, k_full, v_full, strict=False):
            # q: [B, Ls, 1, D] -> [B, 1, Ls, D]
            q_mat = q.to(torch.float32).transpose(1, 2)
            # k: [B, Lt, 1, D] -> [B, 1, Lt, D]
            k_mat = k.to(torch.float32).transpose(1, 2)

            att = torch.matmul(q_mat, k_mat.transpose(2, 3))
            att *= scale
            att = att + full_att_4d.to(att.dtype)
            probs = torch.nn.functional.softmax(att, dim=-1)

            # v: [B, Lt, 1, D] -> [B, 1, Lt, D]
            v_mat = v.permute(0, 2, 1, 3)
            out = torch.matmul(probs.to(v_mat.dtype), v_mat)  # [B, 1, Ls, D]
            att_outputs.append(out)

        # Concat heads: list of [B, 1, Ls, D] -> [B, H, Ls, D]
        att_output = torch.cat(att_outputs, dim=1)
        att_output = att_output.permute(0, 2, 1, 3)  # [B, Ls, H, D]
        att_output = att_output.reshape(
            bsize, suffix_len, self.num_att_heads * self.head_dim
        )

        return self.o_proj(att_output)


class GemmaMLPSplitLinear(torch.nn.Module):
    """
    Wrap a GemmaMLP and replace its large Linear layers with multiple
    smaller Linear layers, each with out_features or in_features at most
    max_mlp_dim. This helps on-device ML by avoiding a single large
    projection (e.g., 16384).

    Splitting strategy:
      - gate_proj, up_proj: split along out_features and process
        per-chunk.
      - down_proj: split along in_features and process per-chunk, then
        sum partial outputs.

    Memory-friendly forward:
      - Delay concatenation and keep tensors small. For each chunk, run
        gate/up projections, apply activation on the small gate chunk,
        do elementwise product with the small up chunk, pass through the
        matching down chunk, and accumulate the result. This avoids
        materializing the full 16384-wide mid activation.
    """

    def __init__(self, model: GemmaMLP, max_mlp_dim: int = 2048) -> None:
        super().__init__()
        self.hidden_size: int = model.hidden_size
        self.intermediate_size: int = model.intermediate_size
        self.act_fn = model.act_fn

        # Preserve config reference if present.
        self.config = getattr(model, "config", None)

        # Source weights / metadata.
        device = model.gate_proj.weight.device
        dtype = model.gate_proj.weight.dtype

        in_f: int = self.hidden_size
        mid_f: int = self.intermediate_size
        out_f: int = self.hidden_size

        def _make_chunks(total_n: int, max_n: int) -> list[int]:
            sizes: list[int] = []
            rem = total_n
            while rem > 0:
                step = min(max_n, rem)
                sizes.append(step)
                rem -= step
            return sizes

        # Build chunks for the intermediate dim.
        mid_chunks: list[int] = _make_chunks(mid_f, max_mlp_dim)

        # -------- gate_proj and up_proj (in -> mid; split by out) -----
        self.gate_proj_chunks = nn.ModuleList()
        self.up_proj_chunks = nn.ModuleList()

        start = 0
        for sz in mid_chunks:
            gate = nn.Linear(in_f, sz, bias=False, device=device, dtype=dtype)
            up = nn.Linear(in_f, sz, bias=False, device=device, dtype=dtype)

            with torch.no_grad():
                g_w = model.gate_proj.weight[start : start + sz, :]
                u_w = model.up_proj.weight[start : start + sz, :]
                gate.weight.copy_(g_w)
                up.weight.copy_(u_w)

            self.gate_proj_chunks.append(gate)
            self.up_proj_chunks.append(up)
            start += sz

        # -------- down_proj (mid -> out; split by in, then sum) --------
        self.down_proj_chunks = nn.ModuleList()

        start = 0
        for sz in mid_chunks:
            down = nn.Linear(sz, out_f, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                # Original W shape: [out_f, mid_f]. Slice cols.
                w_slice = model.down_proj.weight[:, start : start + sz]
                down.weight.copy_(w_slice)
            self.down_proj_chunks.append(down)
            start += sz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: down_proj(act(gate_proj(x)) * up_proj(x)) using chunked
        projections. Keeps tensors small by processing one chunk at a
        time and accumulating the output, avoiding large concatenations.

        Parameters
        ----------
        x
            Input tensor [..., hidden_size].

        Returns
        -------
        result : torch.Tensor
            Output tensor [..., hidden_size].
        """
        y_accum: torch.Tensor | None = None

        # Process each (gate, up, down) chunk trio independently to
        # avoid building a full 16384-wide activation.
        for gate, up, down in zip(
            self.gate_proj_chunks,
            self.up_proj_chunks,
            self.down_proj_chunks,
            strict=False,
        ):
            g = gate(x)  # [..., sz]
            u = up(x)  # [..., sz]
            mid = self.act_fn(g)  # apply activation while tensor is small
            mid = mid * u  # GLU-like elementwise product

            out_chunk = down(mid)  # [..., out_f]
            y_accum = out_chunk if y_accum is None else y_accum + out_chunk

        # y_accum is guaranteed to be set since there is at least one
        # chunk (intermediate_size > 0 in valid configs).
        assert y_accum is not None
        return y_accum
