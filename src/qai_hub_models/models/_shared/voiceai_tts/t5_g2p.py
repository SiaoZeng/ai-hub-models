# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import Module
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models._shared.common import replace_module_recursively
from qai_hub_models.models._shared.voiceai_tts.t5_attention import T5AttentionMod
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

MAX_NUM_INPUT_IDS = 50
NUM_DECODER_BLOCKS = 4
CHARSIU_MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"


class T5Encoder(BaseModel):
    def __init__(self, t5model: T5ForConditionalGeneration) -> None:
        super().__init__()
        self.model = t5model
        self.embed_tokens = t5model.encoder.embed_tokens
        self.block = t5model.encoder.block
        self.final_layer_norm = t5model.encoder.final_layer_norm

    def forward(
        self, input_ids: Tensor, encoder_attention_mask: Tensor
    ) -> list[Tensor]:
        """
        Parameters
        ----------
        input_ids
            shape of (1, MAX_NUM_INPUT_IDS)
        encoder_attention_mask
            shape of (1, MAX_NUM_INPUT_IDS)

        Returns
        -------
        list[Tensor]
           a list of key value states
        """
        input_embeds = self.embed_tokens(input_ids)
        extended_attention_mask = -10000.0 * (1 - encoder_attention_mask).unsqueeze(
            0
        ).unsqueeze(0)
        position_bias = None
        hidden_states = input_embeds

        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = []
        assert isinstance(self.model.decoder, Module) and isinstance(
            self.model.decoder.block, Iterable
        )
        for decoder_block in self.model.decoder.block:
            cross_attn = decoder_block.layer[1].EncDecAttention
            key_states = cross_attn.k(hidden_states)
            key_states = key_states.view(
                key_states.shape[0],
                -1,
                cross_attn.n_heads,
                cross_attn.key_value_proj_dim,
            ).transpose(1, 2)
            value_states = cross_attn.v(hidden_states)
            value_states = value_states.view(
                value_states.shape[0],
                -1,
                cross_attn.n_heads,
                cross_attn.key_value_proj_dim,
            ).transpose(1, 2)
            outputs.append(key_states)
            outputs.append(value_states)

        return outputs

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "input_ids": TensorSpec(shape=(1, MAX_NUM_INPUT_IDS), dtype="int32"),
            "encoder_attention_mask": TensorSpec(
                shape=(1, MAX_NUM_INPUT_IDS), dtype="int32"
            ),
        }

    def get_output_names(self, num_blocks: int = NUM_DECODER_BLOCKS) -> list[str]:
        names = []
        for i in range(num_blocks):
            names.append(f"block_{i}_cross_key_states")
            names.append(f"block_{i}_cross_value_states")
        return names

    @classmethod
    def from_pretrained(cls) -> Self:
        t5model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_ID).eval()
        return cls(t5model)

    def component_precision(self) -> Precision:
        return Precision.w8a16


class T5Decoder(BaseModel):
    def __init__(
        self, t5model: T5ForConditionalGeneration, max_num_input_ids: int
    ) -> None:
        super().__init__()
        self.model = t5model
        self.embed_tokens = t5model.decoder.embed_tokens
        self.block = t5model.decoder.block
        self.final_layer_norm = t5model.decoder.final_layer_norm
        self.max_num_input_ids = max_num_input_ids

        n_heads = self.block[0].layer[0].SelfAttention.n_heads
        position_bias_len = n_heads * max_num_input_ids
        self.position_bias_embedding = torch.nn.Embedding(
            max_num_input_ids, position_bias_len
        )
        position_bias_weight = torch.zeros(
            [max_num_input_ids, position_bias_len], dtype=torch.float32
        )
        for idx in range(max_num_input_ids):
            position_bias = self.compute_position_bias(idx + 1, max_num_input_ids)
            position_bias_weight[idx, :] = position_bias.flatten()
        self.position_bias_embedding.weight = torch.nn.Parameter(position_bias_weight)

        replace_module_recursively(self.block, T5Attention, T5AttentionMod)

    def compute_position_bias(self, key_length: int, max_length: int) -> Tensor:
        attn = self.block[0].layer[0].SelfAttention
        position_bias = attn.compute_bias(key_length, key_length)[:, :, -1:, :]
        position_bias_masked = torch.full(
            [
                position_bias.shape[0],
                position_bias.shape[1],
                position_bias.shape[2],
                max_length,
            ],
            -10000.0,
            dtype=torch.float32,
        )
        position_bias_masked[..., : key_length - 1] = position_bias[
            ..., : key_length - 1
        ]
        position_bias_masked[..., -1] = position_bias[..., -1]
        return position_bias_masked

    def forward(
        self,
        input_ids: Tensor,
        encoder_attention_mask: Tensor,
        position: Tensor,
        *past_key_values: Tensor,
    ) -> tuple[Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            shape of (1, 1)
        encoder_attention_mask
            shape of (1, q_len)
        position
            shape of (1, 1)
        *past_key_values
            a list of previous key-value states, each state is shape of
            (batch_size, n_heads, q_len - 1, dim_per_head) or (batch_size, n_heads, q_len, dim_per_head)

        Returns
        -------
        output : tuple[Tensor, ...]
            logits
                predicted logits
            present_key_values
                updated key values
        """
        input_embeds = self.embed_tokens(input_ids)
        encoder_extended_attention_mask = -10000.0 * (
            1 - encoder_attention_mask
        ).unsqueeze(0).unsqueeze(0)
        encoder_decoder_position_bias = encoder_extended_attention_mask
        hidden_states = input_embeds

        present_key_values = []

        position_bias = self.position_bias_embedding(position).view(
            1, self.block[0].layer[0].SelfAttention.n_heads, 1, -1
        )

        for i, layer_module in enumerate(self.block):
            past_key_value = []
            past_key_value.append(past_key_values[4 * i])
            past_key_value.append(past_key_values[4 * i + 1])
            past_key_value.append(past_key_values[4 * i + 2])
            past_key_value.append(past_key_values[4 * i + 3])
            layer_outputs = layer_module(
                hidden_states,
                position_bias=position_bias,
                # Truthy non-None value triggers cross-attention without recomputing encoder states (already in past_key_value).
                encoder_hidden_states=True,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False,
            )
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            # Return just new key and value projection for self attn
            present_key_values.append(present_key_value_state[0])
            present_key_values.append(present_key_value_state[1])

        hidden_states = self.final_layer_norm(hidden_states)

        logits = self.model.lm_head(hidden_states)  # type: ignore[operator]
        return logits, *present_key_values

    def get_input_spec(self) -> InputSpec:
        n_heads = self.block[0].layer[0].SelfAttention.n_heads
        dim_per_head = self.block[0].layer[0].SelfAttention.key_value_proj_dim
        q_len = self.max_num_input_ids
        num_blocks = len(self.block)
        specs: InputSpec = {
            "input_ids": TensorSpec(shape=(1, 1), dtype="int32"),
            "encoder_attention_mask": TensorSpec(shape=(1, q_len), dtype="int32"),
            "position": TensorSpec(shape=(1, 1), dtype="int32"),
        }
        for i in range(num_blocks):
            specs[f"block_{i}_past_self_key_states"] = TensorSpec(
                shape=(1, n_heads, q_len - 1, dim_per_head),
                dtype="float32",
            )
            specs[f"block_{i}_past_self_value_states"] = TensorSpec(
                shape=(1, n_heads, q_len - 1, dim_per_head),
                dtype="float32",
            )
            specs[f"block_{i}_cross_key_states"] = TensorSpec(
                shape=(1, n_heads, q_len, dim_per_head),
                dtype="float32",
            )
            specs[f"block_{i}_cross_value_states"] = TensorSpec(
                shape=(1, n_heads, q_len, dim_per_head),
                dtype="float32",
            )
        return specs

    def get_output_names(self) -> list[str]:
        names = ["logits"]
        for i in range(len(self.block)):
            names.append(f"block_{i}_present_self_key_states")
            names.append(f"block_{i}_present_self_value_states")
        return names

    @classmethod
    def from_pretrained(cls) -> Self:
        t5model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_ID).eval()
        return cls(t5model, MAX_NUM_INPUT_IDS)

    def component_precision(self) -> Precision:
        return Precision.w8a16
