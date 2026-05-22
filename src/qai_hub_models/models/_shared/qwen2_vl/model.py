# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen2.5-VL text model classes.

These classes extend the existing Qwen2 LLM classes to work with `inputs_embeds`
instead of `input_ids`, as required for VLM integration with Genie SDK.

Key difference from standard Qwen2:
- Uses LLMIOType.genie_input_embeds instead of genie_input_ids
- The embedding layer is bypassed at the model input
- Embedding lookup happens on HOST before merging with vision embeddings
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLMDynamic_AIMETOnnx,
)
from qai_hub_models.models._shared.qwen2.model import (
    Qwen2Base,
    Qwen2Base_AIMETOnnx,
    Qwen2Base_QNN,
    QwenPositionProcessor,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

    from qai_hub_models.utils.input_spec import InputSpec

from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm._utils import (
    _apply_int8_kv_cache_tying_and_lm_head,
    _get_kv_io_map,
)
from qai_hub_models.utils.system_info import has_recommended_memory

logger = logging.getLogger(__name__)

# Chat format constants (same as Qwen2)
START_HEADER = "<|im_start|>"
END_HEADER = "<|im_end|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
END_TOKENS = {"<|im_end|>", "<|endoftext|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant."
DEFAULT_USER_PROMPT = "Give me a short introduction to large language model."

# Vision token placeholder (Qwen2.5-VL format)
VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class _VLMCausalLMWrapper(torch.nn.Module):
    """Wrap text_model + lm_head so the whole forward lives inside one Module.

    This is necessary for ``torch.export`` (dynamo) tracing: when
    ``self.model`` is just the text encoder and the lm_head sits outside,
    dynamo cannot capture the KV-cache output tensors.  By combining them
    here, the forward graph is self-contained and all 57 outputs
    (logits + 56 KV) are preserved.
    """

    def __init__(self, text_model: torch.nn.Module, lm_head: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return {
            "logits": logits,
            "past_key_values": outputs.past_key_values,
        }


def get_vlm_config(model_ckpt: str | os.PathLike | Path | None) -> PretrainedConfig:
    """Construct and return a HuggingFace LLM config."""
    from transformers import AutoConfig

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    llm_config.text_config._attn_implementation = "eager"
    llm_config.text_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    llm_config.text_config.use_cache = True

    return llm_config


class Qwen2VLTextBase(Qwen2Base):
    """
    Base class for Qwen2.5-VL text model.

    Key difference from Qwen2Base:
    - Uses LLMIOType.genie_input_embeds
    - Input is embeddings, not token IDs
    - Loads from full VLM checkpoint and extracts text model
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    # We use the full VLM class for loading, then extract text model
    LMClass = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration  # type: ignore[assignment]

    # Store reference to full VLM for embedding extraction
    _full_vlm: torch.nn.Module | None = None

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        spec = super().get_chat_template()
        assert spec is not None
        spec["vision_start"] = "<|vision_start|>"
        spec["vision_end"] = "<|vision_end|>"
        return spec

    @classmethod
    def edit_llm_config(cls, llm_config: PretrainedConfig) -> PretrainedConfig:
        """
        Extract text_config from the full Qwen2.5VL config.

        The text model operations use text_config.
        """
        # If we already have a text config, return it
        if llm_config.model_type == "qwen2":
            return llm_config

        # Extract text_config from full VLM config
        if hasattr(llm_config, "text_config"):
            return llm_config.text_config

        return llm_config

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int | None = None,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        """
        Get input spec for VLM text model.

        Uses inputs_embeds instead of input_ids. Position embeddings (cos/sin)
        are pre-computed externally and passed as inputs.
        """
        # Use explicit head_dim if provided, otherwise derive from hidden_size
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        embed_dim = head_dim // 2

        input_spec: InputSpec = {}

        # VLM uses inputs_embeds
        input_spec["inputs_embeds"] = (
            (1, sequence_length, hidden_size),
            "float32",
        )

        input_spec["attention_mask"] = (
            (1, 1, sequence_length, context_length),
            "float32",
        )

        input_spec["position_ids_cos"] = (
            (1, 1, sequence_length, embed_dim),
            "float32",
        )
        input_spec["position_ids_sin"] = (
            (1, 1, sequence_length, embed_dim),
            "float32",
        )

        # KV cache for each layer
        assert sequence_length < context_length, (
            "It is currently not supported to set input sequence length to the same "
            "as or longer than context length."
        )

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = (
                (
                    num_key_value_heads,
                    1,
                    head_dim,
                    context_length - sequence_length,
                ),
                "float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = (
                (
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    head_dim,
                ),
                "float32",
            )
        return input_spec

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        is_token_generator: bool = False,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
        _skip_optimizations: list[str] | None = None,
    ) -> None:
        """
        Initialize Qwen2.5-VL text model.

        Overrides parent to load from full VLM checkpoint and extract text model.
        """
        from qai_hub_models.models._shared.llm.model import get_tokenizer

        # Initialize nn.Module first to set up 'training' attribute
        torch.nn.Module.__init__(self)

        if host_device is None:
            host_device = torch.device("cpu")

        self.skip_optimizations = _skip_optimizations
        self.checkpoint = checkpoint

        has_recommended_memory(self.min_memory_recommended)

        self.monkey_patch(skip_optimizations=self.skip_optimizations)
        llm_config = get_vlm_config(self.checkpoint)
        # Keep original config for full VLM operations
        self._original_llm_config = llm_config
        self.llm_config = self.edit_llm_config(llm_config)
        self._verify_ckpt()
        self.tokenizer = get_tokenizer(checkpoint)

        # Cache HF image processor config for vision preprocessing metadata
        from transformers import AutoProcessor

        self._image_processor = AutoProcessor.from_pretrained(
            checkpoint
        ).image_processor

        # Load model using our custom loader
        model, full_vlm, lm_head = self.load_llm_from_checkpoint(
            checkpoint=self.checkpoint,
            llm_config=self.llm_config,
            load_pretrained=load_pretrained,
        )
        model.eval()

        # Extract and store embedding weights before discarding full VLM
        if full_vlm is not None:
            self._embedding_weights = (
                full_vlm.get_input_embeddings().weight.data.clone()  # type: ignore[operator]
            )
        else:
            self._embedding_weights = None

        # Create embedding (use original config for vocab_size)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length,
            config=llm_config,  # type: ignore[arg-type]
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        # Convert lm_head to Conv2d (not part of model.named_modules())
        from qai_hub_models.models._shared.llm.model_adaptations import (
            ConvInplaceLinear,
        )

        if isinstance(lm_head, torch.nn.Linear):
            lm_head = ConvInplaceLinear(lm_head)

        # Wrap text_model + lm_head into a single Module so that
        # torch.export (dynamo) can trace the full graph including KV
        # cache outputs.  Without this wrapper, dynamo only captures
        # logits and drops the 56 KV-cache output tensors.
        assert lm_head is not None
        wrapper = _VLMCausalLMWrapper(model, lm_head)
        wrapper.to(host_device)

        self.sequence_length: int = sequence_length
        self.context_length: int = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = wrapper
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    @classmethod
    def load_llm_from_checkpoint(
        cls,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        load_pretrained: bool = True,
    ) -> tuple[torch.nn.Module, torch.nn.Module | None, torch.nn.Module | None]:
        """
        Load the text model from a Qwen2.5-VL checkpoint.

        Returns (text_model, full_vlm, lm_head) tuple. The full_vlm is kept for
        embedding table extraction. The lm_head is needed for logits computation.
        """
        if load_pretrained:
            # Load full VLM with eager attention for ONNX compatibility
            full_vlm = (
                modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint,
                    attn_implementation="eager",  # Required for ONNX export
                )
            )
            # Extract text model (the language model portion)
            text_model = full_vlm.model
            # Extract lm_head for logits computation
            lm_head = full_vlm.lm_head
            return text_model, full_vlm, lm_head
        # Create uninitialized text model
        text_model = modeling_qwen2_5_vl.Qwen2_5_VLTextModel(llm_config)  # type: ignore[arg-type]
        lm_head = torch.nn.Linear(
            llm_config.hidden_size, llm_config.vocab_size, bias=False
        )
        return text_model, None, lm_head

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"  # Note: HuggingFace uses 'inputs_embeds' not 'input_embeds'
        return "input_ids"

    def get_embedding_weights(self) -> torch.Tensor:
        """Get embedding weights from the stored weights or text model."""
        if self._embedding_weights is not None:
            return self._embedding_weights
        # Fallback: get from text model (inside the wrapper)
        text_model = self.model.model if hasattr(self.model, "model") else self.model
        return text_model.embed_tokens.weight.data  # type: ignore[union-attr, return-value]

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the embedding table."""
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        include_image: bool | int = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt with appropriate tags for Qwen2.5-VL.

        Overrides the base class to use Qwen2.5-VL's ChatML format and
        include vision placeholder tokens when processing images.

        Parameters
        ----------
        user_input_prompt
            The user's text prompt. Defaults to DEFAULT_USER_PROMPT.
        system_context_prompt
            System context/instructions. Defaults to DEFAULT_PROMPT_CONTEXT.
        include_image
            Whether to include vision placeholder tokens in the prompt.
            Pass ``True`` or ``1`` for a single image, an ``int > 1`` for
            multiple images, or ``False``/``0`` for text-only.
            Defaults to True.
        enable_thinking
            Whether to enable thinking mode. Qwen2.5-VL doesn't have native
            thinking mode, so this parameter is ignored.
            Defaults to False.
        **kwargs
            Additional arguments (ignored, for compatibility with base class).

        Returns
        -------
        str
            Formatted prompt string with ChatML tags and optional
            vision placeholders.
        """
        if user_input_prompt is None:
            user_input_prompt = DEFAULT_USER_PROMPT
        if system_context_prompt is None:
            system_context_prompt = DEFAULT_PROMPT_CONTEXT

        # For VLM, include one placeholder per image
        num_images = int(include_image) if isinstance(include_image, (bool, int)) else 0
        if num_images > 0:
            placeholders = "\n".join(VISION_PLACEHOLDER for _ in range(num_images))
            user_content = f"{placeholders}\n{user_input_prompt}"
        else:
            user_content = user_input_prompt

        return f"""{START_HEADER}{SYSTEM_ID}
{system_context_prompt}{END_HEADER}
{START_HEADER}{USER_ID}
{user_content}{END_HEADER}
{START_HEADER}{ASSISTANT_ID}
"""

    def _verify_ckpt(self) -> None:
        """Verify checkpoint is compatible with Qwen2.5-VL."""
        valid_model_types = {"qwen2_5_vl", "qwen2_vl", "qwen2"}
        architectures = getattr(self.llm_config, "architectures", None) or []
        if not (
            self.llm_config.model_type in valid_model_types
            or any("Qwen2" in arch for arch in architectures)
        ):
            raise ValueError(
                "Model config is not compatible with Qwen2.5-VL implementation. "
                f"Expected model_type in {valid_model_types}, got '{self.llm_config.model_type}'"
            )

    @staticmethod
    def monkey_patch(skip_optimizations: list[str] | None = None) -> None:
        """
        Apply monkey patches for Qwen2.5VL ONNX export.

        Adaptations applied:
        - SHA (Split-Head Attention) with M-RoPE support
        - RMSNorm rank-4 for hardware efficiency
        - MLP Conv2d (down_proj only, gate/up temporarily disabled)
        - Bypass rotary embeddings (cos/sin pre-computed externally)
        """
        from qai_hub_models.models._shared.qwen2.model import Qwen2_Optimizations
        from qai_hub_models.models._shared.qwen2_vl.model_adaptations import (
            QCQwen2_5_VLMLP,
            SHAQwen2_5_VLAttention,
        )

        # SHA attention (replaces Qwen2_5_VLAttention class)
        if (
            skip_optimizations
            and Qwen2_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_qwen2_5_vl.Qwen2_5_VLAttention = SHAQwen2_5_VLAttention  # type: ignore[misc, unused-ignore]

        # Bypass rotary embedding module — cos/sin are pre-computed
        # externally and passed as the position_ids tuple, just like
        # pure LLMs (Llama, Qwen2, etc.).
        def bypass_RotaryEmbedding(
            self: torch.nn.Module,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        if not hasattr(
            modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding, "_original_forward"
        ):
            modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding.forward
            )
            modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding.forward = (
                bypass_RotaryEmbedding  # type: ignore[assignment, unused-ignore]
            )

        # Bypass M-RoPE position_ids processing in Qwen2_5_VLTextModel.
        # When position_ids is a tuple (pre-computed cos/sin), the HF code
        # crashes on `position_ids.ndim`. We patch the text model's forward
        # to skip ndim checks and causal mask creation for tuple position_ids.
        _original_text_forward = modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward

        def _patched_text_forward(
            self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: Any = None,
            output_attentions: Any = None,
            output_hidden_states: Any = None,
            return_dict: Any = None,
            cache_position: Any = None,
            **kwargs: Any,
        ) -> Any:
            if isinstance(position_ids, tuple):
                # Pre-computed (cos, sin) — skip HF's M-RoPE ndim processing.
                # Directly run decoder layers with our position embeddings.
                from transformers.modeling_outputs import BaseModelOutputWithPast

                output_attentions = (
                    output_attentions
                    if output_attentions is not None
                    else self.config.output_attentions
                )
                output_hidden_states = (
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.config.output_hidden_states
                )
                use_cache = (
                    use_cache if use_cache is not None else self.config.use_cache
                )
                return_dict = (
                    return_dict
                    if return_dict is not None
                    else self.config.use_return_dict
                )

                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

                # position_embeddings = (cos, sin) from bypass_RotaryEmbedding
                position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

                hidden_states = inputs_embeds
                all_hidden_states = () if output_hidden_states else None
                all_self_attns = () if output_attentions else None

                # Per-layer attention mask multipliers for layers with
                # extreme KV cache dynamic ranges.
                # Hardcoded so dynamo can resolve them during tracing.
                _MASK_SCALES = {0: 16.0, 27: 74.0}

                for layer_idx, decoder_layer in enumerate(self.layers):
                    if output_hidden_states:
                        assert all_hidden_states is not None
                        all_hidden_states += (hidden_states,)  # type: ignore[assignment]

                    _s = _MASK_SCALES.get(layer_idx, 1.0)
                    layer_mask = attention_mask * _s

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=layer_mask,
                        position_ids=None,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    hidden_states = layer_outputs[0]
                    if output_attentions:
                        assert all_self_attns is not None
                        all_self_attns += (layer_outputs[1],)  # type: ignore[assignment]

                hidden_states = self.norm(hidden_states)
                if output_hidden_states:
                    assert all_hidden_states is not None
                    all_hidden_states += (hidden_states,)  # type: ignore[assignment]

                if not return_dict:
                    return tuple(
                        v
                        for v in [
                            hidden_states,
                            past_key_values,
                            all_hidden_states,
                            all_self_attns,
                        ]
                        if v is not None
                    )
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )

            return _original_text_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        if not hasattr(modeling_qwen2_5_vl.Qwen2_5_VLTextModel, "_original_forward"):
            modeling_qwen2_5_vl.Qwen2_5_VLTextModel._original_forward = (  # type: ignore[attr-defined]
                _original_text_forward
            )
            modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = _patched_text_forward

        # MLP Conv2d adaptation (Qwen2MLP is used by decoder layers)
        modeling_qwen2_5_vl.Qwen2MLP = QCQwen2_5_VLMLP  # type: ignore[misc, unused-ignore]

    # forward() is intentionally NOT overridden — LLMBase.forward handles
    # the SHA cache setup, model call, and KV-cache output extraction.
    # The _VLMCausalLMWrapper ensures the text_model + lm_head are called
    # together inside a single Module, which is required for torch.export
    # (dynamo) to capture all 57 outputs (logits + 56 KV-cache tensors).


class Qwen2VLTextBase_AIMETOnnx(Qwen2Base_AIMETOnnx):
    """
    AIMET-ONNX quantized version of Qwen2.5-VL text model.

    Uses inputs_embeds instead of input_ids.
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen2VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen2VLTextBase.get_input_prompt_with_tags
    )

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

        # Load embedding weights from checkpoint for VLM models.
        # The ONNX model uses inputs_embeds (not input_ids), so the embedding
        # layer is not part of the ONNX graph. We need to load it separately
        # for token-to-embedding conversion during evaluation/generation.
        self._embedding_weights = None
        if checkpoint is not None:
            embed_path = Path(checkpoint) / "embedding_weights.raw"
            if embed_path.exists():
                import numpy as np

                embed_np = np.fromfile(str(embed_path), dtype=np.float32)
                vocab_size = self.llm_config.vocab_size
                hidden_size = self.llm_config.hidden_size
                self._embedding_weights = torch.from_numpy(
                    embed_np.reshape(vocab_size, hidden_size)
                )

    def get_embedding_weights(self) -> torch.Tensor:
        """Get embedding weights from checkpoint (not from LM head)."""
        if self._embedding_weights is not None:
            return self._embedding_weights
        raise RuntimeError(
            "VLM embedding weights not loaded. Ensure checkpoint contains "
            "embedding_weights.raw or pass an FP model during from_pretrained."
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the stored embedding table.

        Overrides LLM_AIMETOnnx which extracts from LM head weights — incorrect
        for VLM models where the embedding layer is not in the ONNX graph.
        """
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    @classmethod
    def _configure_quant_sim(
        cls, quant_sim: QuantizationSimModel, precision: Precision
    ) -> QuantizationSimModel:
        assert precision == Precision.w4a16
        kv_io_map = _get_kv_io_map(quant_sim)
        return _apply_int8_kv_cache_tying_and_lm_head(
            quant_sim, kv_io_map, use_16x8_matmuls=False
        )

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """Get interleaved (wikitext + AOKVQA) calibration data for VLM.

        VLM models need calibration on both text-only and multimodal inputs
        to produce representative activation ranges. This aligns with the
        GenAITests Interleaved dataset (Wikitext + AOKVQA).
        """
        import math

        import numpy as np
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from transformers import AutoProcessor

        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.datasets.interleaved_aokvqa_wikitext import (
            InterleavedAOKVQAWikiText,
        )
        from qai_hub_models.models._shared.llm.generator import LLM_Generator
        from qai_hub_models.utils.base_dataset import DatasetSplit
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

        if num_samples == 0:
            num_samples = math.ceil(80000 / self.context_length)

        # Use Interleaved dataset (wikitext + AOKVQA) for VLM calibration.
        # This requires a VLM processor for AOKVQA image processing.
        hf_repo = getattr(self, "_hf_repo_name", None)
        if hf_repo is None and self.checkpoint is not None:
            hf_repo = self.checkpoint
        if hf_repo is None:
            hf_repo = self.llm_config._name_or_path
        processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)

        dataset = instantiate_dataset(
            InterleavedAOKVQAWikiText,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=num_samples,
            processor=processor,
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        assert self.EmbeddingClass is not None
        rope_embeddings = self.EmbeddingClass(
            max_length=self.context_length,
            config=self.llm_config,  # type: ignore[arg-type]
        )
        generator = LLM_Generator(
            [self],
            self.tokenizer,
            rope_embeddings,
        )

        # Load HF vision model for multimodal samples
        vision_model = self._load_calibration_vision_model()
        image_token_id = getattr(self.llm_config, "image_token_id", None)

        with self.remove_quantization():
            for sample in tqdm(
                dataloader,
                total=len(dataloader),
                desc="Pre-filling calibration data (interleaved)",
            ):
                # collate_fn returns (input_ids, attention_mask, label)
                # for text-only, or (..., pixel_values, image_grid_thw)
                # for multimodal samples.
                input_ids, attention_mask, *rest = sample
                pixel_values = rest[1] if len(rest) > 1 else None
                image_grid_thw = rest[2] if len(rest) > 2 else None

                if (
                    pixel_values is not None
                    and image_grid_thw is not None
                    and vision_model is not None
                ):
                    inputs_embeds = self._merge_vision_embeddings(
                        input_ids,
                        pixel_values,
                        image_grid_thw,
                        vision_model,
                        image_token_id,
                    )
                    prefill_iter = generator.prefill(
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                    )
                else:
                    prefill_iter = generator.prefill(input_ids, attention_mask)

                for prefilled_inputs in prefill_iter:
                    for i, tensor in enumerate(prefilled_inputs):
                        inputs[i].append(tensor)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def _load_calibration_vision_model(self) -> torch.nn.Module | None:
        """Load the HF vision model for multimodal calibration samples."""
        try:
            from transformers import AutoModel

            hf_repo = getattr(self, "_hf_repo_name", None)
            if hf_repo is None and self.checkpoint is not None:
                hf_repo = self.checkpoint
            if hf_repo is None:
                hf_repo = self.llm_config._name_or_path

            hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            visual = hf_model.visual.eval()
            # Detach from parent to free LLM weights
            del hf_model
            return visual
        except Exception:
            logger.warning(
                "Failed to load vision model for calibration; "
                "multimodal samples will use text-only prefill.",
                exc_info=True,
            )
            return None

    def _merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        vision_model: torch.nn.Module,
        image_token_id: int | None,
    ) -> torch.FloatTensor:
        """Run HF vision model on pixel_values and merge into text embeddings."""
        with torch.no_grad():
            vision_embeddings = vision_model(pixel_values, grid_thw=image_grid_thw)

        text_embeddings = self.convert_input_ids_to_embeddings(input_ids)

        if image_token_id is not None:
            image_mask = input_ids == image_token_id
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(text_embeddings)
            text_embeddings = text_embeddings.masked_scatter(
                image_mask_expanded,
                vision_embeddings.to(
                    device=text_embeddings.device, dtype=text_embeddings.dtype
                ),
            )

        return torch.FloatTensor(text_embeddings)

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Adapt AIMET encodings for VLM model.

        VLM models use inputs_embeds instead of input_ids, so the embedding
        layer (embed_tokens) is not part of the exported ONNX model. We skip
        the embedding weight handling that the base Qwen2 model does.
        """
        from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

        with open(src_encodings_path) as read_file:
            encodings = json.load(read_file)

        model = onnx.load(onnx_model_path)

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # Skip embedding layer handling - VLM doesn't have embed_tokens in ONNX
        # (it uses inputs_embeds directly)

        # Copy weight encodings to param encodings (same as base Qwen2)
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        propagate_memory_encodings(encodings, model)

        # convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        # Rewrite the shipped encodings file into the layout the downstream
        # split/compile step expects.
        if bundle.aimet_encodings_path is not None:
            encodings_path = str(bundle.aimet_encodings_path)
            self._adapt_aimet_encodings(
                encodings_path, encodings_path, str(bundle.onnx_graph_path)
            )
        return bundle


class Qwen2VLDynamic_AIMETOnnx(LLMDynamic_AIMETOnnx, Qwen2VLTextBase_AIMETOnnx):
    """Dynamic-shape variant of Qwen2VLTextBase_AIMETOnnx."""

    FPModel = Qwen2VLTextBase


class Qwen2VLTextBase_QNN(Qwen2Base_QNN):
    """
    QNN version of Qwen2.5-VL text model.

    Uses inputs_embeds instead of input_ids.
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen2VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen2VLTextBase.get_input_prompt_with_tags
    )


# Re-export position processor (same as Qwen2)
Qwen2VLPositionProcessor = QwenPositionProcessor
