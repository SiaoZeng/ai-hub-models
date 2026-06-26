# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

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
from transformers.models.qwen3_vl import modeling_qwen3_vl

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)
from qai_hub_models.models._shared.lm_driver.generator import (
    PrecomputedCosSinGeneratorMixin,
    TransposedKVGeneratorMixin,
)
from qai_hub_models.models._shared.lm_driver.qwen3_vl import (
    Qwen3VL_Generator,
    Qwen_3_VL,
)
from qai_hub_models.models._shared.qwen3.model import (
    Qwen3Base,
    Qwen3Base_AIMETOnnx,
    Qwen3Base_QNN,
    Qwen3PositionProcessor,
)
from qai_hub_models.models._shared.qwen3_vl.vision_encoder import (
    Qwen3VLVisionWrapper,
)
from qai_hub_models.models._shared.vlm.model import (
    VLMDynamic_AIMETOnnx,
)
from qai_hub_models.utils.input_spec import TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

    from qai_hub_models.utils.base_dataset import BaseDataset
    from qai_hub_models.utils.input_spec import InputSpec


from qai_hub_models.utils.system_info import has_recommended_memory

logger = logging.getLogger(__name__)

END_TOKENS = {"<|im_end|>", "<|endoftext|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant."
DEFAULT_USER_PROMPT = "Give me a short introduction to large language model."


def _vlm_eval_dataset_classes() -> list[type[BaseDataset]]:
    """Eval datasets for VLM models: the text-only LLM tasks plus MMMU and multimodal prompts."""
    from qai_hub_models.datasets.mmmu import MMMU
    from qai_hub_models.datasets.prompts import MultimodalPrompts
    from qai_hub_models.models._shared.llm.model import LLMBase

    return [*LLMBase.get_eval_dataset_classes(), MMMU, MultimodalPrompts]


class HubCompatibleQwen3VLGenerator(  # type: ignore[misc]
    PrecomputedCosSinGeneratorMixin, TransposedKVGeneratorMixin, Qwen3VL_Generator
):
    pass


class _VLMCausalLMWrapper(torch.nn.Module):
    """Wrap language_model + lm_head so the whole forward lives inside one Module.

    This is necessary for ``torch.export`` (dynamo) tracing: when
    ``self.model`` is just the text encoder and the lm_head sits outside,
    dynamo cannot capture the KV-cache output tensors. By combining them
    here, the forward graph is self-contained and all outputs
    (logits + KV) are preserved.

    For Qwen3-VL, the language model's forward accepts deepstack kwargs
    (visual_pos_masks, deepstack_visual_embeds) which must be passed through.
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
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return {
            "logits": logits,
            "past_key_values": outputs.past_key_values,
        }


def get_vlm_config(model_ckpt: str | os.PathLike | Path | None) -> PretrainedConfig:
    """Construct and return a HuggingFace LLM config for Qwen3-VL."""
    from transformers import AutoConfig

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    # The config may be the full VLM config (Qwen3VLConfig, has .text_config) when
    # loaded from the HF repo, or the bare text config (Qwen3VLTextConfig, no
    # .text_config) when loaded from a split/quantized checkpoint. Resolve the
    # text config for either layout, mirroring _verify_ckpt's handling.
    text_config = getattr(llm_config, "text_config", llm_config)
    text_config._attn_implementation = "eager"
    text_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    text_config.use_cache = True

    return llm_config


class Qwen3VLTextBase(Qwen3Base):
    """
    Base class for Qwen3-VL text model.

    Key differences from Qwen3Base:
    - Uses LLMIOType.genie_input_embeds
    - Input is embeddings, not token IDs
    - Loads from full VLM checkpoint and extracts text model
    - Handles deepstack visual embeddings injected at intermediate layers
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    GeneratorClass = HubCompatibleQwen3VLGenerator

    # We use the full VLM class for loading, then extract text model
    LMClass = modeling_qwen3_vl.Qwen3VLForConditionalGeneration  # type: ignore[assignment, unused-ignore]

    VisionModelWrapper = Qwen3VLVisionWrapper

    # Store reference to full VLM for embedding extraction
    _full_vlm: torch.nn.Module | None = None

    @classmethod
    def get_visual_output_names(cls, config: PretrainedConfig) -> tuple[str, ...]:
        return Qwen_3_VL.get_visual_output_names()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        spec = super().get_chat_template()
        assert spec is not None
        spec["vision_start"] = "<|vision_start|>"
        spec["vision_end"] = "<|vision_end|>"
        return spec

    @classmethod
    def edit_llm_config(cls, llm_config: PretrainedConfig) -> PretrainedConfig:
        """Extract text_config from the full Qwen3VL config."""
        if llm_config.model_type == "qwen3":
            return llm_config

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
        num_deepstack_layers: int = 0,
        num_visual_tokens: int = 256,
    ) -> InputSpec:
        """
        Get input spec for VLM text model.

        Uses inputs_embeds instead of input_ids. Position embeddings (cos/sin)
        are pre-computed externally and passed as inputs.
        Includes deepstack visual embeddings injected at intermediate layers.
        """
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        embed_dim = head_dim // 2

        input_spec: InputSpec = {}

        # VLM uses inputs_embeds
        input_spec["inputs_embeds"] = TensorSpec(
            shape=(1, sequence_length, hidden_size),
            dtype="float32",
        )

        input_spec["attention_mask"] = TensorSpec(
            shape=(1, 1, sequence_length, context_length),
            dtype="float32",
        )

        input_spec["position_ids_cos"] = TensorSpec(
            shape=(1, 1, sequence_length, embed_dim),
            dtype="float32",
        )
        input_spec["position_ids_sin"] = TensorSpec(
            shape=(1, 1, sequence_length, embed_dim),
            dtype="float32",
        )

        # KV cache for each layer
        assert sequence_length < context_length, (
            "It is currently not supported to set input sequence length to the same "
            "as or longer than context length."
        )

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = TensorSpec(
                shape=(
                    num_key_value_heads,
                    1,
                    head_dim,
                    context_length - sequence_length,
                ),
                dtype="float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = TensorSpec(
                shape=(
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    head_dim,
                ),
                dtype="float32",
            )

        # Deepstack: visual_pos_masks marks which sequence positions contain
        # vision tokens; deepstack_visual_embeds_i are per-layer visual features
        # injected at intermediate decoder layers.
        if num_deepstack_layers > 0:
            input_spec["visual_pos_masks"] = TensorSpec(
                shape=(1, sequence_length),
                dtype="bool",
            )
            for i in range(num_deepstack_layers):
                input_spec[f"deepstack_visual_embeds_{i}"] = TensorSpec(
                    shape=(num_visual_tokens, hidden_size),
                    dtype="float32",
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
        Initialize Qwen3-VL text model.

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
        # (patch_size, merge_size, mean/std). A split/quantized checkpoint may
        # only contain tokenizer files (no preprocessor_config.json), in which
        # case AutoProcessor returns a bare tokenizer with no .image_processor;
        # fall back to the base HF repo, which carries the same static metadata.
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self._image_processor = getattr(processor, "image_processor", None)
        if self._image_processor is None:
            hf_repo = getattr(self, "_hf_repo_name", None)
            assert hf_repo is not None, (
                "Checkpoint has no image processor and no _hf_repo_name fallback."
            )
            self._image_processor = AutoProcessor.from_pretrained(
                hf_repo, trust_remote_code=True
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
            config=llm_config.text_config,
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

        # Wrap text_model + lm_head into a single Module
        assert lm_head is not None
        wrapper = _VLMCausalLMWrapper(model, lm_head)
        wrapper.to(host_device).float()

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
        Load the text model from a Qwen3-VL checkpoint.

        Returns (text_model, full_vlm, lm_head) tuple. The full_vlm is kept for
        embedding table extraction. The lm_head is needed for logits computation.

        Qwen3-VL hierarchy: model.model (Qwen3VLModel) contains .language_model
        (Qwen3VLTextModel) and .visual (Qwen3VLVisionModel).
        """
        if load_pretrained:
            full_vlm = (
                modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
                    checkpoint,
                    attn_implementation="eager",
                )
            )
            # Extract the text model (language_model inside model)
            text_model = full_vlm.model.language_model
            lm_head = full_vlm.lm_head
            return text_model, full_vlm, lm_head
        # Create uninitialized text model
        text_model = modeling_qwen3_vl.Qwen3VLTextModel(llm_config)  # type: ignore[arg-type, unused-ignore]
        lm_head = torch.nn.Linear(
            llm_config.hidden_size, llm_config.vocab_size, bias=False
        )
        return text_model, None, lm_head

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    def get_embedding_weights(self) -> torch.Tensor:
        """Get embedding weights from the stored weights or text model."""
        if self._embedding_weights is not None:
            return self._embedding_weights
        text_model = self.model.model if hasattr(self.model, "model") else self.model
        return text_model.embed_tokens.weight.data  # type: ignore[union-attr, return-value]

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the embedding table."""
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Override to extract deepstack inputs from *args and pass to model."""
        from transformers.cache_utils import DynamicCache

        from qai_hub_models.models._shared.llm.model import (  # type: ignore[attr-defined]
            SHADynamicCacheNewValueOnly,
        )

        # args layout: (cos, sin, *kv_caches, [visual_pos_masks, *deepstack_embeds])
        position_ids = args[:2]
        num_kv_tensors = self.llm_config.num_hidden_layers * 2
        past_key_values_tensors = args[2 : 2 + num_kv_tensors]
        extra_args = args[2 + num_kv_tensors :]

        # Extract deepstack inputs if present
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if len(extra_args) > 0:
            visual_pos_masks = extra_args[0]
            if len(extra_args) > 1:
                deepstack_visual_embeds = list(extra_args[1:])

        # Build KV cache
        assert isinstance(self.llm_config.num_key_value_heads, int)
        if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(
                zip(
                    past_key_values_tensors[::2],
                    past_key_values_tensors[1::2],
                    strict=False,
                )
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                k = torch.cat(k_split, dim=1).permute(0, 1, 3, 2)
                v = torch.cat(v_split, dim=1)
                kv_cache.update(k, v, layer_idx, {})
        else:
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(
                    past_key_values_tensors[::2],
                    past_key_values_tensors[1::2],
                    strict=False,
                )
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                kv_cache.update(k_split, v_split, layer_idx, {})

        model_kwargs: dict[str, Any] = {
            self.main_input_name: input_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
        }
        if visual_pos_masks is not None:
            model_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_visual_embeds is not None:
            model_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

        out = self.model(**model_kwargs)

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
                if hasattr(out_cache, "key_cache"):
                    keys = out_cache.key_cache[layer]
                    values = out_cache.value_cache[layer]
                elif hasattr(out_cache.layers[layer], "keys"):
                    keys = out_cache.layers[layer].keys
                    values = out_cache.layers[layer].values
                else:
                    keys = out_cache.layers[layer][0]
                    values = out_cache.layers[layer][1]

                seq_len = input_tokens.shape[1]
                k = keys[:, :, -seq_len:, :].permute(1, 0, 3, 2)
                v = values[:, :, -seq_len:, :].permute(1, 0, 2, 3)

            elif hasattr(out_cache, "key_cache"):
                k = torch.cat(out_cache.key_cache[layer], dim=0)
                v = torch.cat(out_cache.value_cache[layer], dim=0)
            elif hasattr(out_cache.layers[layer], "keys"):
                k = torch.cat(out_cache.layers[layer].keys, dim=0)
                v = torch.cat(out_cache.layers[layer].values, dim=0)
            else:
                k = torch.cat(out_cache.layers[layer][0], dim=0)
                v = torch.cat(out_cache.layers[layer][1], dim=0)
            flat_output_past_key_values += [k, v]

        return [out["logits"], *flat_output_past_key_values]

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        include_image: bool | int = False,
        enable_thinking: bool = False,
        tokenizer: PreTrainedTokenizer | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt with appropriate tags for Qwen3-VL.

        Overrides the base class to use Qwen3-VL's ChatML format and
        include vision placeholder tokens when processing images.
        Uses the tokenizer's chat template with structured content for images.

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
            Defaults to False.
        enable_thinking
            Whether to enable thinking mode.
            Defaults to False.
        tokenizer
            Required. The tokenizer to use for applying the chat template.
        add_generation_prompt
            Whether to append the assistant turn header.
            Defaults to True.
        continue_final_message
            Whether to continue the final message instead of starting a new one.
            Defaults to False.
        **kwargs
            Additional arguments (ignored, for compatibility with base class).

        Returns
        -------
        str
            Formatted prompt string with ChatML tags and optional
            vision placeholders.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required for get_input_prompt_with_tags")
        if user_input_prompt is None:
            user_input_prompt = DEFAULT_USER_PROMPT
        if system_context_prompt is None:
            system_context_prompt = DEFAULT_PROMPT_CONTEXT

        num_images = int(include_image) if isinstance(include_image, (bool, int)) else 0
        if num_images > 0:
            content: list[dict[str, str]] = [
                {"type": "image"} for _ in range(num_images)
            ]
            content.append({"type": "text", "text": user_input_prompt})
        else:
            content = [{"type": "text", "text": user_input_prompt}]

        messages: list[dict[str, Any]] = []
        if system_context_prompt:
            messages.append({"role": "system", "content": system_context_prompt})
        messages.append({"role": "user", "content": content})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking,
        )
        assert isinstance(prompt, str)
        return prompt

    def _verify_ckpt(self) -> None:
        """Verify checkpoint is compatible with Qwen3-VL."""
        valid_model_types = {"qwen3_vl", "qwen3", "qwen3_vl_text"}
        architectures = getattr(self.llm_config, "architectures", None) or []
        if not (
            self.llm_config.model_type in valid_model_types
            or any("Qwen3" in arch for arch in architectures)
        ):
            raise ValueError(
                "Model config is not compatible with Qwen3-VL implementation. "
                f"Expected model_type in {valid_model_types}, got '{self.llm_config.model_type}'"
            )

    @staticmethod
    def monkey_patch(skip_optimizations: list[str] | None = None) -> None:
        """
        Apply monkey patches for Qwen3-VL ONNX export.

        Adaptations applied:
        - SHA (Split-Head Attention) for Qwen3VLTextAttention
        - MLP Conv2d (down_proj)
        - Bypass rotary embeddings (cos/sin pre-computed externally)
        - Export-friendly _deepstack_process (avoids boolean indexing)
        """
        from qai_hub_models.models._shared.qwen3.model import Qwen3_Optimizations
        from qai_hub_models.models._shared.qwen3_vl.model_adaptations import (
            QCQwen3VLTextMLP,
            SHAQwen3VLTextAttention,
        )

        # SHA attention
        if (
            skip_optimizations
            and Qwen3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_qwen3_vl.Qwen3VLTextAttention = SHAQwen3VLTextAttention  # type: ignore[misc, unused-ignore]

        # Bypass rotary embedding module
        def bypass_RotaryEmbedding(
            self: torch.nn.Module,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        if not hasattr(
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding, "_original_forward"
        ):
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward
            )
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward = (
                bypass_RotaryEmbedding  # type: ignore[assignment, unused-ignore]
            )

        # Patch Qwen3VLTextModel.forward to handle tuple position_ids
        # (pre-computed cos/sin from bypass_RotaryEmbedding) and to use
        # an export-friendly _deepstack_process.
        _original_text_forward = modeling_qwen3_vl.Qwen3VLTextModel.forward

        def _exportable_deepstack_process(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            """Export-friendly deepstack: avoids boolean indexing (dynamic shapes)."""
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            # Use float mask broadcasting instead of boolean indexing
            mask_float = visual_pos_masks.unsqueeze(-1).float()
            # Scatter visual_embeds into the positions marked by the mask.
            # cumsum gives a 1-based index into visual_embeds for each True position.
            indices = (visual_pos_masks.long().cumsum(dim=-1) - 1).clamp(min=0)
            indices = indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            gathered = torch.gather(
                visual_embeds.unsqueeze(0).expand(hidden_states.shape[0], -1, -1),
                dim=1,
                index=indices,
            )
            return hidden_states + gathered * mask_float

        def _patched_text_forward(
            self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: Any = None,
            visual_pos_masks: Any = None,
            deepstack_visual_embeds: Any = None,
            **kwargs: Any,
        ) -> Any:
            if isinstance(position_ids, tuple):
                # Pre-computed (cos, sin) — skip HF's ndim processing.
                from transformers.modeling_outputs import BaseModelOutputWithPast

                use_cache = (
                    use_cache if use_cache is not None else self.config.use_cache
                )

                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

                # position_embeddings = (cos, sin) from bypass_RotaryEmbedding
                position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

                hidden_states = inputs_embeds

                for layer_idx, decoder_layer in enumerate(self.layers):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=None,
                        past_key_values=past_key_values,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs

                    # Deepstack: inject visual embeddings at early layers
                    if (
                        deepstack_visual_embeds is not None
                        and visual_pos_masks is not None
                        and layer_idx < len(deepstack_visual_embeds)
                    ):
                        hidden_states = _exportable_deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_idx],
                        )

                hidden_states = self.norm(hidden_states)

                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                )

            return _original_text_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if not hasattr(modeling_qwen3_vl.Qwen3VLTextModel, "_original_forward"):
            modeling_qwen3_vl.Qwen3VLTextModel._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                _original_text_forward
            )
            modeling_qwen3_vl.Qwen3VLTextModel.forward = _patched_text_forward

        # MLP Conv2d adaptation
        modeling_qwen3_vl.Qwen3VLTextMLP = QCQwen3VLTextMLP  # type: ignore[misc, unused-ignore]


class Qwen3VLTextBase_AIMETOnnx(Qwen3Base_AIMETOnnx):
    """
    AIMET-ONNX quantized version of Qwen3-VL text model.

    Uses inputs_embeds instead of input_ids.
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen3VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen3VLTextBase.get_input_prompt_with_tags
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

        # Full VLM config (needed by the generator for vision-related fields)
        hf_repo = getattr(self, "_hf_repo_name", None) or (
            str(checkpoint) if checkpoint else None
        )
        assert hf_repo is not None
        self._original_llm_config = get_vlm_config(hf_repo)

        # Load embedding weights from checkpoint for VLM models.
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
        """Convert input token IDs to embeddings using the stored embedding table."""
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Adapt AIMET encodings for VLM model.

        VLM models use inputs_embeds instead of input_ids, so the embedding
        layer (embed_tokens) is not part of the exported ONNX model.
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

        # Copy weight encodings to param encodings
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
        if bundle.aimet_encodings_path is not None:
            encodings_path = str(bundle.aimet_encodings_path)
            self._adapt_aimet_encodings(
                encodings_path, encodings_path, str(bundle.onnx_graph_path)
            )
        return bundle


class Qwen3VLDynamic_AIMETOnnx(VLMDynamic_AIMETOnnx, Qwen3VLTextBase_AIMETOnnx):
    """Dynamic-shape variant of Qwen3VLTextBase_AIMETOnnx.

    Inherits the VLM calibration / weight-optimization data pipeline from
    VLMDynamic_AIMETOnnx; only model-specific config lives here.
    """

    FPModel = Qwen3VLTextBase

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()


class Qwen3VLTextBase_QNN(Qwen3Base_QNN):
    """QNN version of Qwen3-VL text model."""

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen3VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen3VLTextBase.get_input_prompt_with_tags
    )


# Re-export position processor
Qwen3VLPositionProcessor = Qwen3PositionProcessor
