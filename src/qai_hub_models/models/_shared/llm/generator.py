# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import transformers
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.models._shared.llm.common import LLMIOType, cleanup
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
    Embedding,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamic_AIMETOnnx,
    LLMDynamicBase,
)

if TYPE_CHECKING:
    from PIL import Image


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """Clip past key value to feed next iteration"""
    ret = []

    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            orig_key_shape = new_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            past_key_vals.append(torch.zeros(key_shape, device=device))

            orig_value_shape = new_key_vals[i + 1].shape
            value_shape = (
                orig_value_shape[0],
                orig_value_shape[1],
                0,
                orig_value_shape[3],
            )
            past_key_vals.append(torch.zeros(value_shape, device=device))

    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            orig_key_shape = past_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            new_key_vals.append(torch.zeros(key_shape, device=device))

            orig_value_shape = past_key_vals[i + 1].shape
            value_shape = (
                orig_value_shape[0],
                orig_value_shape[1],
                0,
                orig_value_shape[3],
            )
            new_key_vals.append(torch.zeros(value_shape, device=device))

    # Key and Values are concatenated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        key_cache = torch.cat(
            [past_key_vals[i].to(device), new_key_vals[i].to(device)],
            dim=3,
        )
        key_cache = key_cache[:, :, :, -length:]
        val_cache = torch.cat(
            [
                past_key_vals[i + 1].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )
        val_cache = val_cache[:, :, -length:, :]

        ret.append(key_cache)
        ret.append(val_cache)
    return ret


class LLM_Loader:
    def __init__(
        self,
        model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN],
        sequence_length: int,
        model_params: dict[str, Any],
        host_device: torch.device,
    ) -> None:
        self.model_cls = model_cls
        self.sequence_length = sequence_length
        self.model_params = model_params
        self.loaded_model: LLMBase | LLM_AIMETOnnx | LLM_QNN | None = None
        self.host_device = host_device

    def load(self) -> LLMBase | LLM_AIMETOnnx | LLM_QNN:
        if self.loaded_model is None:
            is_dynamic = issubclass(
                self.model_cls, (LLMDynamicBase, LLMDynamic_AIMETOnnx)
            )
            kwargs = dict(self.model_params)
            if not is_dynamic:
                kwargs["sequence_length"] = self.sequence_length
            else:
                kwargs.pop("context_length", None)
                kwargs.pop("sequence_length", None)
            self.loaded_model = self.model_cls.from_pretrained(**kwargs).to(
                self.host_device
            )

        assert self.loaded_model is not None
        return self.loaded_model

    def release(self) -> None:
        # Defer to the model's own release(): for AIMET models that means
        # tearing down quant_sim *and* evicting any class-level cache slot
        # (SingleSlotCacheMixin), not just nulling _quant_sim on the live
        # instance — otherwise the next from_pretrained() returns the broken
        # cached instance.
        if self.loaded_model is not None:
            self.loaded_model.release()
        self.loaded_model = None

    def __del__(self) -> None:
        self.release()
        # Python can be in a weird state when __del__ gets called, so we
        # have to make sure these still exist.
        if "gc" in globals() and gc is not None:
            gc.collect()
        if "torch" in globals() and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class LLM_Generator(GenerationMixin, torch.nn.Module):
    _is_stateful = False

    def __init__(
        self,
        models: list[LLMBase | LLM_AIMETOnnx | LLM_QNN | LLM_Loader],
        tokenizer: transformers.PreTrainedTokenizerBase,
        embedding: Embedding,
        accumulate_logits_on_cpu: bool = False,
        # VLM support
        vision_encoder: torch.nn.Module | None = None,
        hf_repo_name: str | None = None,  # for AutoProcessor/AutoConfig
    ) -> None:
        super().__init__()

        self.models = models
        self.models.sort(key=lambda model: model.sequence_length)

        self.selected_model = (
            self.models[-1].load()
            if isinstance(self.models[-1], LLM_Loader)
            else self.models[-1]
        )
        self.selected_sequence_length: int | None = self.models[-1].sequence_length

        self.tokenizer = tokenizer
        self.embedding = embedding
        self.accumulate_logits_on_cpu = accumulate_logits_on_cpu

        # VLM support
        self.vision_encoder = vision_encoder
        self.hf_repo_name = hf_repo_name
        self._vision_processor = None  # Lazy-loaded

    def release(self) -> None:
        # Tear down every model we own. release() on an AIMET model also
        # evicts it from any class-level cache (SingleSlotCacheMixin); the
        # ORT InferenceSession behind quant_sim holds a CUDA arena outside
        # PyTorch's allocator, so its destructor must run to free that
        # memory before the next from_pretrained() in the same process.
        for model in self.models:
            if hasattr(model, "release"):
                model.release()

        if hasattr(self.selected_model, "release"):
            self.selected_model.release()

        # Clean up VLM components
        if self.vision_encoder is not None:
            del self.vision_encoder
            self.vision_encoder = None
        self._vision_processor = None
        cleanup()

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.selected_model.llm_config

    @property
    def main_input_name(self) -> str:
        # Always report "input_ids" to HuggingFace's generate().
        # HF's _prepare_model_inputs detects inputs_embeds in kwargs
        # and promotes it for the first forward pass automatically.
        return "input_ids"

    @property
    def llm_io_type(self) -> LLMIOType:
        assert self.selected_model is not None
        return self.selected_model.llm_io_type

    @property
    def _supports_cache_class(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        host_device = getattr(self.selected_model, "host_device", None)
        if host_device is not None:
            return host_device

        # Note: torch.nn.Module.device does not exist according to PyTorch
        # documentation and mypy.
        return next(iter(self.selected_model.parameters())).device

    @property
    def is_vlm(self) -> bool:
        """Check if this generator supports vision-language models."""
        return self.vision_encoder is not None

    @property
    def vision_processor(self) -> transformers.ProcessorMixin:
        """Lazy-load the processor for VLM input processing."""
        if self._vision_processor is None:
            if self.hf_repo_name is None:
                raise ValueError("hf_repo_name required for VLM processor")
            from transformers import AutoProcessor

            self._vision_processor = AutoProcessor.from_pretrained(
                self.hf_repo_name, trust_remote_code=True
            )
        return self._vision_processor  # type: ignore[return-value, unused-ignore]

    def prepare_vlm_inputs(
        self,
        input_prompt: str,
        image: Image.Image | str | Path | list[Image.Image | str | Path],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Prepare VLM inputs by processing image(s) and merging embeddings.

        This handles:
        1. Image preprocessing via AutoProcessor
        2. Vision encoder execution to get vision embeddings
        3. Text tokenization
        4. Embedding merge (replacing image tokens with vision embeddings)

        Parameters
        ----------
        input_prompt
            The text prompt to send to the model.
        image
            One or more images, each as a PIL Image or path to image file.

        Returns
        -------
        merged_embeddings : torch.Tensor
            Tensor of merged text and vision embeddings.
        input_tokens_dict : dict[str, torch.Tensor]
            Dictionary containing input_ids and attention_mask.
        """
        from PIL import Image as PILImage
        from transformers import AutoConfig

        if self.vision_encoder is None:
            raise ValueError("Vision encoder not set. Cannot prepare VLM inputs.")
        if self.hf_repo_name is None:
            raise ValueError("hf_repo_name required for VLM")

        device = self.device

        # Normalise to a list of PIL images
        if not isinstance(image, list):
            image = [image]
        images: list[PILImage.Image] = []
        for img in image:
            if isinstance(img, (str, Path)):
                img = PILImage.open(img).convert("RGB")
            images.append(img)

        # Resize every image to match the vision encoder's expected dimensions
        if hasattr(self.vision_encoder, "_image_height"):
            expected_h = int(self.vision_encoder._image_height)  # type: ignore[arg-type, unused-ignore]
            expected_w = int(self.vision_encoder._image_width)  # type: ignore[arg-type, unused-ignore]
            images = [
                img.resize((expected_w, expected_h))
                if img.size != (expected_w, expected_h)
                else img
                for img in images
            ]

        # Use the model's get_input_prompt_with_tags for consistent prompt formatting
        # Pass the number of images so the right number of placeholders are inserted
        formatted_text = self.selected_model.get_input_prompt_with_tags(
            user_input_prompt=input_prompt,
            include_image=len(images),  # type: ignore[arg-type]
        )

        # Process inputs - processor expands vision placeholders to match image tokens
        processed = self.vision_processor(  # type: ignore[operator, unused-ignore]
            text=[formatted_text],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]
        pixel_values = processed["pixel_values"]

        # Run vision encoder, one image at a time if needed.
        # The VEG may have fixed input shapes for a single image, so we
        # split multi-image pixel_values into per-image chunks.
        self.vision_encoder.eval()
        with torch.no_grad():
            veg = self.vision_encoder
            patch_size = veg._patch_size
            img_h = veg._image_height
            img_w = veg._image_width
            single_seq_len = (img_h // patch_size) * (img_w // patch_size)  # type: ignore[operator, unused-ignore]
            total_patches = pixel_values.shape[0]

            if total_patches == single_seq_len or len(images) == 1:
                vision_embeddings = veg(pixel_values=pixel_values)
            elif total_patches % single_seq_len == 0:
                chunks = pixel_values.split(single_seq_len, dim=0)
                vision_embeddings = torch.cat(
                    [veg(pixel_values=c) for c in chunks], dim=0
                )
            else:
                # Dynamic-shape VEG or unexpected layout — try full tensor
                vision_embeddings = veg(pixel_values=pixel_values)

        # Get image token ID from config
        config = AutoConfig.from_pretrained(self.hf_repo_name, trust_remote_code=True)
        image_token_id = config.image_token_id

        # Convert input_ids to text embeddings using the model's embedding table
        text_embeddings = self.selected_model.convert_input_ids_to_embeddings(input_ids)

        # Find image token positions and merge embeddings
        image_mask = input_ids == image_token_id  # (batch, seq_len)

        # Count image tokens - should match vision embedding tokens
        num_image_tokens = image_mask.sum().item()
        num_vision_tokens = vision_embeddings.shape[0]

        if num_image_tokens != num_vision_tokens:
            print(
                f"Warning: Image token count ({num_image_tokens}) != "
                f"vision embedding count ({num_vision_tokens})"
            )

        # Merge: replace image token positions with vision embeddings
        # vision_embeddings shape: (num_tokens, hidden_size)
        # text_embeddings shape: (batch, seq_len, hidden_size)
        merged_embeddings = text_embeddings.clone()

        # Expand mask for hidden dimension
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(text_embeddings)

        # Use masked_scatter to replace image token embeddings with vision embeddings
        merged_embeddings = merged_embeddings.masked_scatter(
            image_mask_expanded,
            vision_embeddings.to(
                device=merged_embeddings.device, dtype=merged_embeddings.dtype
            ),
        )

        # Free vision encoder to reclaim GPU memory before text model runs
        del self.vision_encoder
        self.vision_encoder = None
        self._vision_processor = None
        gc.collect()
        torch.cuda.empty_cache()

        return merged_embeddings, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | DynamicCache | None]:
        """
        Prepare inputs for one generation step.

        HuggingFace's generate() calls this before each forward().
        Static-shape padding/truncation happens in forward(), not here.

        For VLM: HF keeps inputs_embeds in model_kwargs across all
        iterations. On the first call (no KV cache), we use them.
        On subsequent calls, the KV cache has more entries than
        input_ids (which only tracks generated tokens), so we use
        the last token from input_ids.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You must specify at least one of input_ids or inputs_embeds"
            )

        if past_key_values is None:
            num_processed_tokens = 0
        elif hasattr(past_key_values, "value_cache"):
            num_processed_tokens = (
                0
                if len(past_key_values.value_cache) == 0
                or past_key_values.value_cache[0] == []
                else past_key_values.value_cache[0].shape[-2]
            )
        elif past_key_values.layers and hasattr(past_key_values.layers[0], "values"):  # type: ignore[attr-defined, unused-ignore]
            num_processed_tokens = (
                0
                if past_key_values.layers[0].values is None  # type: ignore[attr-defined, unused-ignore]
                else past_key_values.layers[0].values.shape[-2]  # type: ignore[attr-defined, unused-ignore]
            )
        else:
            raise ValueError("Unsupported KV cache type")

        inputs: dict[str, torch.Tensor | DynamicCache | None] = {}
        if inputs_embeds is not None and num_processed_tokens < inputs_embeds.shape[1]:
            inputs = {"inputs_embeds": inputs_embeds[:, num_processed_tokens:, :]}
        elif input_ids is not None and num_processed_tokens < input_ids.shape[1]:
            inputs = {"input_ids": input_ids[:, num_processed_tokens:]}
        elif input_ids is not None:
            # Decode after VLM prefill: KV cache reflects the full
            # embeddings length, but input_ids only has generated tokens.
            inputs = {"input_ids": input_ids[:, -1:]}
        else:
            inputs = {"inputs_embeds": inputs_embeds[:, num_processed_tokens:, :]}  # type: ignore[index]

        return inputs | {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def select_model(self, num_input_tokens: int) -> LLM_AIMETOnnx | LLM_QNN | LLMBase:
        # Select the model with the smallest sequence length that can fit all of num_input_tokens
        # If there is no model that can consume num_input_tokens in one inference, select the model with the largest
        # sequence length
        new_selected_model = self.models[
            -1
        ]  # start off by selecting model with largest sequence length
        for model in self.models:
            if (
                num_input_tokens <= model.sequence_length
                and model.sequence_length < new_selected_model.sequence_length
            ):
                new_selected_model = model  # if there is any model with a smaller sequence length that works, select it

        if self.selected_sequence_length == new_selected_model.sequence_length:
            return self.selected_model

        print(
            f"Switching from sequence_length={self.selected_sequence_length} to sequence_length={new_selected_model.sequence_length}"
        )
        # release the model to preserve memory
        if isinstance(
            self.selected_model,
            (LLM_Loader, LLM_AIMETOnnx, LLM_QNN),
        ):
            self.selected_model.release()

        self.selected_model = (
            new_selected_model.load()
            if isinstance(new_selected_model, LLM_Loader)
            else new_selected_model
        )
        self.selected_sequence_length = new_selected_model.sequence_length
        return self.selected_model

    @staticmethod
    def slice_inputs_for_inference(
        inputs: torch.Tensor, attention_mask: torch.Tensor, sequence_length: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            yield (
                inputs[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
            )

    def prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # If primary method of accepting inputs is inputs_embeds, but input_ids are provided (ie - in generation)
        # then convert tokens to embeddings
        if self.llm_io_type == LLMIOType.genie_input_embeds and input_ids is not None:
            inputs_embeds = cast(
                torch.FloatTensor,
                self.selected_model.convert_input_ids_to_embeddings(input_ids),
            )
            input_ids = None

        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)
        input_tokens = input_tokens.to(
            dtype=torch.int32 if input_ids is not None else torch.float32
        )

        device = input_tokens.device
        batch_size = input_tokens.shape[0]
        input_length = input_tokens.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        if input_ids is not None:
            input_tokens_extension = torch.full(
                (batch_size, sequence_length - input_length),
                fill_value=getattr(self.tokenizer, "eos_token_id", 0),
                dtype=input_tokens.dtype,
                device=device,
            )
        else:
            embedding_dim = input_tokens.shape[2]
            input_tokens_extension = torch.zeros(
                (batch_size, sequence_length - input_length, embedding_dim),
                dtype=input_tokens.dtype,
                device=device,
            )

        padded_input_tokens = torch.cat((input_tokens_extension, input_tokens), dim=1)
        attention_mask_extension = torch.zeros(
            (batch_size, sequence_length - input_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded_attention_mask = torch.cat(
            (torch.zeros_like(attention_mask_extension), attention_mask), dim=-1
        )

        input_specs = self.selected_model.get_input_spec(
            llm_config=self.selected_model.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
        )
        # Initialization of KV cache padding
        dummy_past_key_values = [
            torch.zeros(shape, device=device)
            for k, (shape, _) in input_specs.items()
            if k.startswith("past_")
        ]

        current_key_value_length = (
            past_key_values[1].shape[-2] if past_key_values else 0
        )
        key_value_padding_length = (
            context_length - sequence_length
        ) - current_key_value_length

        padded_past_key_values = get_past_keyval_with_shift(
            past_key_vals=dummy_past_key_values,
            new_key_vals=past_key_values,
            length=context_length - sequence_length,
            device=device,
        )

        kv_cache_attention_mask = torch.cat(
            (
                torch.zeros((batch_size, key_value_padding_length)),
                torch.ones((batch_size, current_key_value_length)),
            ),
            dim=-1,
        ).to(device=device)
        padded_attention_mask = torch.cat(
            (kv_cache_attention_mask, padded_attention_mask), dim=-1
        )

        position_ids = torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
        position_ids = position_ids.clip(0, context_length - 1)
        position_ids = position_ids[..., -sequence_length:]

        attention_mask_converter = AttentionMaskConverter(True)
        cm_attention_mask = attention_mask_converter.to_4d(
            padded_attention_mask,
            query_length=sequence_length,
            key_value_length=context_length,
            dtype=torch.float32,
        )
        attention_mask_min_clip = getattr(
            self.selected_model, "attention_mask_min_clip", None
        )
        if attention_mask_min_clip is not None:
            cm_attention_mask = cm_attention_mask.clip(min=attention_mask_min_clip)

        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            return (
                padded_input_tokens,
                cm_attention_mask,
                position_ids,
                *padded_past_key_values,
            )
        position_ids_cos, position_ids_sin = self.embedding.get_embedding(position_ids)
        return (
            padded_input_tokens,
            cm_attention_mask,
            position_ids_cos,
            position_ids_sin,
            *padded_past_key_values,
        )

    def combine_local_and_global_outputs(
        self,
        model: LLMBase | LLM_AIMETOnnx | LLM_QNN,
        num_valid_input_tokens: int,
        local_outputs: tuple[torch.Tensor, ...],
        global_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        device = local_outputs[0].device
        logits_device = "cpu" if self.accumulate_logits_on_cpu else device

        # strip logits corresponding to padding tokens
        local_logits = local_outputs[0]
        local_logits = torch.narrow(
            local_logits,
            1,
            local_logits.shape[1] - num_valid_input_tokens,
            num_valid_input_tokens,
        ).to(logits_device)

        # concatenate logits from local inference to global output
        if "logits" in global_outputs:
            assert isinstance(global_outputs["logits"], torch.Tensor)
            global_outputs["logits"] = torch.cat(
                [global_outputs["logits"], local_logits], dim=1
            )
        else:
            global_outputs["logits"] = local_logits

        # strip KV cache corresponding to padding tokens
        local_past_key_values = get_past_keyval_with_shift(
            past_key_vals=[],
            new_key_vals=list(local_outputs[1:]),
            length=num_valid_input_tokens,
            device=device,
        )

        past_key_values_list = global_outputs["past_key_values"]
        assert isinstance(past_key_values_list, list)

        # shift global KV cache, concatenate local KV cache
        current_key_value_length = (
            past_key_values_list[1].shape[-2] if past_key_values_list else 0
        )
        global_outputs["past_key_values"] = get_past_keyval_with_shift(
            past_key_vals=past_key_values_list,
            new_key_vals=local_past_key_values,
            length=min(
                current_key_value_length + num_valid_input_tokens,
                model.context_length - model.sequence_length,
            ),
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)

        # Select which model to use
        model = self.select_model(input_tokens.shape[1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        global_outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "past_key_values": (
                []
                if past_key_values is None or past_key_values.get_seq_length() == 0
                else list(
                    itertools.chain.from_iterable(past_key_values.to_legacy_cache())
                )
            )
        }

        for input_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_tokens, attention_mask, model.sequence_length
        ):
            past_key_values_list = global_outputs["past_key_values"]
            assert isinstance(past_key_values_list, list)

            prepared_inputs = self.prepare_inputs(
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=past_key_values_list,
                sequence_length=model.sequence_length,
                context_length=model.context_length,
                inputs_embeds=cast(torch.FloatTensor, input_slice)
                if inputs_embeds is not None
                else None,
            )

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model,
                input_slice.shape[1],
                local_outputs,
                global_outputs,
            )

        # make sure logits are on the correct device (necessary for generation)
        # the underlying mock_torch_onnx_inference function does not necessarily move outputs back to CUDA
        assert isinstance(global_outputs["logits"], torch.Tensor)
        logits = global_outputs["logits"].to(
            device="cpu" if self.accumulate_logits_on_cpu else input_tokens.device
        )

        # Convert KV Cache outputs into HF DynamicCache
        past_key_values = DynamicCache()
        for layer_idx in range(len(global_outputs["past_key_values"]) // 2):
            past_key_values.update(
                global_outputs["past_key_values"][layer_idx * 2],
                global_outputs["past_key_values"][layer_idx * 2 + 1],
                layer_idx,
            )
        return CausalLMOutputWithPast(
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,  # type: ignore[arg-type, unused-ignore]
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        if len(self.models) > 1:
            raise RuntimeError("Prefill should only be invoked using a single model")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)

        # Select which model to use
        model = self.select_model(input_tokens.shape[1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        # slice input ids and attention mask to drop last few tokens
        total_num_inferences = math.ceil(input_tokens.shape[1] / model.sequence_length)
        num_tokens_to_preconsume = (total_num_inferences - 1) * model.sequence_length

        input_tokens_to_preconsume = input_tokens[:, :num_tokens_to_preconsume]
        attention_mask_to_preconsume = attention_mask[:, :num_tokens_to_preconsume]

        preconsumed_outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "past_key_values": (
                []
                if past_key_values is None or past_key_values.get_seq_length() == 0
                else list(
                    itertools.chain.from_iterable(past_key_values.to_legacy_cache())
                )
            )
        }

        for input_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_tokens_to_preconsume,
            attention_mask_to_preconsume,
            model.sequence_length,
        ):
            past_key_values_list = preconsumed_outputs["past_key_values"]
            assert isinstance(past_key_values_list, list)

            prepared_inputs = self.prepare_inputs(
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=past_key_values_list,
                sequence_length=model.sequence_length,
                context_length=model.context_length,
                inputs_embeds=cast(torch.FloatTensor, input_slice)
                if inputs_embeds is not None
                else None,
            )

            yield tuple(tensor.cpu() for tensor in prepared_inputs)

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model,
                input_slice.shape[1],
                local_outputs,
                preconsumed_outputs,
            )

        remaining_input_tokens = input_tokens[:, num_tokens_to_preconsume:]
        remaining_attention_mask = attention_mask[:, num_tokens_to_preconsume:]
        past_key_values_list = preconsumed_outputs["past_key_values"]
        assert isinstance(past_key_values_list, list)
        prefilled_inputs = self.prepare_inputs(
            input_ids=remaining_input_tokens if input_ids is not None else None,
            attention_mask=remaining_attention_mask,
            past_key_values=past_key_values_list,
            sequence_length=model.sequence_length,
            context_length=model.context_length,
            inputs_embeds=cast(torch.FloatTensor, remaining_input_tokens)
            if inputs_embeds is not None
            else None,
        )

        yield tuple(tensor.cpu() for tensor in prefilled_inputs)
