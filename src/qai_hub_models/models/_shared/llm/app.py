# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from qai_hub_models.utils.base_model import BaseModel
from transformers import GenerationConfig, TextStreamer, set_seed

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    get_llm_config,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec


class IndentedTextStreamer(TextStreamer):
    def __init__(self, line_start: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_width = shutil.get_terminal_size().columns
        self.printed_width = 0
        self.line_start = line_start

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if len(text) == 0:
            return

        # If the incoming text would cause the printed output to wrap around, start a new line
        if self.printed_width + len(text) >= self.terminal_width:
            print(flush=True)
            self.printed_width = 0

        # If we are on a new line, print the line starter before the text
        if self.printed_width == 0:
            text = self.line_start + text

        # If there are multiple newlines, make sure that the line starter is present at every new line
        # (except the last one, since that will be taken care of when we try to print the something to that new line
        # for the first time)
        if text.count("\n") > 1:
            last_index = text.rfind("\n")
            before_last = text[:last_index]
            after_last = text[last_index:]
            modified_before_last = before_last.replace("\n", "\n" + self.line_start)
            text = modified_before_last + after_last

        print(text, flush=True, end="" if not stream_end else None)

        # Update the counter of characters on this line
        if text.endswith("\n"):
            self.printed_width = 0
        else:
            self.printed_width += len(text)


class ChatApp:
    """
    This class is a demonstration of how to use Llama model to build a basic ChatApp.
    This App uses two models:
        * Prompt Processor
            - Instantiation with sequence length 128. Used to process user
              prompt.
        * Token Generator
            - Instantiation with sequence length 1. Used to predict
              auto-regressive response.
    """

    def __init__(
        self,
        model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN],
        get_input_prompt_with_tags: Callable,
        tokenizer: Any,
        end_tokens: set[str],
        seed: int = 42,
        # VLM parameters
        vision_encoder_cls: type[BaseModel] | None = None,
        hf_repo_name: str | None = None,
        hidden_size: int | None = None,
        vlm_image_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Base ChatApp that generates one response for given input token.

            model_cls: Model class that will be used to instantiate model
            get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags
            tokenizer: Tokenizer to use
            end_tokens: Set of end tokens to convey end of token generation
            seed: Random seed
            vision_encoder_cls: Vision encoder class for VLM (optional)
            hf_repo_name: HF repo name, used to load AutoProcessor/AutoConfig for VLM image preprocessing
            hidden_size: Hidden size for VLM embedding table (optional)
            vlm_image_size: (width, height) override for VEG image size (optional)
        """
        self.model_cls = model_cls
        self.get_input_prompt_with_tags = get_input_prompt_with_tags
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self.seed = seed
        # VLM parameters
        self.vision_encoder_cls = vision_encoder_cls
        self.hf_repo_name = hf_repo_name
        self.hidden_size = hidden_size
        self.vlm_image_size = vlm_image_size

    def generate_output_prompt(
        self,
        input_prompt: str,
        context_length: int,
        max_output_tokens: int,
        checkpoint: CheckpointSpec | None = None,
        model_from_pretrained_extra: dict | None = None,
        image_path: str | Path | list[str] | None = None,
        sequence_length: int | None = None,
    ) -> None:
        from qai_hub_models.models._shared.llm.generator import (
            LLM_Generator,
            LLM_Loader,
        )

        if model_from_pretrained_extra is None:
            model_from_pretrained_extra = {}
        set_seed(self.seed)

        host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_params = {
            "context_length": context_length,
            "host_device": host_device,
            **model_from_pretrained_extra,
        }
        checkpoint_for_model = (
            None if checkpoint == "DEFAULT_UNQUANTIZED" else checkpoint
        )
        if checkpoint_for_model is not None:
            model_params["checkpoint"] = checkpoint_for_model

        if sequence_length is not None:
            export_seq_lengths = [sequence_length, 1]
        else:
            export_seq_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS
        models: list[LLM_Loader | LLMBase | LLM_AIMETOnnx | LLM_QNN] = [
            LLM_Loader(self.model_cls, sl, model_params, host_device)
            for sl in export_seq_lengths
        ]
        if "fp_model" in model_from_pretrained_extra:
            config = model_from_pretrained_extra["fp_model"].llm_config
        elif checkpoint_for_model is not None:
            config = get_llm_config(checkpoint_for_model)
        else:
            model = models[-1]
            # This only works for loaders
            assert isinstance(model, LLM_Loader)
            # This is expensive, so only done as last resort
            config = model.load().llm_config
            model.release()

        # Load vision encoder for VLM
        vision_encoder = None
        if self.vision_encoder_cls is not None:
            veg_kwargs: dict[str, Any] = {}
            if self.vlm_image_size is not None:
                veg_kwargs["image_width"] = self.vlm_image_size[0]
                veg_kwargs["image_height"] = self.vlm_image_size[1]

            if checkpoint_for_model is None:
                # FP mode — no checkpoint, load from HuggingFace
                veg_kwargs["precision"] = Precision.float
            else:
                # Quantized mode — use checkpoint path; precision defaults
                # to the Component class's DEFAULT_PRECISION
                veg_kwargs["checkpoint"] = checkpoint_for_model

            vision_encoder = self.vision_encoder_cls.from_pretrained(
                device=host_device,
                **veg_kwargs,
            )

        # TODO: Use instance in model already?
        assert hasattr(self.model_cls, "EmbeddingClass")
        assert self.model_cls.EmbeddingClass is not None
        rope_embedding = self.model_cls.EmbeddingClass(
            max_length=context_length, config=config
        )
        inferencer = LLM_Generator(
            models,
            self.tokenizer,
            rope_embedding,
            vision_encoder=vision_encoder,
            hf_repo_name=self.hf_repo_name,
        )

        # can set temperature, topK, topP, etc here
        end_token_ids = []
        for token in self.end_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                token_id = token_ids[0]
                end_token_ids.append(token_id)
        end_token_ids.append(self.tokenizer.eos_token_id)
        inferencer.generation_config = GenerationConfig(
            max_new_tokens=max_output_tokens,
            eos_token_id=end_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.8,
        )

        # For VLM, skip_prompt=True because the "prompt" consists of placeholder
        # input_ids (all pad_token_id) that decode to gibberish. The actual prompt
        # text is encoded as inputs_embeds, not input_ids.
        skip_prompt = inferencer.is_vlm
        streamer = IndentedTextStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=skip_prompt,
            line_start="    + ",
        )

        if inferencer.is_vlm and image_path is not None:
            # VLM with image — run vision encoder and merge embeddings
            inputs_embeds, input_tokens = inferencer.prepare_vlm_inputs(
                input_prompt=input_prompt,
                image=image_path,  # type: ignore[arg-type]
            )
            inferencer.generate(  # type: ignore[operator, misc, unused-ignore]
                inputs_embeds=inputs_embeds,
                attention_mask=input_tokens["attention_mask"],
                generation_config=inferencer.generation_config,
                streamer=streamer,
            )
        else:
            # LLM or VLM text-only — tokenize and pass input_ids
            prompt_kwargs: dict[str, Any] = dict(user_input_prompt=input_prompt)
            if inferencer.is_vlm:
                prompt_kwargs["include_image"] = False
            input_prompt_processed = self.get_input_prompt_with_tags(
                **prompt_kwargs,
            )
            input_tokens = self.tokenizer(
                input_prompt_processed,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(host_device)
            inferencer.generate(  # type: ignore[operator, misc, unused-ignore]
                inputs=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                generation_config=inferencer.generation_config,
                streamer=streamer,
            )
        inferencer.release()
