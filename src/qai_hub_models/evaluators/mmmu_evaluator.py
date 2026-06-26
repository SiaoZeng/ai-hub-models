# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.evaluators.llm_evaluator import LLMEvaluator
from qai_hub_models.utils.base_evaluator import _DataLoader
from qai_hub_models.utils.metrics import (
    MMMU,
    MetricMetadata,
)

if TYPE_CHECKING:
    from transformers import GenerationMixin

# MMMU has variable-length options (up to ~9), so include A-I.
NUM_CHOICES = 9


class MMMUEvaluator(LLMEvaluator):
    """Evaluator for computing MMMU accuracy of a Vision-Language Model.

    Works like MMLUEvaluator but uses the MMMU metric and supports
    variable-length answer options.
    """

    # The generator returns full (seq_len, vocab) logits before we slice out
    # the answer columns; for a 7B VLM with long image-token sequences that
    # tensor is ~1 GB, so keep it off the GPU.
    accumulate_logits_on_cpu = True

    def __init__(
        self,
        context_length: int,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.context_length = context_length
        self.device = device
        self.tokenizer = tokenizer
        self.choices = self._get_choices(tokenizer)
        self.reset()

    @property
    def is_distance_metric(self) -> bool:
        return False

    @staticmethod
    def _get_choices(
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        def tokenize_letter(letter: str) -> torch.Tensor:
            return tokenizer(letter, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0, -1:]

        return torch.cat(
            [
                tokenize_letter(f"Answer: {chr(ord('A') + i)}")
                for i in range(NUM_CHOICES)
            ],
            dim=-1,
        )

    def add_batch(
        self,
        output: CausalLMOutputWithPast,
        gt: torch.Tensor | str,
    ) -> None:
        self.batch_index += 1
        assert output.logits is not None
        logits = output.logits[0]

        choices = self.choices.to(logits.device)
        answers = logits[:, choices]
        index = answers[-1].argmax()
        prediction = choices[index]

        top_token_id = logits[-1].argmax()
        self.top_is_valid += int(top_token_id in choices)

        # gt may be a (1,1) tensor with answer token ID (text-only samples)
        # or a string letter like "A"/"B"/"C"/"D" (multimodal samples).
        if isinstance(gt, torch.Tensor):
            gt_val = gt.item()
        elif isinstance(gt, str):
            letter_idx = ord(gt.strip().upper()) - ord("A")
            gt_val = self.choices[letter_idx].item()
        else:
            gt_val = gt
        correct = prediction.item() == gt_val
        self.correct_answers += int(correct)

    def reset(self) -> None:
        self.correct_answers = 0
        self.top_is_valid = 0
        self.batch_index = 0

    def get_accuracy_score(self) -> float:
        if self.batch_index == 0:
            return 0.0
        return self.correct_answers / self.batch_index

    def formatted_accuracy(self) -> str:
        return textwrap.dedent(
            f"""
                MMMU: {self.get_accuracy_score():.2%} (higher is better)
                Top prediction is valid answer: {self.top_is_valid / max(1, self.batch_index):.1%}
            """
        ).lstrip()

    def for_each_batch(
        self,
        generator: GenerationMixin,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: (
            Callable[[list[torch.Tensor], CausalLMOutputWithPast, torch.Tensor], None]
            | None
        ) = None,
    ) -> None:
        total_samples = 0
        batch_size = 1
        num_samples = num_samples or len(data)

        with tqdm(
            total=num_samples,
            desc="Number of samples completed",
        ) as pbar:
            for sample in data:
                # Unpack: (input_ids, attention_mask, ground_truth
                #          [, pixel_values[, image_grid_thw]])
                input_ids, attention_mask, ground_truth, *rest = sample  # type: ignore[misc]
                pixel_values = rest[0] if len(rest) > 0 else None
                image_grid_thw = rest[1] if len(rest) > 1 else None

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)

                inputs = [input_ids, attention_mask]
                outputs = generator(  # type: ignore[operator, unused-ignore]
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

                if callback:
                    callback(inputs, outputs, ground_truth)
                # Free KV cache and GPU tensors between samples
                del outputs, inputs
                torch.cuda.empty_cache()
                total_samples += 1
                pbar.update(batch_size)
                if total_samples >= num_samples:
                    break

    def get_metric_metadata(self) -> MetricMetadata:
        return MMMU
