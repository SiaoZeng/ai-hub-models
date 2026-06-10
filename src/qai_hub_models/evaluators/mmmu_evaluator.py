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
from transformers import AutoConfig, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.evaluators.llm_evaluator import LLMEvaluator
from qai_hub_models.evaluators.metrics import MMMU, MetricMetadata
from qai_hub_models.utils.base_evaluator import _DataLoader

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.generator import LLM_Generator

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

        answers = logits[:, self.choices]
        index = answers[-1].argmax()
        prediction = self.choices[index]

        top_token_id = logits[-1].argmax()
        self.top_is_valid += int(top_token_id in self.choices)

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
        generator: LLM_Generator,
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
        # NOTE: Do not wrap the forward pass in torch.autocast("cuda"). Its
        # default dtype is float16, and Qwen2.5-VL activations exceed fp16's
        # range, which corrupts the logits (the top token is frequently not
        # even an answer letter). The FP weights are already float32, so
        # autocast saves little memory but tanks accuracy. Run in full fp32.

        # Resolve image_token_id and the VEG device once; neither the vision
        # encoder nor its device changes across samples.
        image_token_id = None
        if generator.is_vlm and generator.hf_repo_name is not None:
            config = AutoConfig.from_pretrained(
                generator.hf_repo_name, trust_remote_code=True
            )
            image_token_id = config.image_token_id

        # Send pixel_values to the VEG's device (may differ from the evaluator
        # device for quantized LLM + FP VEG).
        veg_device = self.device
        if generator.vision_encoder is not None and hasattr(
            generator.vision_encoder, "parameters"
        ):
            veg_device = next(generator.vision_encoder.parameters()).device

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

                # For VLM samples with images, run the generator's VEG
                # and merge vision + text embeddings, then feed inputs_embeds.
                if (
                    pixel_values is not None
                    and generator.vision_encoder is not None
                    and image_token_id is not None
                ):
                    inputs_embeds = self._prepare_vlm_inputs(
                        generator,
                        image_token_id,
                        input_ids.to(veg_device),
                        pixel_values.to(veg_device),
                        image_grid_thw=image_grid_thw,
                    ).to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    with torch.no_grad():
                        outputs = generator(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                        )
                    inputs = [inputs_embeds, attention_mask]
                else:
                    inputs = [input_ids, attention_mask]
                    inputs = [inp.to(self.device) for inp in inputs]
                    with torch.no_grad():
                        outputs = generator(*inputs)

                if callback:
                    callback(inputs, outputs, ground_truth)
                # Free KV cache and GPU tensors between samples
                del outputs, inputs
                torch.cuda.empty_cache()
                total_samples += 1
                pbar.update(batch_size)
                if total_samples >= num_samples:
                    break

    @staticmethod
    def _prepare_vlm_inputs(
        generator: LLM_Generator,
        image_token_id: int,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the generator's VEG and merge with text embeddings.

        Uses the same VEG (quantized or FP) that was loaded in evaluate.py,
        so MMMU evaluation reflects the actual VEG quality.
        """
        veg = generator.vision_encoder
        assert veg is not None

        # The VEG bakes its RoPE / attention-mask / window buffers in __init__
        # for one fixed grid, so it only accepts a single image whose patch
        # count equals veg.seq_len. This holds because vlm_image_size is
        # mandatory and the dataset resizes every image to exactly it; we call
        # the VEG once per image (no grid_thw argument) rather than batching.
        # The per-image assert below turns a future image-size change into a
        # clear error instead of a confusing buffer/shape mismatch deeper in.
        with torch.no_grad():
            if image_grid_thw is not None:
                # Use image_grid_thw to split pixel_values per image.
                # Each row is (t, h, w); t*h*w = number of patches for that image.
                per_image_patches = [
                    int(thw[0] * thw[1] * thw[2]) for thw in image_grid_thw
                ]
                assert all(p == veg.seq_len for p in per_image_patches), (
                    f"VEG expects {veg.seq_len} patches per image (fixed grid), "
                    f"got {per_image_patches}; every image must be resized to "
                    "vlm_image_size."
                )
                chunks = pixel_values.split(per_image_patches, dim=0)
                vision_embeddings = torch.cat(
                    [veg(pixel_values=c) for c in chunks], dim=0
                )
            else:
                # Fallback: assume all patches belong to a single image
                vision_embeddings = veg(pixel_values=pixel_values)

        # Convert input_ids to text embeddings
        text_embeddings = generator.selected_model.convert_input_ids_to_embeddings(
            input_ids
        )

        # Merge: replace image token positions with vision embeddings
        image_mask = input_ids == image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(text_embeddings)
        merged = text_embeddings.clone()
        return merged.masked_scatter(
            image_mask_expanded,
            vision_embeddings.to(merged.dtype),
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return MMMU
