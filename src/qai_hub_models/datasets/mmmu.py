# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import ast
import re
from typing import Any

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit


class MMMU(BaseDataset):
    """Multimodal Multitask Understanding (MMMU) dataset.

    A benchmark for evaluating vision-language models on multimodal
    multiple-choice questions spanning diverse academic subjects.

    Each sample contains one or more images, a question, and variable-length
    answer options (A, B, C, D, ...). Only multiple-choice questions are used.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.VAL,
        num_samples: int = 0,
        seed: int = 42,
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.processor = processor
        self.image_size = image_size

        if split in (DatasetSplit.VAL, DatasetSplit.TEST):
            # MMMU test split has no answers; always use validation for eval
            split_str = "validation"
        else:
            raise ValueError("MMMU dataset supports `val` or `test` splits")

        # Load all subject configs and filter to multiple-choice only
        configs = [c for c in get_dataset_config_names("MMMU/MMMU") if c != "default"]

        all_ds = [
            load_dataset("MMMU/MMMU", config, split=split_str) for config in configs
        ]
        combined = concatenate_datasets(all_ds)
        self.dataset = combined.filter(
            lambda x: x["question_type"] == "multiple-choice"
        )
        self.dataset = self.dataset.shuffle(seed)

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> tuple[Any, ...]:
        item = batch[0]
        result: tuple[Any, ...] = (
            item["input_ids"],
            item["attention_mask"],
            item["label"],
        )
        # For VLM multimodal samples, include pixel_values and image_grid_thw
        # so the evaluator can run them through the vision encoder.
        for key in ("pixel_values", "image_grid_thw"):
            if key in item:
                result = (*result, item[key])
        return result

    def __len__(self) -> int:
        if self.num_samples != 0:
            return min(self.num_samples, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        question = sample["question"]
        options = (
            ast.literal_eval(sample["options"])
            if isinstance(sample["options"], str)
            else sample["options"]
        )

        # Collect all images by slot (1-indexed, up to 7)
        image_slots = {i: sample.get(f"image_{i}") for i in range(1, 8)}

        # Format answer choices
        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )

        if self.processor is not None:
            return self._getitem_multimodal(
                question, options, choices_text, image_slots, sample
            )
        return self._getitem_text_only(question, choices_text, options, sample)

    def _getitem_multimodal(
        self,
        question: str,
        options: list[str],
        choices_text: str,
        image_slots: dict[int, Any],
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Process sample using VLM processor (images + text).

        Uses pure LM format (no chat template) ending with "Answer:" so
        the next predicted token is the answer letter. Vision tokens
        (<|vision_start|>/<|vision_end|>) are inserted manually for images.
        """
        valid_images = [img for img in image_slots.values() if img is not None]
        IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

        # Build prompt text, replacing <image N> placeholders with vision tokens
        ordered_images: list[Any] = []
        text_parts: list[str] = []
        parts = re.split(r"(<image \d+>)", question)
        for part in parts:
            m = re.match(r"<image (\d+)>", part)
            if m:
                img_idx = int(m.group(1))
                img = image_slots.get(img_idx)
                if img is not None:
                    text_parts.append(IMAGE_PLACEHOLDER)
                    ordered_images.append(img)
                else:
                    text_parts.append(part)
            else:
                text_parts.append(part)

        # If no placeholders but images exist, prepend them
        if not ordered_images and valid_images:
            for _img in valid_images:
                text_parts.insert(0, IMAGE_PLACEHOLDER)
            ordered_images = list(valid_images)

        prompt = "".join(text_parts).strip()
        prompt = f"{prompt}\n{choices_text}\nAnswer:"

        if self.image_size is not None:
            ordered_images = [image.resize(self.image_size) for image in ordered_images]

        inputs = self.processor(
            text=[prompt],
            images=ordered_images if ordered_images else None,
            return_tensors="pt",
        )

        assert inputs["input_ids"].shape[1] <= self.context_length
        inputs.pop("mm_token_type_ids", None)

        inputs["label"] = sample["answer"]
        inputs["num_options"] = len(options)

        return inputs

    def _getitem_text_only(
        self,
        question: str,
        choices_text: str,
        options: list[str],
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Process sample as text-only (no images)."""
        assert self.tokenizer is not None
        # Strip image references from the question
        clean_question = re.sub(r"<image \d+>", "", question).strip()
        formatted = f"{clean_question}\n{choices_text}\nAnswer:"

        tokenized = self.tokenizer(
            formatted,
            return_token_type_ids=False,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][:, -self.context_length :]
        attention_mask = tokenized["attention_mask"][:, -self.context_length :]

        # Encode answer label as token id
        answer_letter = sample["answer"].strip().upper()
        answer_token = self.tokenizer(
            f"Answer: {answer_letter}",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": answer_token,
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/MMMU/MMMU",
            split_description="validation split",
        )
