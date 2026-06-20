# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForMaskedLM, DistilBertTokenizer
from typing_extensions import Self

from qai_hub_models.models._shared.bert_hf.model import BaseBertModel
from qai_hub_models.utils.base_model import SerializationSettings

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "distilbert/distilbert-base-uncased"


class DistilbertBase(BaseBertModel):
    """Exportable HuggingFace Distillbert Model"""

    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
        super().__init__(model, tokenizer)
        # Tied input/output embedding weights become a single shared
        # initializer under torch.export, which AIMET rejects in quantize.
        self.serialization_settings = SerializationSettings(use_pt2=False)

    @staticmethod
    def default_weights() -> str:
        return WEIGHTS_NAME

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> Self:
        """Load HuggingFace Bert Model for Embeddings."""
        model = AutoModelForMaskedLM.from_pretrained(weights)
        tokenizer = DistilBertTokenizer.from_pretrained(weights)
        return cls(model, tokenizer)
