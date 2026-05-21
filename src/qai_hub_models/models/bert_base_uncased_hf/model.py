# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import AutoModelForMaskedLM, BertTokenizer
from typing_extensions import Self

from qai_hub_models.models._shared.bert_hf.model import BaseBertModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "google-bert/bert-base-uncased"


class BertBaseUncasedHf(BaseBertModel):
    """Exportable HuggingFace BERT Model"""

    @staticmethod
    def default_weights() -> str:
        return WEIGHTS_NAME

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> Self:
        """Load HuggingFace Bert Model for Embeddings."""
        model = AutoModelForMaskedLM.from_pretrained(weights)
        tokenizer = BertTokenizer.from_pretrained(weights)
        return cls(model, tokenizer)
