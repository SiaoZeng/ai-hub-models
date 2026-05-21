# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import AutoModelForMaskedLM, MobileBertTokenizer

from qai_hub_models.models._shared.bert_hf.model import BaseBertModel
from qai_hub_models.models._shared.bert_hf.model_patches import (
    patch_get_extended_attention_mask,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "google/mobilebert-uncased"


class MobileBertUncasedGoogle(BaseBertModel):
    """Exportable HuggingFace Distillbert Model"""

    @staticmethod
    def default_weights() -> str:
        return WEIGHTS_NAME

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> MobileBertUncasedGoogle:
        """Load HuggingFace Bert Model for Embeddings."""
        model = AutoModelForMaskedLM.from_pretrained(weights)
        tokenizer = MobileBertTokenizer.from_pretrained(weights)
        model.mobilebert.get_extended_attention_mask = patch_get_extended_attention_mask
        return cls(model, tokenizer)
