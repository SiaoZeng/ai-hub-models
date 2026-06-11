# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
)
from qai_hub_models.models._shared.lm_driver.generator import (
    Generator,
    PrecomputedCosSinGeneratorMixin,
    TransposedKVGeneratorMixin,
    VLM_Generator,
)
from qai_hub_models.models._shared.lm_driver.utils.rope_embedding import (
    RopeEmbeddingProtocol,
)

ModelType = LLMBase | LLM_AIMETOnnx | LLM_QNN


class HubCompatibleVLMGenerator(  # type: ignore[misc]
    PrecomputedCosSinGeneratorMixin, TransposedKVGeneratorMixin, VLM_Generator
):
    pass


class _ModelAdapter(torch.nn.Module):
    """Adapts an AIHM model to the lm_driver Generator model contract.

    The Generator expects its ``model`` to be an nn.Module with:
    - ``.config`` -> PretrainedConfig
    - ``.device`` -> torch.device
    - ``.dtype`` -> torch.dtype
    - ``.rope_embedding`` -> RopeEmbeddingProtocol
    - callable via ``forward(*args)``
    - ``.cleanup()`` for teardown
    """

    def __init__(
        self,
        model: ModelType,
        rope_embedding: RopeEmbeddingProtocol,
    ) -> None:
        super().__init__()
        self._model = model
        self._rope_embedding = rope_embedding

    @property
    def config(self) -> PretrainedConfig:
        return self._model.llm_config

    @property
    def device(self) -> torch.device:
        if (
            hasattr(self._model, "host_device")
            and self._model.host_device is not None
            and isinstance(self._model.host_device, torch.device)
        ):
            return self._model.host_device
        return next(iter(self._model.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def rope_embedding(self) -> RopeEmbeddingProtocol:
        return self._rope_embedding

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self._model(*args)
        if isinstance(outputs, list):
            return tuple(outputs)
        return outputs

    def release(self) -> None:
        self._model.release()

    def cleanup(self) -> None:
        self._model.release()


def make_generator(
    model: ModelType,
    sequence_length: int | list[int] | None = None,
    context_length: int | None = None,
    vision_model: torch.nn.Module | None = None,
    model_cls: type | None = None,
    device: torch.device | None = None,
) -> Generator:
    """Create a Generator from an AIHM LLM model (text-only or VLM)."""
    if model_cls is None:
        model_cls = type(model)
    if sequence_length is None:
        sequence_length = DEFAULT_SEQUENCE_LENGTH
    if context_length is None:
        context_length = DEFAULT_CONTEXT_LENGTH

    GeneratorCls = model_cls.GeneratorClass  # type: ignore[attr-defined]
    adapted = _ModelAdapter(model, rope_embedding=model.embedding)
    if device is None:
        device = adapted.device

    if issubclass(GeneratorCls, VLM_Generator):
        embedding_weights = model.get_embedding_weights()  # type: ignore[operator, unused-ignore]
        embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights.float(),
            freeze=True,
        ).to(device)
        config = model._original_llm_config
        if vision_model is not None:
            VisionWrapper = model_cls.VisionModelWrapper  # type: ignore[attr-defined]
            vision_model = VisionWrapper(vision_model).to(device).float()
        generator_kwargs: dict = dict(
            backbone_model=adapted,
            vision_model=vision_model,
            embedding=embedding,
            tokenizer=model.tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            config=config,
        )
        if hasattr(model_cls, "get_visual_output_names"):
            generator_kwargs["visual_output_names"] = model_cls.get_visual_output_names(
                config
            )
        return GeneratorCls(**generator_kwargs)
    return GeneratorCls(
        model=adapted,
        tokenizer=model.tokenizer,
        sequence_length=sequence_length,
        context_length=context_length,
    )
