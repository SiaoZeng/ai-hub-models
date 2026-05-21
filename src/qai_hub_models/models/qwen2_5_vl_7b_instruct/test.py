# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EXPORT_CONTEXT_LENGTHS,
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    LLM_QNN,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.models._shared.llm.test import CompileJobCache
from qai_hub_models.models.common import Precision
from qai_hub_models.models.qwen2_5_vl_7b_instruct import (
    MODEL_ID,
    Model,
    VisionEncoder,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct.export import export_model
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    MODEL_ASSET_VERSION,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import cs_8_elite_gen_5
from qai_hub_models.utils.llm_helpers import (
    log_evaluate_test_result,
    log_perf_on_device_result,
)

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen2_5_VL_7B_PreSplit.clear_cache()
    Qwen2_5_VL_7B_QuantizablePreSplit.clear_cache()
    Qwen2_5_VL_7B_QuantizablePreSplit.from_pretrained(checkpoint=checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT", "wikitext", 9.75, 0, marks=pytest.mark.nightly),
        ("DEFAULT", "mmlu", 0.689, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 8.38, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.73, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    dataset_cls = next(
        d
        for d in Qwen2_5_VL_7B_PreSplit.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen2_5_VL_7B_PreSplit.clear_cache()
    Qwen2_5_VL_7B_QuantizablePreSplit.clear_cache()
    actual_metric, _ = evaluate(
        quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # placeholder — no QNN variant yet
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        skip_fp_model_eval=True,
        kwargs=dict(
            checkpoint=checkpoint,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
        vision_encoder_cls=VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint="DEFAULT_W4A16" if checkpoint == "DEFAULT" else checkpoint,
        metric=task,
        value=actual_metric,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=0.03, atol=0)


def _get_llm_perf_params() -> list[tuple[Precision, ScorecardDevice]]:
    params = get_llm_perf_parametrization(
        MODEL_ID,
        default_devices=[cs_8_elite_gen_5],
        default_precisions=[Precision.w4a16],
    )
    return params if params else [(Precision.w4a16, cs_8_elite_gen_5)]


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(("precision", "device"), _get_llm_perf_params())
def test_llm_perf(
    precision: Precision,
    device: ScorecardDevice,
    compile_job_cache: CompileJobCache,
    llm_perf_config: LLMPerfConfig,
) -> None:
    tps, ttft = test.run_llm_perf_test(
        model_id=MODEL_ID,
        export_model_func=export_model,
        device=device,
        precision=precision,
        compile_job_cache=compile_job_cache,
        output_dir=test.GENIE_BUNDLES_ROOT,
        model_cls=Model,  # type: ignore[arg-type]
        model_asset_version=MODEL_ASSET_VERSION,
        num_splits=NUM_SPLITS,
        export_context_lengths=llm_perf_config.export_context_lengths
        or DEFAULT_EXPORT_CONTEXT_LENGTHS,
        export_sequence_lengths=llm_perf_config.export_sequence_lengths
        or DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        num_layers_per_split=NUM_LAYERS_PER_SPLIT,
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        ttft_ms=ttft,
    )
