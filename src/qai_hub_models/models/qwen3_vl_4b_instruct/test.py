# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.llm_helpers import (
    log_evaluate_test_result,
    log_perf_on_device_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    LLM_QNN,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.models.qwen3_vl_4b_instruct import (
    MODEL_ID,
    VisionEncoder,
)
from qai_hub_models.models.qwen3_vl_4b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Qwen3_VL_4B_PreSplit,
    Qwen3_VL_4B_QuantizablePreSplit,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import cs_8_elite_gen_5_qrd

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_VL_4B_PreSplit.release()
    Qwen3_VL_4B_QuantizablePreSplit.release()
    Qwen3_VL_4B_QuantizablePreSplit.from_pretrained(checkpoint=checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT", "wikitext", 10.58, 0, marks=pytest.mark.nightly),
        ("DEFAULT", "mmlu", 0.698, 1000),
        ("DEFAULT", "mmmu", 0.535, 200),
        # Image+prompt generation + LLM-grader smoke test (5 samples). Weekly
        # (evaluate-only) since VLM generation is slow. The grader label can
        # flip across hosts, so expected_metric is a floor.
        ("DEFAULT_UNQUANTIZED", "wikitext", 9.85, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.72, 0),
        ("DEFAULT_UNQUANTIZED", "mmmu", 0.555, 200),
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
        for d in Qwen3_VL_4B_PreSplit.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_VL_4B_PreSplit.release()
    Qwen3_VL_4B_QuantizablePreSplit.release()
    actual_metric, _ = evaluate(
        quantized_model_cls=Qwen3_VL_4B_QuantizablePreSplit,
        fp_model_cls=Qwen3_VL_4B_PreSplit,
        qnn_model_cls=LLM_QNN,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
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
        default_devices=[cs_8_elite_gen_5_qrd],
        default_precisions=[Precision.w4a16],
    )
    return params if params else [(Precision.w4a16, cs_8_elite_gen_5_qrd)]


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(("precision", "device"), _get_llm_perf_params())
def test_llm_perf(
    precision: Precision,
    device: ScorecardDevice,
    llm_perf_config: LLMPerfConfig,
) -> None:
    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=MODEL_ID,
        device=device,
        precision=precision,
        output_dir=test.GENIE_BUNDLES_ROOT,
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
    )
