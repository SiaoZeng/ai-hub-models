# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLM_QNN,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.models._shared.llm.test import CompileJobCache
from qai_hub_models.models.llama_v3_2_1b_instruct import Model
from qai_hub_models.models.llama_v3_2_1b_instruct.demo import llama_3_2_1b_chat_demo
from qai_hub_models.models.llama_v3_2_1b_instruct.export import (
    export_model,
)
from qai_hub_models.models.llama_v3_2_1b_instruct.model import (
    HF_REPO_NAME,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NUM_SPLITS,
    FPSplitModelWrapper,
    Llama3_2_1B_PreSplit,
    Llama3_2_1B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_8_elite_qrd, cs_x_elite
from qai_hub_models.scorecard.utils.testing_export_eval import run_llm_compile
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export_result import MultiGraphCollectionExportResult
from qai_hub_models.utils.llm_helpers import (
    create_genie_config,
    log_evaluate_test_result,
    log_perf_on_device_result,
)

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [f"llama_v3_2_1b_instruct_part_{i}_of_3.bin" for i in range(1, 4)]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 128256,
                "bos-token": 128000,
                "eos-token": [128001, 128008, 128009],
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 64,
                        "allow-async-init": False,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                    "positional-encoding": {
                        "type": "rope",
                        "rope-dim": 32,
                        "rope-theta": 500000,
                        "rope-scaling": {
                            "rope-type": "llama3",
                            "factor": 8.0,
                            "low-freq-factor": 1.0,
                            "high-freq-factor": 4.0,
                            "original-max-position-embeddings": 8192,
                        },
                    },
                },
            },
        }
    }

    assert expected_config == actual_config


# Full model tests
@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained()


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT_W4", "wikitext", 16.79, 0, marks=pytest.mark.nightly),
        ("DEFAULT_W4", "mmlu", 0.399, 1000),
        ("DEFAULT_W4", "tiny_mmlu", 0.43, 0),
        pytest.param("DEFAULT_W4A16", "wikitext", 17.47, 0, marks=pytest.mark.nightly),
        ("DEFAULT_W4A16", "mmlu", 0.390, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.18, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.482, 1000),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.41, 0),
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
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    extra_kwargs = (
        {"_skip_quantsim_creation": False, "fp_model": None} if is_unquantized else {}
    )
    actual_metric, _ = evaluate(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        skip_fp_model_eval=not is_unquantized,
        kwargs=dict(
            checkpoint=checkpoint,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            **extra_kwargs,
        ),
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint=checkpoint,
        metric=task,
        value=actual_metric,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=0.03, atol=0)


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    checkpoint_path = test.setup_test_quantization(
        QuantizedSplitModelWrapper,
        FPSplitModelWrapper,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
        use_dynamic_shapes=True,
    )
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.nightly
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device", "checkpoint"),
    [
        (Precision.w4, ScorecardCompilePath.GENIE, cs_8_elite_qrd, "DEFAULT_W4"),
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite, "DEFAULT_W4A16"),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    checkpoint: CheckpointSpec,
) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    # Pass both prompt (ar128) and token (ar1) sequence lengths so the
    # genie bundle includes both model types. Without ar1, Genie must use
    # the ar128 model for token generation, halving TPS on-device.
    result = run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            sequence_length=[DEFAULT_SEQUENCE_LENGTH, 1],
            context_length=[DEFAULT_CONTEXT_LENGTH],
            _skip_quantsim_creation=True,
            output_dir=test.GENIE_BUNDLES_ROOT,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()
    assert (genie_bundle_path / "sample_prompt.txt").exists()

    # TODO(https://github.com/qcom-ai-hub/tetracode/issues/19349): remove once
    # the QDC w4/w4a16 mix-up is resolved.
    assert isinstance(result, MultiGraphCollectionExportResult)
    print(f"[provenance] precision={precision} bundle={genie_bundle_path}")
    for compile_key, compile_job in (result.compile_jobs or {}).items():
        print(f"[provenance] compile_job[{compile_key}]={compile_job.job_id}")
    for link_key, link_job in (result.link_jobs or {}).items():
        print(f"[provenance] link_job[{link_key}]={link_job.job_id}")


@pytest.mark.nightly
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test can be run on GPU only. Also needs QDC package to run.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite),
        (Precision.w4, ScorecardCompilePath.GENIE, cs_8_elite_qrd),
    ],
)
@pytest.mark.qdc
def test_qdc(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    if scorecard_path.runtime != TargetRuntime.GENIE:
        pytest.skip("This test is only valid for Genie runtime.")
    if not (genie_bundle_path / "genie_config.json").exists():
        pytest.fail("The genie bundle does not exist.")

    from qai_hub_models.utils.qdc.genie_jobs import submit_genie_bundle_to_qdc_device

    # TODO(https://github.com/qcom-ai-hub/tetracode/issues/19349): remove once
    # the QDC w4/w4a16 mix-up is resolved.
    metadata = ModelMetadata.from_json(genie_bundle_path / "metadata.json")
    print(f"[provenance] precision={precision} bundle={genie_bundle_path}")
    print(f"[provenance] metadata.json precision={metadata.precision}")

    qdc_job_name = f"Genie {MODEL_ID} {precision}"
    tps, prefill_tps, min_ttft_ms, _ = submit_genie_bundle_to_qdc_device(
        os.environ["QDC_API_TOKEN"],
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=qdc_job_name,
    )
    assert tps is not None and min_ttft_ms is not None, "QDC execution failed."
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=min_ttft_ms,
    )
    # With both ar128 and ar1 in the genie bundle, TPS should match v1.
    if precision == Precision.w4:
        assert tps > 24.0
        assert min_ttft_ms < 100.0
    else:
        assert tps > 13.0
        assert min_ttft_ms < 135.0


def _get_llm_perf_params() -> list[tuple[Precision, ScorecardDevice]]:
    params = get_llm_perf_parametrization(
        MODEL_ID,
        default_devices=[cs_8_elite_qrd],
        default_precisions=[Precision.w4],
    )
    return params if params else [(Precision.w4, cs_8_elite_qrd)]


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    return LLMPerfConfig.from_environment()


@pytest.fixture(scope="session")
def compile_job_cache() -> CompileJobCache:
    return CompileJobCache()


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires GPU and the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(("precision", "device"), _get_llm_perf_params())
def test_llm_perf(
    precision: Precision,
    device: ScorecardDevice,
    compile_job_cache: CompileJobCache,
    llm_perf_config: LLMPerfConfig,
) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()

    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=MODEL_ID,
        export_model_func=export_model,
        device=device,
        precision=precision,
        compile_job_cache=compile_job_cache,
        output_dir=test.GENIE_BUNDLES_ROOT,
        model_cls=Llama3_2_1B_QuantizablePreSplit,
        model_asset_version=MODEL_ASSET_VERSION,
        num_splits=NUM_SPLITS,
        export_context_lengths=llm_perf_config.export_context_lengths,
        export_sequence_lengths=llm_perf_config.export_sequence_lengths,
        fp_model_cls=FPSplitModelWrapper,
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
