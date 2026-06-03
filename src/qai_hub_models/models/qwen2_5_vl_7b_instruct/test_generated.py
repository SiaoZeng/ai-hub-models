# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import qai_hub as hub
import torch

import qai_hub_models.models.qwen2_5_vl_7b_instruct as _model_module
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.qwen2_5_vl_7b_instruct import MODEL_ID, Model
from qai_hub_models.models.qwen2_5_vl_7b_instruct.export import (
    compile_model,
    export_model,
    link_model,
    upload_model,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.errors import CachedScorecardJobError
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_evaluation_parameterized_pytest_config,
    get_export_parameterized_pytest_config,
    get_link_parameterized_pytest_config,
    pytest_device_idfn,
)
from qai_hub_models.scorecard.utils.testing import skip_invalid_runtime_device
from qai_hub_models.scorecard.utils.testing_export_eval import (
    accuracy_on_sample_inputs_via_export,
    compile_via_export,
    export_test_e2e,
    link_via_export,
    split_and_group_accuracy_validation_output_batches,
    torch_inference_for_accuracy_validation_outputs,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.validation import perform_runtime_model_validation

# All runtime + precision pairs that are enabled for testing and are compatibile with this model.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
ENABLED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w4a16: [
        TargetRuntime.GENIE,
    ],
}


# All runtime + precision pairs that are enabled for testing and have no known failure reasons.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
PASSING_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w4a16: [
        TargetRuntime.GENIE,
    ],
}


EVAL_DEVICE = ScorecardDevice.get("Samsung Galaxy S25 (Family)")
HAS_EVAL_DATASET = len(Model.get_eval_dataset_classes()) > 0


@pytest.mark.compile
def test_runtime_model_validation() -> None:
    perform_runtime_model_validation(
        Model, MODEL_ID, getattr(_model_module, "App", None)
    )


ALL_COMPONENTS = Model.component_class_names


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_compile_parameterized_pytest_config(
        MODEL_ID,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compile
def test_compile(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        compile_via_export(
            compile_model,
            MODEL_ID,
            Model.from_pretrained(checkpoint=f"DEFAULT_{str(precision).upper()}"),
            precision,
            scorecard_path,
            device,
            is_aimet=True,
            upload_model=upload_model,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_link_parameterized_pytest_config(
        MODEL_ID,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.link
def test_link(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        link_via_export(
            link_model,
            MODEL_ID,
            Model.from_pretrained(checkpoint=f"DEFAULT_{str(precision).upper()}"),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def torch_val_outputs() -> list[np.ndarray]:
    """
    Because the below method downloads a dataset over the internet,
    it is called in a fixture so it can be reused.
    """
    if not HAS_EVAL_DATASET:
        return []
    return torch_inference_for_accuracy_validation_outputs(MODEL_ID)


@pytest.fixture(scope="module")
def torch_evaluate_mock_outputs(
    torch_val_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Because the below method does some memory movement,
    it is called in a fixture so its output can be reused.
    """
    if not HAS_EVAL_DATASET:
        return []
    return split_and_group_accuracy_validation_output_batches(torch_val_outputs)


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compute_device_accuracy
def test_val_accuracy(
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    torch_val_outputs: list[np.ndarray],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    try:
        accuracy_on_sample_inputs_via_export(
            export_model,
            MODEL_ID,
            Model.from_pretrained(checkpoint=f"DEFAULT_{str(precision).upper()}"),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_export_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
        requires_aot_prepare=True,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.export
def test_export(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        export_test_e2e(
            export_model,
            Model,
            MODEL_ID,
            precision,
            scorecard_path,
            device,
            ALL_COMPONENTS,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


# Cache serialize() and hub.upload_model() across the module so the same
# (component, graph, input_spec) is serialized once and the resulting bytes are
# uploaded once -- matters most for multi-GB AIMET LLM bundles.
@pytest.fixture(scope="module", autouse=True)
def cached_serialize_for_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[pytest.MonkeyPatch, None, None]:
    cache_dir = tmp_path_factory.mktemp("serialize_cache")
    with pytest.MonkeyPatch.context() as mp:
        model_cache: dict[str, Path] = {}
        upload_cache: dict[str, hub.Model] = {}

        real_upload_model = hub.upload_model

        def _cached_upload_model(
            model: hub.client.SourceModel | str,
            name: str | None = None,
            project: str | hub.client.Project | None = None,
        ) -> hub.Model:
            key = str(model)
            cached = upload_cache.get(key)
            if cached is None:
                cached = real_upload_model(model, name, project)
                upload_cache[key] = cached
            return cached

        mp.setattr(hub, "upload_model", _cached_upload_model)
        serialize_component_graph = Model.serialize_component_graph

        def _cached_serialize_component_graph(
            self: Model,
            component_name: str,
            graph_name: str | None,
            output_dir: str | os.PathLike,
            input_spec: InputSpec | None = None,
        ) -> Path:
            precision = self.components[component_name].component_precision()
            model_key = f"{component_name}|{graph_name}|{precision}|{input_spec}"
            cached = model_cache.get(model_key)
            if not cached:
                cached = serialize_component_graph(
                    self, component_name, graph_name, cache_dir, input_spec
                )
                model_cache[model_key] = cached
            return cached

        mp.setattr(
            Model, "serialize_component_graph", _cached_serialize_component_graph
        )
        yield mp
