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

import qai_hub_models.models.qwen3_vl_4b_instruct as _model_module
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.qwen3_vl_4b_instruct import MODEL_ID, Model
from qai_hub_models.models.qwen3_vl_4b_instruct.export import (
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
    get_export_parameterized_pytest_config,
    get_link_parameterized_pytest_config,
    pytest_device_idfn,
)
from qai_hub_models.scorecard.utils.testing import skip_invalid_runtime_device
from qai_hub_models.scorecard.utils.testing_export_eval import (
    compile_via_export,
    export_test_e2e,
    link_via_export,
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
        TargetRuntime.GENIEX_QAIRT,
    ],
}


# All runtime + precision pairs that are enabled for testing and have no known failure reasons.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
PASSING_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w4a16: [
        TargetRuntime.GENIE,
        TargetRuntime.GENIEX_QAIRT,
    ],
}


EVAL_DEVICE = ScorecardDevice.get("Samsung Galaxy S25 (Family)")
HAS_EVAL_DATASET = len(Model.get_eval_dataset_classes()) > 0


@pytest.mark.compile
def test_runtime_model_validation() -> None:
    perform_runtime_model_validation(
        Model, MODEL_ID, getattr(_model_module, "App", None)
    )


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
        is_llm=True,
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
    # Collection models are not torch-accuracy validated in the scorecard.
    return []


@pytest.fixture(scope="module")
def torch_evaluate_mock_outputs(
    torch_val_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Because the below method does some memory movement,
    it is called in a fixture so its output can be reused.
    """
    # Collection models are not torch-accuracy validated in the scorecard.
    return []


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_export_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
        requires_aot_prepare=True,
        is_llm=True,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.llm_export
def test_export(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    # When all compile/link jobs are cached, the released asset bytes come from
    # target_model.download() off the cached jobs -- the ONNX that
    # Model.from_pretrained would have built and the input_spec the splitter
    # would have inferred are both unused. Skip the ~30-min torch.onnx export
    # (inside resolve_default_checkpoint) and the splitter (inside
    # get_input_spec), keeping the (component, graph) iteration shape so
    # upload_model and compile_model still walk the right keys.
    from pathlib import Path as _StubPath
    from typing import Any

    from transformers import AutoConfig, AutoTokenizer

    from qai_hub_models.models._shared.llama3.model import (
        LlamaDynamicQuantizablePreSplitMixin,
    )
    from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

    def _stub_resolve_default_checkpoint(
        cls: Any, precision: Precision, host_device: object, fp_model: object
    ) -> tuple[str, None]:
        precision_checkpoint = cls.default_checkpoint[precision]
        encodings_path = _StubPath(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                f"{precision_checkpoint}/model.encodings",
            ).fetch()
        )
        ckpt = encodings_path.parent
        # Pre-populate tokenizer.json and config.json in the encodings dir so
        # downstream LLM_AIMETOnnx.__init__'s get_tokenizer(checkpoint) /
        # get_llm_config(checkpoint) reads from disk instead of needing the
        # 14 GB FP HuggingFace load.
        if not (ckpt / "tokenizer.json").exists():
            AutoTokenizer.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(
                ckpt
            )
        if not (ckpt / "config.json").exists():
            AutoConfig.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(ckpt)
        return str(ckpt), None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LlamaDynamicQuantizablePreSplitMixin,
            "resolve_default_checkpoint",
            classmethod(_stub_resolve_default_checkpoint),
        )
        mp.setattr(
            Model,
            "get_component_graph_input_spec",
            lambda self, component_name, graph_name, *a, **kw: {},
        )
        mp.setattr(
            Model,
            "get_component_graph_hub_compile_options",
            lambda self, *a, **kw: "",
        )
        skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
        try:
            export_test_e2e(
                export_model, Model, MODEL_ID, precision, scorecard_path, device
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
