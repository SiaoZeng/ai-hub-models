# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared utilities for LLM performance collection.

Provides:
- LLMPerfConfig: env-var driven configuration dataclass
- get_llm_perf_parametrization: generates (precision, device) pytest params
- update_perf_yaml: writes TPS/TTFT metrics into a model's perf.yaml

The compile/QDC test logic lives in _shared/llm/test.py (run_llm_perf_test).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from qai_hub_models import Precision
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import (
    LLM_COMPILE_DEVICES,
    LLM_W4FP16_COMPILE_DEVICES,
    get_canonical_chipset_name,
)
from qai_hub_models.scorecard.envvars import (
    LLMPerfPrecisionsEnvvar,
    LLMPerfReleaseAssetsEnvvar,
    SpecialLLMPerfPrecisionSetting,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ScorecardAssetYaml


@dataclass
class LLMPerfConfig:
    """Configuration for LLM performance collection.

    Loads configuration from environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all"
    - QAIHM_TEST_DEVICES: Comma-separated device names
    - SKIP_PERF_UPDATE: If set, skip updating perf.yaml files
    - QAIRT_SDK_PATH: Path to QAIRT SDK for auto devices
    """

    models: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    skip_perf_update: bool = False
    qairt_sdk_path: str | None = None

    @classmethod
    def from_environment(cls) -> LLMPerfConfig:
        """Create config from environment variables."""
        models_str = os.environ.get("QAIHM_LLM_MODELS", "")
        devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")

        models = [m.strip() for m in models_str.split(",") if m.strip()]
        devices = [d.strip() for d in devices_str.split(",") if d.strip()]

        return cls(
            models=models,
            devices=devices,
            skip_perf_update=bool(os.environ.get("SKIP_PERF_UPDATE")),
            qairt_sdk_path=os.environ.get("QAIRT_SDK_PATH"),
        )


def get_supported_precisions(model_id: str) -> list[Precision]:
    """Get the supported precisions for a model from code-gen.yaml."""
    code_gen = QAIHMModelCodeGen.from_model(model_id)
    return code_gen.supported_precisions


def load_release_assets_for_model(model_id: str) -> QAIHMModelReleaseAssets:
    """Load release assets for ``model_id``, preferring an in-flight workflow artifact.

    When ``LLMPerfReleaseAssetsEnvvar`` is set to a combined release-assets.yaml
    (the kind uploaded as a per-split workflow artifact, with
    ``models: {<model_id>: ...}``), pull this model's entry from there instead of
    the committed per-model copy. Lets a workflow run consume the release-assets.yaml
    the LLM asset-upload step just produced before the consolidated PR has merged.
    """
    if not LLMPerfReleaseAssetsEnvvar.is_default():
        path = LLMPerfReleaseAssetsEnvvar.get()
        if path.exists():
            entry = ScorecardAssetYaml.from_yaml(path).models.get(model_id)
            if entry is not None:
                return entry
    return QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)


def _get_devices_for_precision(
    precision: Precision,
    override_devices: list[ScorecardDevice] | None,
) -> list[ScorecardDevice]:
    """Return the devices applicable to a given precision.

    If override_devices is provided (from QAIHM_TEST_DEVICES), intersects
    that list with the compile-device sets so only valid combos are returned.
    Otherwise uses LLM_COMPILE_DEVICES (+ LLM_W4FP16_COMPILE_DEVICES for w4).
    """
    compile_devices: list[ScorecardDevice] = list(LLM_COMPILE_DEVICES)
    if precision == Precision.w4:
        compile_devices += LLM_W4FP16_COMPILE_DEVICES

    if override_devices is None:
        return compile_devices

    compile_set = set(compile_devices)
    return [d for d in override_devices if d in compile_set]


def get_llm_perf_parametrization(
    model_id: str,
    default_devices: list[ScorecardDevice] | None = None,
    default_precisions: list[Precision] | None = None,
) -> list[tuple[Precision, ScorecardDevice]]:
    """Generate pytest parametrization for LLM performance tests.

    Selects devices per precision based on LLM_COMPILE_DEVICES (all precisions)
    and LLM_W4FP16_COMPILE_DEVICES (w4 only).

    Environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all". If set and this
      model is not in the list, returns [] so the test is skipped.
    - QAIHM_TEST_DEVICES: Comma-separated device names. When set, acts as a
      filter over the compile-device sets (only devices in both lists are used).
    - QAIHM_LLM_PERF_PRECISIONS: See :class:`LLMPerfPrecisionsEnvvar`.
      ``default`` (the envvar default) honors the test's ``default_precisions``
      arg, falling back to supported_precisions when unset; ``all`` uses every
      supported precision; explicit precisions are intersected with the
      model's supported_precisions.
    """
    models_str = os.environ.get("QAIHM_LLM_MODELS", "")
    if models_str and models_str.strip().lower() != "all":
        allowed = [m.strip() for m in models_str.split(",") if m.strip()]
        if model_id not in allowed:
            return []

    devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")
    override_devices: list[ScorecardDevice] | None
    if devices_str and devices_str.strip().lower() == "all":
        override_devices = None
    elif devices_str:
        device_names = [d.strip() for d in devices_str.split(",") if d.strip()]
        override_devices = [
            ScorecardDevice._registry[name]
            for name in device_names
            if name in ScorecardDevice._registry
        ]
    else:
        override_devices = default_devices

    supported_precisions = get_supported_precisions(model_id)
    precision_setting = LLMPerfPrecisionsEnvvar.get()
    if SpecialLLMPerfPrecisionSetting.ALL in precision_setting:
        precisions = supported_precisions
    elif SpecialLLMPerfPrecisionSetting.DEFAULT in precision_setting:
        precisions = default_precisions or supported_precisions
    else:
        supported_set = set(supported_precisions)
        precisions = [
            Precision.parse(p)
            for p in precision_setting
            if isinstance(p, str) and Precision.parse(p) in supported_set
        ]

    result: list[tuple[Precision, ScorecardDevice]] = []
    for precision in precisions:
        result.extend(
            (precision, device)
            for device in _get_devices_for_precision(precision, override_devices)
        )
    return result


def update_perf_yaml(
    model_id: str,
    device_name: str,
    precision: Precision,
    context_length: int,
    tps: float,
    ttft_ms: float,
    prefill_tps: float | None = None,
) -> None:
    """Update the perf.yaml file for a model with new LLM metrics."""
    perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)

    info = QAIHMModelInfo.from_model(model_id)
    component_name = info.name

    device = ScorecardDevice.get(device_name, return_unregistered=True)
    if device not in perf.supported_devices:
        perf.supported_devices.append(device)

    chipset = get_canonical_chipset_name(device.chipset)
    if chipset not in perf.supported_chipsets:
        perf.supported_chipsets.append(chipset)

    if precision not in perf.precisions:
        perf.precisions[precision] = QAIHMModelPerf.PrecisionDetails()

    precision_details = perf.precisions[precision]

    if component_name not in precision_details.components:
        precision_details.components[component_name] = QAIHMModelPerf.ComponentDetails()

    component_details = precision_details.components[component_name]

    if device not in component_details.performance_metrics:
        component_details.performance_metrics[device] = {}

    device_metrics = component_details.performance_metrics[device]
    genie_path = ScorecardProfilePath.GENIE

    if genie_path not in device_metrics:
        device_metrics[genie_path] = QAIHMModelPerf.PerformanceDetails()

    perf_details = device_metrics[genie_path]

    # Max TTFT is estimated assuming linear scaling with prompt length.
    estimated_max_ttft_ms = ttft_ms * (context_length / 128)
    llm_metric = QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
        context_length=context_length,
        tokens_per_second=tps,
        time_to_first_token_range_milliseconds=QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds(
            min=ttft_ms,
            max=estimated_max_ttft_ms,
        ),
        prefill_tokens_per_second=prefill_tps,
    )

    if perf_details.llm_metrics is None:
        perf_details.llm_metrics = [llm_metric]
    else:
        found = False
        for i, existing in enumerate(perf_details.llm_metrics):
            if existing.context_length == context_length:
                perf_details.llm_metrics[i] = llm_metric
                found = True
                break
        if not found:
            perf_details.llm_metrics.append(llm_metric)

    perf.to_model_yaml(model_id)
    print(f"Updated perf.yaml for {model_id}")
