# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

import pytest

from qai_hub_models.configs.devices_and_chipsets_yaml import load_similar_devices
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ProfileScorecardJobYaml
from qai_hub_models.utils.collection_model_helpers import get_components
from qai_hub_models.utils.hub_clients import deployment_is_prod

EXPECTED_MODEL_CARD = QAIHMModelPerf.from_yaml(
    os.path.join(os.path.dirname(__file__), "perf_gt.yaml")
)


def test_generate_perf(hub_test_deployment: str) -> None:
    if not deployment_is_prod(hub_test_deployment):
        pytest.skip(
            f"This test uses jobs only accessible on production AI Hub Workbench. Enabled deployment: {hub_test_deployment}"
        )

    # Verify the model card is made correctly given the info
    job_ids = ProfileScorecardJobYaml.from_file(
        os.path.join(os.path.dirname(__file__), "profile_job_ids.yaml")
    )
    model_info = QAIHMModelInfo.from_model("trocr")
    component_names = get_components(model_info.id)

    paramaterizations: list[
        tuple[Precision, ScorecardProfilePath, ScorecardDevice]
    ] = []
    for device in ScorecardDevice.all_devices():
        for path in ScorecardProfilePath:
            for precision in model_info.code_gen_config.supported_precisions:
                if path.supports_precision(precision) and device.npu_supports_precision(
                    precision
                ):
                    paramaterizations.append((precision, path, device))  # noqa: PERF401

    model_perf = job_ids.summary_from_model(
        model_info.id,
        paramaterizations,
        component_names,
    )
    model_card = model_perf.get_perf_card()
    assert model_card == EXPECTED_MODEL_CARD


def _build_perf_card(model_id: str) -> QAIHMModelPerf:
    """Build a perf card from the checked-in intermediate job IDs."""
    job_ids = ProfileScorecardJobYaml.from_file(
        ScorecardArtifact.PROFILE_YAML.intermediates_path
    )
    model_info = QAIHMModelInfo.from_model(model_id)
    component_names = get_components(model_info.id)

    parameterizations: list[
        tuple[Precision, ScorecardProfilePath, ScorecardDevice]
    ] = []
    for device in ScorecardDevice.all_devices():
        for path in ScorecardProfilePath:
            for precision in model_info.code_gen_config.supported_precisions:
                if path.supports_precision(precision) and device.npu_supports_precision(
                    precision
                ):
                    parameterizations.append((precision, path, device))  # noqa: PERF401

    model_perf = job_ids.summary_from_model(
        model_info.id,
        parameterizations,
        component_names,
    )
    model_card = model_perf.get_perf_card(
        include_failed_jobs=False,
        include_unpublished_runtimes=False,
        exclude_form_factors=model_info.private_perf_form_factors or [],
        model_name=model_info.name,
    )
    model_card.apply_similar_devices(load_similar_devices())
    return model_card


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow (~2 min, requires prod Hub). Set RUN_SLOW_TESTS=1 to enable.",
)
def test_generate_perf_with_similar_devices(hub_test_deployment: str) -> None:
    if not deployment_is_prod(hub_test_deployment):
        pytest.skip(
            f"This test uses jobs only accessible on production AI Hub Workbench. Enabled deployment: {hub_test_deployment}"
        )

    model_card = _build_perf_card("inception_v3")
    expected = QAIHMModelPerf.from_model("inception_v3")
    assert str(model_card) == str(expected)
