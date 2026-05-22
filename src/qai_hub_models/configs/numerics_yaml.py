# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from qai_hub_models_cli.proto import numerics_pb2
from qai_hub_models_cli.proto.shared import range_pb2

from qai_hub_models import Precision
from qai_hub_models.configs.proto_helpers import precision_to_proto, runtime_to_proto
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def get_numerics_yaml_path(model_id: str) -> Path:
    return QAIHM_MODELS_ROOT / model_id / "numerics.yaml"


class QAIHMModelNumerics(BaseQAIHMConfig):
    class DeviceDetails(BaseQAIHMConfig):
        partial_metric: float

    class Range(BaseQAIHMConfig):
        min: float | None = None
        max: float | None = None

    class MetricDetails(BaseQAIHMConfig):
        dataset_name: str
        dataset_link: str
        dataset_split_description: str
        metric_name: str
        metric_description: str
        metric_unit: str
        metric_range: QAIHMModelNumerics.Range
        # Maximum allowed deviation between torch and device accuracy, or
        # between actual accuracy and the benchmark value.
        # Paths exceeding this threshold are disabled by the scorecard.
        metric_enablement_threshold: float | None = None
        # Expected accuracy from info.yaml's numerics_benchmark.
        benchmark_value: float | None = None
        num_partial_samples: int
        partial_torch_metric: float
        device_metric: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = Field(default_factory=dict)

    metrics: list[MetricDetails] = Field(default_factory=list)

    def to_model_yaml(self, model_id: str) -> Path:
        path = get_numerics_yaml_path(model_id)
        self.to_yaml(path)
        return path

    def is_empty(self) -> bool:
        return len(self.metrics) == 0

    def to_proto(self, aihm_version: str, model_id: str) -> numerics_pb2.ModelNumerics:
        metrics: list[numerics_pb2.ModelNumerics.NumericsMetric] = []
        for m in self.metrics:
            device_metrics: list[
                numerics_pb2.ModelNumerics.NumericsMetric.DeviceNumericsMetrics
            ] = []
            for device, prec_dict in m.device_metric.items():
                for precision, path_dict in prec_dict.items():
                    for path, details in path_dict.items():
                        device_metrics.append(
                            numerics_pb2.ModelNumerics.NumericsMetric.DeviceNumericsMetrics(
                                device=str(device),
                                precision=precision_to_proto(precision),
                                runtime=runtime_to_proto(path.runtime),
                                partial_metric=details.partial_metric,
                            )
                        )

            metric_range = range_pb2.DoubleRange(
                min=m.metric_range.min, max=m.metric_range.max
            )
            metrics.append(
                numerics_pb2.ModelNumerics.NumericsMetric(
                    dataset_name=m.dataset_name,
                    dataset_link=m.dataset_link,
                    dataset_split_description=m.dataset_split_description,
                    metric_name=m.metric_name,
                    metric_description=m.metric_description,
                    metric_unit=m.metric_unit,
                    metric_range=metric_range,
                    metric_enablement_threshold=m.metric_enablement_threshold,
                    benchmark_value=m.benchmark_value,
                    num_partial_samples=m.num_partial_samples,
                    partial_torch_metric=m.partial_torch_metric,
                    device_metrics=device_metrics,
                )
            )

        return numerics_pb2.ModelNumerics(
            aihm_version=aihm_version,
            model_id=model_id,
            metrics=metrics,
        )

    @classmethod
    def from_model(
        cls: type[QAIHMModelNumerics], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelNumerics | None:
        numerics_path = get_numerics_yaml_path(model_id)
        if not_exists_ok and not os.path.exists(numerics_path):
            return None
        return cls.from_yaml(numerics_path)
