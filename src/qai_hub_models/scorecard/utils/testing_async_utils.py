# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, cast

import qai_hub as hub
from pydantic import Field
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models import Precision
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.evaluators.metrics import MetricMetadata
from qai_hub_models.scorecard import (
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.params import ScExportTestParams
from qai_hub_models.scorecard.results.yaml import CompileScorecardJobYaml
from qai_hub_models.utils.asset_loaders import load_yaml, qaihm_temp_dir
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.base_dataset import DatasetMetadata
from qai_hub_models.utils.file_hash import file_hashes_are_identical
from qai_hub_models.utils.onnx.torch_wrapper import extract_onnx_zip

__all__ = [
    "CompileJobsAreIdenticalCache",
    "append_line_to_file",
    "cache_dataset",
    "callable_side_effect",
    "get_accuracy_columns",
    "get_accuracy_metadata_columns",
    "get_accuracy_numerics_columns",
    "get_cached_dataset",
    "get_cached_dataset_entries",
    "get_job_date",
    "write_accuracy",
]

# If a model has many outputs, how many of them to store PSNR for
MAX_PSNR_VALUES = 10


def callable_side_effect(side_effects: Iterator) -> Callable:
    """
    Return a function that:
        * Gets the next value in side_effects.
            * If the value is not callable, returns it directly.
            * If the value is callable, calls() it (using passthrough function arguments)
                and returns the function's result.

        Example,
        def my_func(input: str) -> str:
            return str + '_hello_world'
        f = callable_side_effect('1', '3', my_func)

        f("hello_world") # returns: '1'
        f("testing 123") # returns: '3'
        f("beep") # returns 'beep_hello_world'
        f("boop") # raises error (out of values to iterate over)
    """

    def f(*args: Any, **kwargs: Any) -> object:
        result = next(side_effects)
        if callable(result):
            return result(*args, **kwargs)
        return result

    return f


def append_line_to_file(path: os.PathLike, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def get_accuracy_metadata_columns() -> list[str]:
    return [
        "dataset_name",
        "dataset_link",
        "split_description",
        "metric_name",
        "metric_unit",
        "metric_description",
        "metric_min",
        "metric_max",
        "metric_threshold",
        "num_samples",
    ]


def get_accuracy_numerics_columns() -> list[str]:
    cols = ["Torch Accuracy", "Sim Accuracy", "Device Accuracy"]
    cols.extend(f"PSNR_{i}" for i in range(MAX_PSNR_VALUES))
    return cols


def get_accuracy_columns() -> list[str]:
    cols = ["model_id", "precision", "runtime"]
    cols.extend(get_accuracy_numerics_columns())
    cols.extend(["date", "branch", "chipset"])
    cols.extend(get_accuracy_metadata_columns())
    return cols


def cache_dataset(model_id: str, dataset_name: str, dataset: hub.Dataset) -> None:
    append_line_to_file(
        ScorecardArtifact.DATASET_IDS.touch(),
        f"{model_id}_{dataset_name}: {dataset.dataset_id}",
    )


def get_cached_dataset(model_id: str, dataset_name: str) -> hub.Dataset | None:
    dataset_ids = load_yaml(ScorecardArtifact.DATASET_IDS.touch())
    key = f"{model_id}_{dataset_name}"
    if key not in dataset_ids:
        return None
    return hub.get_dataset(dataset_ids[key])


def get_cached_dataset_entries(
    model_id: str, dataset_name: str
) -> DatasetEntries | None:
    if x := get_cached_dataset(model_id, dataset_name):
        return cast(DatasetEntries, x.download())
    return None


def get_job_date() -> str:
    date_file = ScorecardArtifact.DATE.touch()
    if date_file.stat().st_size == 0:
        curr_date = datetime.today().strftime("%Y-%m-%d")
        with open(date_file, "w") as f:
            f.write(curr_date)
        return curr_date
    with open(date_file) as f:
        return f.read()


def write_accuracy(
    model_name: str,
    chipset: str,
    precision: Precision,
    path: ScorecardProfilePath,
    psnr_values: list[str],
    torch_accuracy: float | None = None,
    device_accuracy: float | None = None,
    sim_accuracy: float | None = None,
    dataset_name: str | None = None,
    dataset_metadata: DatasetMetadata | None = None,
    metric_metadata: MetricMetadata | None = None,
    num_samples: int | None = None,
) -> None:
    line = f"{model_name},{precision!s},{path.value},"
    line += f"{torch_accuracy:.3g}," if torch_accuracy is not None else ","
    line += f"{sim_accuracy:.3g}," if sim_accuracy is not None else ","
    line += f"{device_accuracy:.3g}," if device_accuracy is not None else ","
    if len(psnr_values) >= MAX_PSNR_VALUES:
        line += ",".join(psnr_values[:10])
    else:
        # If the psnr list is empty, we only want 9 commas after
        line += ",".join(psnr_values) + "," * min(MAX_PSNR_VALUES - len(psnr_values), 9)
    line += f",{get_job_date()},main,{chipset}"

    line += ","
    if dataset_name is not None:
        line += dataset_name

    if dataset_metadata is not None:
        line += f",{dataset_metadata.link},{dataset_metadata.split_description}"
    else:
        line += ",,"

    if metric_metadata is not None:
        line += f",{metric_metadata.name},{metric_metadata.unit},{metric_metadata.description},{metric_metadata.range[0]},{metric_metadata.range[1]},{metric_metadata.float_vs_device_threshold}"
    else:
        line += ",,,"

    line += ","
    if num_samples is not None:
        line += str(num_samples)

    accuracy_file = ScorecardArtifact.ACCURACY_CSV.touch()
    if accuracy_file.stat().st_size == 0:
        append_line_to_file(accuracy_file, ",".join(get_accuracy_columns()))
    append_line_to_file(accuracy_file, line)


class CompileJobsAreIdenticalCache(BaseQAIHMConfig):
    """
    When running scorecard, users have the option to enable a caching feature. This feature:
        * Checks if compile assets from the previous scorecard and the current scorecard are the same.
        * If the compiled assets are the same, profile / inference jobs from the previous scorecard are re-used.

    One challenge in the implementation the above caching feature is that we need to download the output of both
    compile jobs in each test and MD5 each file to determine if they're the same. The "naive" implementation runs this
    on every individual profile & inference test. This is unecessary for assets that aren't device-specific.

    This caches the sameness of current compile jobs and previous compile jobs, so we only need to compute sameness once.

    -----

    SUCCESSFUL COMPILE JOBS
        If:
            * both compile jobs from the previous scorecard and the current scorecard are succesful
            * the produced asset file is the same (md5 hash)
        The jobs are considered "identical".

    FAILING and MISSING COMPILE JOBS
        If:
            * a compile job is missing in EITHER OR BOTH the previous / current scorecard
            * a compile job failed in EITHER OR BOTH the previous / current scorecard,
        The jobs are considered NOT identical. This is the case even if both scorecard jobs fail with the same reason.

    COMPONENT MODELS
        Component models are considered "identical" only if all components' compile jobs are identical (per the above guidelines).
    """

    compile_jobs_are_identical: dict[str, bool] = Field(default_factory=dict)

    def is_identical(self, params: ScExportTestParams) -> bool:
        """
        Returns true if:
            * All compile jobs for the given parameters passed in previous and current scorecards,
            and
            * All compiled assets for the given parameters are identical between the previous and current scorecards.

        If the sameness of relevant compile jobs is not cached, sameness will be computed and added to the "CompileJobsAreIdentical" cache.
        """
        try:
            identical = all(
                self.compile_jobs_are_identical[pp.compile_job_id]
                for pp in params.all_compile_job_params
            )
        except KeyError:
            # This set of parameters is missing from the cache, we need to add them.
            previous_compile_jobs = (
                CompileScorecardJobYaml.from_intermediates().get_all_jobs(params)
            )
            current_compile_jobs = (
                CompileScorecardJobYaml.from_test_artifacts().get_all_jobs(params)
            )

            identical = True
            for (curr_params, curr_job), (prev_params, prev_job) in zip(
                previous_compile_jobs.items(),
                current_compile_jobs.items(),
                strict=False,
            ):
                assert curr_params == prev_params
                if (
                    not curr_job
                    or not prev_job
                    or not self.compile_jobs_are_same(curr_job.job, prev_job.job)
                ):
                    self.compile_jobs_are_identical[curr_params.compile_job_id] = False
                    identical = False

        return identical

    @classmethod
    def compile_jobs_are_same(
        cls,
        current_compile_job: hub.CompileJob,
        previous_compile_job: hub.CompileJob,
    ) -> bool:
        """
        Compare the MD5 hashes of the compiled models for two jobs.

        Parameters
        ----------
        current_compile_job
            The current compile job.
        previous_compile_job
            The previous compile job.

        Returns
        -------
        jobs_are_same : bool
            True if the MD5 hashes of the compiled models for the two jobs are the same, False otherwise.
        """
        if previous_compile_job.get_status().failure:
            return False

        if "qnn_lib_aarch64_android" in current_compile_job.options:
            # shared libraries do have stable md5 hashes between hub jobs
            return False

        if current_compile_job.get_status().failure:
            return False

        if ToolVersions.from_job(previous_compile_job) != ToolVersions.from_job(
            current_compile_job
        ):
            return False

        # The temporary directory and all its contents will be automatically cleaned up when the 'with' block is exited
        with qaihm_temp_dir() as tmp_dir:
            current_model_file = cast(
                str,
                current_compile_job.download_target_model(
                    os.path.join(tmp_dir, "current_model")
                ),
            )
            previous_model_file = cast(
                str,
                previous_compile_job.download_target_model(
                    os.path.join(tmp_dir, "previous_model")
                ),
            )

            # ONNX zip files from hub don't have stable hashes,
            # but the model.onnx and model.data _do_ have stable hashes.
            # Unzip if necessary, then compare each model file.
            current_model_files: list[os.PathLike | str] = []
            previous_model_files: list[os.PathLike | str] = []
            for model_file, model_files in [
                (current_model_file, current_model_files),
                (previous_model_file, previous_model_files),
            ]:
                if model_file.endswith(".onnx.zip"):
                    model_files.extend(
                        [x for x in extract_onnx_zip(model_file) if os.path.exists(x)]
                    )
                else:
                    model_files.append(model_file)

            # Compare the MD5 hashes of the compiled models
            return all(
                file_hashes_are_identical(current_model, previous_model)
                for current_model, previous_model in zip(
                    current_model_files, previous_model_files, strict=False
                )
            )
