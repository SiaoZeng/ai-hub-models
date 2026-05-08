# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from enum import Enum
from pathlib import Path

import qai_hub as hub

from qai_hub_models.scorecard.envvars import ArtifactsDirEnvvar
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

INTERMEDIATES_DIR = QAIHM_PACKAGE_ROOT / "scorecard" / "intermediates"


def test_artifacts_dir() -> Path:
    """Get the path in which all test artifacts are stored."""
    return ArtifactsDirEnvvar.get()


class ScorecardArtifact(Enum):
    # Results
    ACCURACY_CSV = "accuracy.csv"
    EXPORT_CSV = "export-summary.csv"
    RESULTS_CSV = "results.csv"

    # Cached State
    DATE = "date.txt"
    ENVIRONMENT_FILE = "environment.env"
    TOOL_VERSIONS = "tool-versions.yaml"
    QUANTIZE_YAML = "quantize-jobs.yaml"
    COMPILE_YAML = "compile-jobs.yaml"
    COMPILE_JOBS_IDENTICAL_CACHE = "compile-jobs-are-identical-cache.yaml"
    LINK_YAML = "link-jobs.yaml"
    PROFILE_YAML = "profile-jobs.yaml"
    INFERENCE_YAML = "inference-jobs.yaml"
    RELEASE_ASSETS = "release-assets.yaml"
    DATASET_IDS = "dataset-ids.yaml"
    CPU_ACCURACY = "cpu-accuracy.yaml"

    def touch(self) -> Path:
        """Get the path for this test artifact. Will touch() the artifact if it does not exist."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    @property
    def path(self) -> Path:
        """Get the path for this test artifact."""
        return test_artifacts_dir() / self.value

    def exists(self) -> bool:
        """Returns true if the artifact exists and is non-empty."""
        path = self.path
        return path.exists() and path.stat().st_size > 0

    @property
    def intermediates_path(self) -> Path:
        """Get the path for this artifact in the checked-in scorecard intermediates."""
        return INTERMEDIATES_DIR / self.value


def get_async_test_job_cache_artifact(job_type: hub.JobType) -> ScorecardArtifact:
    """
    Loads the appropriate Scorecard job cache for the type of the given job.

    Parameters;
        job_type: hub.JobType
            Job Type
    """
    if job_type == hub.JobType.COMPILE:
        return ScorecardArtifact.COMPILE_YAML
    if job_type == hub.JobType.PROFILE:
        return ScorecardArtifact.PROFILE_YAML
    if job_type == hub.JobType.INFERENCE:
        return ScorecardArtifact.INFERENCE_YAML
    if job_type == hub.JobType.QUANTIZE:
        return ScorecardArtifact.QUANTIZE_YAML
    if job_type == hub.JobType.LINK:
        return ScorecardArtifact.LINK_YAML
    raise NotImplementedError(
        f"No file for storing test jobs of type {job_type.display_name}"
    )
