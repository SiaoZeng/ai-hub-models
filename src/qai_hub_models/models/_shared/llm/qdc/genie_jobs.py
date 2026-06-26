# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import re
import shutil
import statistics
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType
from transformers import AutoTokenizer

from qai_hub_models.models._shared.llm.model import LLMBase
from qai_hub_models.models._shared.llm.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    POLL_INTERVAL,
    QDCDevice,
    QDCJobs,
)

DEFAULT_LLM_SYSTEM_PROMPT = LLMBase.default_system_prompt

DEFAULT_EVAL_PROMPTS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "eval_prompts.json",
    )
)


def create_zip(zip_path: str, source_dir: str | os.PathLike) -> None:
    """Create a zip archive from source_dir at zip_path."""
    if isinstance(source_dir, os.PathLike):
        source_dir = str(source_dir)

    files_to_zip = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, source_dir)
            files_to_zip.append((file_path, arcname))

    # Use ZIP_STORED (no compression) for speed - the files are already compressed binaries
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for file_path, arcname in files_to_zip:
            zf.write(file_path, arcname)


def _prepare_eval_prompts_in_bundle(
    genie_bundle_path: str,
    prompts: list[str],
    model_id: str | None = None,
) -> None:
    """Write individual prompt files into the genie bundle's prompts/ directory.

    Uses the tokenizer from the bundle to apply the correct chat template.
    If the bundle tokenizer lacks a chat_template, loads the tokenizer from
    the model's HF repo instead.
    """
    tokenizer = AutoTokenizer.from_pretrained(genie_bundle_path)

    if not getattr(tokenizer, "chat_template", None) and model_id:
        model_module = importlib.import_module(f"qai_hub_models.models.{model_id}")
        hf_repo = getattr(model_module, "HF_REPO_NAME", None)
        if hf_repo:
            tokenizer = AutoTokenizer.from_pretrained(hf_repo)

    prompts_dir = os.path.join(genie_bundle_path, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_file = os.path.join(prompts_dir, f"prompt_{idx:03d}.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(formatted)  # type: ignore[arg-type, unused-ignore]


class GenieArtifactHandler(ABC):
    """Abstract base class for Genie artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
    ) -> str:
        """Create artifact bundle and return path to the zip file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class GenieAndroidArtifactHandler(GenieArtifactHandler):
    def __init__(self, test_script: str) -> None:
        self.test_script: str = test_script

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
    ) -> str:
        # Copy the test script
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        # Copy 'run_android.py' and rename it to 'test_appium.py' since pytest looks for files starting with 'test_'.
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", self.test_script),
            os.path.join(test_folder, "test_appium.py"),
        )

        # Replace the Hexagon and QAIRT version placeholders with actual values
        test_appium_path = os.path.join(test_folder, "test_appium.py")
        with open(test_appium_path, encoding="utf-8") as f:
            file_content = f.read()
        with open(test_appium_path, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("<<HEXAGON_VERSION>>", hexagon_version)
                .replace("<<QAIRT_VERSION>>", qairt_version)
                .replace("<<NUM_TRIALS>>", str(num_trials))
            )

        # Requirements
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "requirements.txt"),
            dest_dir,
        )

        # Bundle the Genie test content
        genie_folder = os.path.join(dest_dir, "genie_bundle")
        os.makedirs(genie_folder, exist_ok=True)
        shutil.copytree(genie_bundle_path, genie_folder, dirs_exist_ok=True)

        # Create zip in parent directory to avoid zipping the zip itself
        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieAutoArtifactHandler(GenieAndroidArtifactHandler):
    """Artifact handler for automotive (auto) devices.

    Extends the Android handler by bundling the QAIRT SDK into the artifact,
    since auto devices cannot download it at runtime.
    """

    def __init__(self, test_script: str, qairt_sdk_path: str) -> None:
        """
        Parameters
        ----------
        test_script
            Filename of the Appium/PyTest script to bundle (e.g., ``run_auto_android.py``).
        qairt_sdk_path
            Path to the QAIRT SDK zip file to bundle with the artifact.
            Must be an accessible, valid zip file.
        """
        super().__init__(test_script)
        if not os.path.isfile(qairt_sdk_path):
            raise FileNotFoundError(
                f"QAIRT SDK path '{qairt_sdk_path}' does not exist or is not a file. "
                "Please verify the --qairt-sdk-path argument."
            )
        self.qairt_sdk_path: str = qairt_sdk_path

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
    ) -> str:
        # Build the standard Android artifact first
        zip_path = super().create_artifact(
            curr_dirname,
            genie_bundle_path,
            dest_dir,
            hexagon_version,
            qairt_version,
            num_trials,
        )

        # Append the QAIRT SDK into the artifact zip under genie_bundle/
        print(
            f"[QDC] Adding QAIRT SDK from {self.qairt_sdk_path} to {zip_path}...",
            flush=True,
        )
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.write(
                self.qairt_sdk_path,
                arcname=os.path.join("genie_bundle", "qairt_sdk.zip"),
            )
        print("[QDC] QAIRT SDK addition to zip complete", flush=True)
        return zip_path


class GenieLinuxArtifactHandler(GenieArtifactHandler):
    """Artifact handler for Linux IoT devices (e.g., IQ9).
    Uses Bash test framework — no Appium wrapper needed.
    """

    @property
    def entry_script(self) -> str:
        return "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
    ) -> str:
        script_name = "run_linux.sh"
        # Copy the bash script directly into dest_dir
        script_dest = os.path.join(dest_dir, script_name)
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", script_name),
            script_dest,
        )

        # Replace the Hexagon and QAIRT version placeholders with actual values
        with open(script_dest, encoding="utf-8") as f:
            file_content = f.read()
        with open(script_dest, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{NUM_TRIALS}", str(num_trials))
            )

        # Bundle the Genie test content
        genie_folder = os.path.join(dest_dir, "genie_bundle")
        os.makedirs(genie_folder, exist_ok=True)
        shutil.copytree(genie_bundle_path, genie_folder, dirs_exist_ok=True)

        # Create zip in parent directory to avoid zipping the zip itself
        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieWindowsArtifactHandler(GenieArtifactHandler):
    @property
    def entry_script(self) -> str:
        return "C:\\Temp\\TestContent\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
    ) -> str:
        script_name = "run_windows.ps1"
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", script_name),
            dest_dir,
        )
        dest_script = os.path.join(dest_dir, script_name)
        shutil.copytree(genie_bundle_path, dest_dir, dirs_exist_ok=True)
        with open(dest_script, encoding="utf-8") as f:
            file_content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{NUM_TRIALS}", str(num_trials))
            )

        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieQDCJobs(QDCJobs):
    """
    QDC job handler for Genie workloads.

    Handles uploading Genie bundles and parsing performance metrics
    from Genie benchmark logs.
    """

    def _get_artifact_handler(
        self,
        qdc_device: QDCDevice,
        qairt_sdk_path: str | None = None,
    ) -> GenieArtifactHandler:
        """Get the appropriate artifact handler based on device platform.

        Parameters
        ----------
        qdc_device
            QDCDevice instance (passed to avoid redundant instantiation).
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.

        Returns
        -------
        genie_artifact_handler: GenieArtifactHandler
            Instance of the appropriate GenieArtifactHandler subclass.
        """
        if qdc_device.windows_platform:
            return GenieWindowsArtifactHandler()
        if qdc_device.iot_platform:
            return GenieLinuxArtifactHandler()
        if qdc_device.auto_platform:
            if qairt_sdk_path is None:
                raise ValueError(
                    "qairt_sdk_path is required for auto devices. "
                    "Please provide the path to the automotive QAIRT SDK zip file."
                )
            return GenieAutoArtifactHandler(
                test_script="run_auto_android.py", qairt_sdk_path=qairt_sdk_path
            )
        if qdc_device.mobile_platform:
            return GenieAndroidArtifactHandler(test_script="run_android.py")
        raise ValueError("Unsupported platform type for Genie artifact handler.")

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        genie_bundle_path: str,
        qairt_sdk_path: str | None = None,
        qairt_version: str = "2.45.40.260406",
        eval_prompts: list[str] | None = None,
        num_trials: int = 25,
        model_id: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Prepare and upload Genie artifacts for the job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        genie_bundle_path
            Directory path containing the genie bundle.
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.
        qairt_version
            QAIRT SDK version to download on-device (e.g. ``"2.45.40.260406"``).
        eval_prompts
            If provided, list of prompts to evaluate. Each prompt is formatted
            using the bundle's tokenizer and run sequentially on device.
        num_trials
            Number of profiling trials to run.
        model_id
            Model identifier used to load the HF tokenizer if the bundle
            tokenizer lacks a chat template.

        Returns
        -------
        job_artifacts: list[str]
            List of artifact IDs returned by QDC upload.
        entry_script: str | None
            Optional entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(qdc_device, qairt_sdk_path)

        bundle_path_to_use = genie_bundle_path
        temp_bundle_dir = None
        if eval_prompts:
            temp_bundle_dir = tempfile.mkdtemp(prefix="genie_eval_bundle_")
            shutil.copytree(genie_bundle_path, temp_bundle_dir, dirs_exist_ok=True)
            _prepare_eval_prompts_in_bundle(temp_bundle_dir, eval_prompts, model_id)
            bundle_path_to_use = temp_bundle_dir

        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_path = artifact_handler.create_artifact(
                    curr_dirname,
                    bundle_path_to_use,
                    tmpdirname,
                    qdc_device.hexagon_version,
                    qairt_version,
                    num_trials,
                )
                upload_response = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
        finally:
            if temp_bundle_dir:
                shutil.rmtree(temp_bundle_dir, ignore_errors=True)

        return [upload_response], artifact_handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute and print performance metrics from job logs.

        Parameters
        ----------
        job_log_files
            List of job log files retrieved from QDC.

        Returns
        -------
        avg_tokens_per_second : float | None
            Average tokens per second.
        min_time_to_first_token: float | None
            Minimum time to first token in ms.
        prefill_tokens_per_second : float | None
            Prefill (prompt-processing) tokens per second.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tps: list[float] = []
            ttft: list[float] = []
            prefill_tps: list[float] = []

            if job_log_files:
                for job_log in job_log_files:
                    target_path = os.path.join(
                        tmpdirname, "logs", f"{job_log.filename}.zip"
                    )
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    self.download_job_log_files(job_log.filename, target_path)

                    if "genie" in job_log.filename:
                        print("On device output (genie.log):")
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        genie_log_path = os.path.join(tmpdirname, "genie.log")
                        displayed = False
                        for encoding in ("utf-8", "utf-16", "utf-16-le"):
                            try:
                                with open(genie_log_path, encoding=encoding) as file:
                                    genie_content = file.read()
                                    print(genie_content)
                                    displayed = True
                                    break
                            except Exception:
                                pass
                        if not displayed:
                            print(f"Warning: Could not read {genie_log_path}")

                    if "profile" in job_log.filename:
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        profile_path = os.path.join(
                            tmpdirname, job_log.filename.split("/")[-1]
                        )
                        with open(profile_path, encoding="utf-8") as file:
                            file_content = json.loads(file.read())

                        components = file_content.get("components", [])
                        if (
                            isinstance(components, list)
                            and len(components) > 0
                            and isinstance(components[0], dict)
                            and "events" in components[0]
                            and isinstance(components[0]["events"], list)
                            and len(components[0]["events"]) > 1
                        ):
                            component = components[0]["events"][1]
                            tps.append(
                                float(component["token-generation-rate"]["value"])
                            )
                            ttft.append(
                                float(component["time-to-first-token"]["value"])
                            )
                            prefill_tps.append(
                                float(component["prompt-processing-rate"]["value"])
                            )
                        else:
                            print(
                                "Warning: Unexpected profile log structure, "
                                "skipping metrics for this file."
                            )

        if len(tps) > 0:
            # TTFT in profile logs is in microseconds, convert to milliseconds
            ttft_ms = [t / 1000.0 for t in ttft]

            print("Perf metrics:")
            print(f"  Tokens Per Second (all trials): {tps}")
            print(f"  Time to First Token ms (all trials): {ttft_ms}")
            print(f"  Prefill Tokens Per Second (all trials): {prefill_tps}")
            print(
                f"  Tokens Per Second — average: {statistics.mean(tps):.2f}, median: {statistics.median(tps):.2f}"
            )
            print(
                f"  Time to First Token (ms) — average: {statistics.mean(ttft_ms):.2f}, median: {statistics.median(ttft_ms):.2f}"
            )
            print(
                f"  Prefill Tokens Per Second — average: {statistics.mean(prefill_tps):.2f}, median: {statistics.median(prefill_tps):.2f}"
            )
            return (
                statistics.median(tps),
                statistics.median(prefill_tps),
                statistics.median(ttft_ms),
            )

        print("No performance metrics found.")
        return None, None, None

    @staticmethod
    def _parse_eval_outputs(content: str) -> dict[int, str]:
        """Parse a single eval_outputs.txt file with delimiter markers.

        Format: ===EVAL_IDX_NNN=== followed by the model output for that prompt.
        """
        outputs: dict[int, str] = {}
        parts = re.split(r"===EVAL_IDX_(\d+)===\n?", content)
        for i in range(1, len(parts) - 1, 2):
            idx = int(parts[i])
            outputs[idx] = parts[i + 1].strip()
        return outputs

    @staticmethod
    def _extract_model_output(raw_output: str) -> str:
        """Extract just the model's response from raw genie-t2t-run output.

        The raw output mixes debug logs, the chat-templated prompt echo, and
        the actual response between ``[BEGIN]:`` and ``[END]`` markers. We
        return only the text between those markers; if neither is present,
        fall back to the raw output stripped.
        """
        begin_marker = "[BEGIN]:"
        end_marker = "[END]"
        begin_idx = raw_output.find(begin_marker)
        if begin_idx == -1:
            return raw_output.strip()
        text = raw_output[begin_idx + len(begin_marker) :]
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
        return text.strip()

    def compute_eval_results(
        self,
        job_log_files: list,
        prompts: list[str],
    ) -> list[dict]:
        """Parse eval outputs from job logs.

        The device scripts write a single eval_outputs.txt file with
        delimiter markers (===EVAL_IDX_NNN===) separating each prompt's
        output.

        Parameters
        ----------
        job_log_files
            List of job log files retrieved from QDC.
        prompts
            Original list of prompts (used to attach prompt text to results).

        Returns
        -------
        results: list[dict]
            List of dicts with keys: idx, prompt, output.
        """
        outputs: dict[int, str] = {}

        with tempfile.TemporaryDirectory() as tmpdirname:
            for job_log in job_log_files:
                if "eval_outputs" not in job_log.filename:
                    continue

                target_path = os.path.join(
                    tmpdirname, "logs", f"{job_log.filename}.zip"
                )
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                self.download_job_log_files(job_log.filename, target_path)

                safe_root = pathlib.Path(tmpdirname).resolve()
                with zipfile.ZipFile(target_path) as zf:
                    for member in zf.namelist():
                        dest = (safe_root / member).resolve()
                        if not str(dest).startswith(str(safe_root) + os.sep):
                            raise ValueError(
                                f"Zip slip detected in log archive: {member}"
                            )
                    zf.extractall(safe_root)

                extracted_name = job_log.filename.split("/")[-1]
                extracted_path = os.path.join(tmpdirname, extracted_name)
                if not os.path.exists(extracted_path):
                    continue

                content = None
                for encoding in ("utf-8", "utf-16", "utf-16-le"):
                    try:
                        with open(extracted_path, encoding=encoding) as f:
                            content = f.read()
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        pass

                if content is None:
                    print(f"Warning: Could not decode {extracted_name}")
                    continue

                outputs = self._parse_eval_outputs(content)

        results: list[dict] = [
            {
                "idx": idx,
                "prompt": prompts[idx] if idx < len(prompts) else "",
                "output": self._extract_model_output(outputs.get(idx, "")),
            }
            for idx in sorted(outputs.keys())
        ]

        if not results:
            print("Warning: No eval results found in job logs.")
            print("Available log files:")
            for job_log in job_log_files:
                print(f"  {job_log.filename}")

        return results


def save_eval_results_json(results: list[dict], output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    if not results:
        print("No results to save.")
        return

    results.sort(key=lambda r: r.get("idx", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


_USE_DEFAULT_PROMPTS = object()

# Bounded retries for a QDC job that finishes with a non-"Successful" result;
# these device-side failures are usually transient (device availability/flakiness).
_QDC_EXECUTION_MAX_ATTEMPTS = 2


def submit_genie_bundle_to_qdc_device(
    api_token: str,
    device: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str = "2.45.40.260406",
    eval_prompts: list[str] | None | object = None,
    num_trials: int = 25,
    model_id: str | None = None,
) -> tuple[float | None, float | None, float | None, list[dict]]:
    """
    Submit a Genie bundle to QDC for execution on the specified device.

    Runs profiling and (optionally) evaluation in a single job. Eval is
    skipped by default; pass ``_USE_DEFAULT_PROMPTS`` for the built-in 100
    questions, or a list of prompts to use a custom set.

    Parameters
    ----------
    api_token
        API token for QDC authentication.
    device
        Hub device name to run the job on.
    genie_bundle_path
        Directory where genie files are stored. Must contain 'sample_prompt.txt'.
    job_name
        Name of QDC job.
    qairt_sdk_path
        Path to the QAIRT SDK zip file. Required for auto devices.
    qairt_version
        QAIRT SDK version to download on-device (e.g. ``"2.45.40.260406"``).
    eval_prompts
        Eval is off by default. Pass ``_USE_DEFAULT_PROMPTS`` to use the
        built-in eval_prompts.json (100 questions), or a list of prompts to
        evaluate a custom set.
    num_trials
        Number of profiling trials to run.
    model_id
        Model identifier used to load the HF tokenizer if the bundle
        tokenizer lacks a chat template.

    Returns
    -------
    tuple[float | None, float | None, float | None, list[dict]]
        (avg_tokens_per_second, prefill_tokens_per_second,
        min_time_to_first_token, eval_results)
        where eval_results is a list of dicts with keys: idx, prompt, output.
    """
    prompts_to_use: list[str] | None
    if eval_prompts is _USE_DEFAULT_PROMPTS:
        with open(DEFAULT_EVAL_PROMPTS_PATH, encoding="utf-8") as f:
            prompts_to_use = json.load(f)
    elif isinstance(eval_prompts, list):
        prompts_to_use = eval_prompts
    else:
        prompts_to_use = None

    qdc_device = QDCDevice(device)
    genie_job = GenieQDCJobs(
        api_key=api_token,
        app_name_header="GenieQDCJobApp",
    )

    job_artifacts, entry_script = genie_job.add_job_artifacts(
        qdc_device,
        genie_bundle_path,
        qairt_sdk_path,
        qairt_version,
        eval_prompts=prompts_to_use,
        num_trials=num_trials,
        model_id=model_id,
    )

    # A job can reach state "Completed" yet fail device-side (result "Unsuccessful"),
    # yielding no perf logs; retry the (usually transient) failure, then raise.
    last_failure_reason: str | None = None
    for attempt in range(1, _QDC_EXECUTION_MAX_ATTEMPTS + 1):
        job_id = genie_job.submit_automated_job(
            qdc_device, job_artifacts, entry_script, job_name=job_name
        )
        if job_id is None:
            raise RuntimeError("Job submission failed.")

        print(f"Submitted QDC job with ID: {job_id}")
        job_status = genie_job.status(job_id)
        job_result = genie_job.result(job_id)
        print(
            f"QDC job {job_id} completed with status: {job_status}, "
            f"result: {job_result}"
        )
        genie_job.log_upload_status(job_id)
        job_log_files = genie_job.get_job_log_files(job_id)
        time.sleep(POLL_INTERVAL)

        # Treat a non-"Successful" terminal result as a device-side failure.
        if job_result is not None and job_result != "Successful":
            last_failure_reason = (
                f"QDC job {job_id} on device '{device}' finished with "
                f"status='{job_status}', result='{job_result}'"
            )
            print(
                f"[attempt {attempt}/{_QDC_EXECUTION_MAX_ATTEMPTS}] "
                f"{last_failure_reason}"
            )
            if attempt < _QDC_EXECUTION_MAX_ATTEMPTS:
                print("Retrying QDC job execution...")
                continue
            raise RuntimeError(
                f"{last_failure_reason} after {_QDC_EXECUTION_MAX_ATTEMPTS} "
                f"attempt(s). The device-side job did not complete successfully; "
                f"check the QDC job logs for details."
            )

        tps, prefill_tps, ttft = genie_job.compute_metrics(job_log_files)

        eval_results: list[dict] = []
        if prompts_to_use:
            eval_results = genie_job.compute_eval_results(job_log_files, prompts_to_use)

        return tps, prefill_tps, ttft, eval_results

    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError(
        f"QDC job execution failed for device '{device}': {last_failure_reason}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-token",
        type=str,
        required=True,
        help="API token for authentication.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=HUB_DEVICE_TO_QDC_DEVICE_MAP.keys(),
        help="Device to use for the job.",
    )
    parser.add_argument(
        "--genie-bundle-path",
        type=str,
        required=True,
        help="Directory where genie files are stored.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        default="LLM Genie",
        help="QDC job name.",
    )
    parser.add_argument(
        "--qairt-sdk-path",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to QAIRT SDK zip file. Required when targeting automotive devices "
            "(e.g., SA8295P ADP, SA7255P ADP, SA8775P ADP). "
            "Omitting this for an auto device will raise a ValueError at job submission time."
        ),
    )
    parser.add_argument(
        "--qairt-version",
        type=str,
        required=False,
        default="2.45.40.260406",
        help="QAIRT SDK version to download on-device (e.g. 2.45.40.260406).",
    )
    parser.add_argument(
        "--eval-prompts",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to JSON file with list of prompt strings for evaluation. "
            "If not provided, uses the built-in eval_prompts.json (100 questions)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=False,
        default=None,
        help="Path to save eval results as JSON.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        required=False,
        default=25,
        help="Number of profiling trials to run (default: 25).",
    )

    args = parser.parse_args()

    eval_prompts = None
    if args.eval_prompts:
        with open(args.eval_prompts, encoding="utf-8") as f:
            eval_prompts = json.load(f)
        print(f"Loaded {len(eval_prompts)} eval prompts from {args.eval_prompts}")

    if not os.path.exists(os.path.join(args.genie_bundle_path, "sample_prompt.txt")):
        raise FileNotFoundError(
            f"sample_prompt.txt not found in {args.genie_bundle_path}. "
            "Please add a file with prompt to run on-device."
        )

    tps, prefill_tps, ttft, eval_results = submit_genie_bundle_to_qdc_device(
        args.api_token,
        args.device,
        args.genie_bundle_path,
        args.job_name,
        args.qairt_sdk_path,
        args.qairt_version,
        eval_prompts=eval_prompts,
        num_trials=args.num_trials,
    )

    if args.output_json:
        save_eval_results_json(eval_results, args.output_json)
