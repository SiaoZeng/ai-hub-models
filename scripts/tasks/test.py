# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import glob
import os
import subprocess
import sys
from collections.abc import Iterable
from tempfile import TemporaryDirectory

from .changes import get_all_models, get_changed_files_in_package
from .constants import (
    BUILD_ROOT,
    PY_PACKAGE_INSTALL_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_SRC_ROOT,
    REPO_ROOT,
    STORE_ROOT_ENV_VAR,
)
from .task import (
    CompositeTask,
    PyTestTask,
    RunCommandsTask,
    Task,
)
from .util import (
    can_support_aimet,
    check_code_gen_field,
    check_info_field,
    get_is_hub_quantized,
    get_model_python_version_requirements,
    get_requires_aot_prepare,
    is_quantized_llm_model,
    model_needs_aimet,
    on_ci,
)
from .venv import (
    CreateVenvTask,
    InstallGlobalRequirementsTask,
    RunCommandsWithVenvTask,
    SyncLocalQAIHMVenvTask,
    SyncModelRequirementsVenvTask,
    SyncModelVenvTask,
)


def _model_has_nightly_tests(model_name: str) -> bool:
    """Return True if pytest can collect any nightly-marked tests for this model."""
    test_path = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "test.py")
    if not os.path.exists(test_path):
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "nightly",
            test_path,
        ],
        check=False,
        capture_output=True,
    )
    # Exit code 5 = no tests collected → no nightly tests for this model.
    # Exit code 0 = tests found; exit code 2 = collection error (import failure
    # because deps aren't installed yet) — treat conservatively as "has tests".
    return result.returncode != 5


def _model_has_llm_perf_tests(model_name: str) -> bool:
    """Return True if a model has model_type_llm in info.yaml and llm_perf-marked tests in test.py."""
    if not check_info_field(model_name, "model_type_llm"):
        return False
    test_path = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "test.py")
    if not os.path.exists(test_path):
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "llm_perf",
            test_path,
        ],
        check=False,
        capture_output=True,
    )
    # Exit code 5 = no tests collected → no llm_perf tests for this model.
    # Exit code 0 = tests found; exit code 2 = collection error (import failure
    # because deps aren't installed yet) — treat conservatively as "has tests".
    return result.returncode != 5


class PyTestQAIHMTask(PyTestTask):
    """Pytest utils."""

    def __init__(self, venv: str | None) -> None:
        all_dirs_except_models = [
            f"{PY_PACKAGE_SRC_ROOT}/{x}"
            for x in os.listdir(PY_PACKAGE_SRC_ROOT)
            if x not in {"models", "__pycache__", "scorecard"}
        ]

        # Static scorecard tests are expensive (calls to Hub), so only run them if the static scorecard changes.
        scorecard_files = [
            os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard", x)
            for x in os.listdir(os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard"))
        ]

        if not get_changed_files_in_package("src/qai_hub_models/scorecard/static"):
            scorecard_files.remove(f"{PY_PACKAGE_SRC_ROOT}/scorecard/static")
        all_dirs_except_models.extend(scorecard_files)

        all_dirs_except_models = [x for x in all_dirs_except_models if os.path.isdir(x)]

        # Get JUnit XML path from environment variable
        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")

        super().__init__(
            "Test QAIHM",
            venv=venv,
            files_or_dirs=" ".join(all_dirs_except_models),
            parallel=True,
            junit_xml_path=junit_xml_path,
        )


class GPUPyTestModelsTask(CompositeTask):
    """Run all tests for LLM models."""

    def __init__(
        self,
        venv: str | None,
        model_names: str = "all",  # Comma-separated list of model names, or "all" for all models.
        run_evaluate: bool = True,
        run_compile: bool = True,
        run_qdc: bool = True,
        run_demo: bool = True,
        raise_on_failure: bool = True,
        nightly_only: bool = False,  # If True, only run tests marked with @pytest.mark.nightly
    ) -> None:
        home_dir = os.path.expanduser("~")
        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")
        tmp_dir = os.environ.get("TMPDIR") or os.path.join(home_dir, "tmp")

        models_to_test = []
        for model_name in get_all_models():
            if (
                os.path.exists(
                    os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "test.py")
                )
                and (is_quantized_llm_model(model_name))
                and not check_code_gen_field(model_name, "skip_hub_tests_and_scorecard")
            ):
                models_to_test.append(model_name)
        if model_names != "all":
            requested = {m.strip() for m in model_names.split(",") if m.strip()}
            models_to_test = [m for m in models_to_test if m in requested]
        tasks = []
        common_command = f"mkdir -p {tmp_dir}"
        for model_name in models_to_test:
            base_dir = os.path.dirname(junit_xml_path)
            filename_parts = os.path.splitext(os.path.basename(junit_xml_path))
            test_suites = []
            if run_evaluate:
                test_suites.append("evaluate")
            if run_compile:
                test_suites.append("compile_ram_intensive")
            if run_qdc:
                test_suites.append("qdc")
            if run_demo:
                test_suites.append("demo")

            # When running nightly-only, skip models that have no nightly-marked tests.
            if nightly_only and not _model_has_nightly_tests(model_name):
                continue

            # Create a per-model virtual environment to isolate dependencies
            # (mirrors the scorecard approach to avoid cross-model dep conflicts).
            model_venv = os.path.join(home_dir, "model_envs", model_name)
            tasks.append(CreateVenvTask(model_venv))
            tasks.append(
                SyncModelVenvTask(
                    model_name,
                    model_venv,
                    include_dev_deps=True,
                )
            )

            # Install QDC wheel and optional GPU-specific requirements into the model venv.
            qdc_wheel_glob = os.path.join(REPO_ROOT, "qualcomm_device_cloud_sdk-*.whl")
            has_gpu_reqs = os.path.exists(
                os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "requirements-gpu.txt")
            )
            gpu_req_rel_path = (
                f"src/qai_hub_models/models/{model_name}/requirements-gpu.txt"
            )
            install_cmds = [f"pip install $(ls {qdc_wheel_glob})"]
            if has_gpu_reqs:
                # onnxruntime and onnxruntime-gpu share files in the onnxruntime/
                # namespace, so uninstalling one deletes files the other still
                # claims to own. On reused venvs this leaves onnxruntime-gpu
                # half-broken (e.g. `GraphOptimizationLevel` missing). Clean
                # both, install reqs, then uninstall the transitively re-pulled
                # onnxruntime and force-reinstall onnxruntime-gpu to repair
                # any shared files that got removed.
                install_cmds.append("pip uninstall -y onnxruntime onnxruntime-gpu")
                install_cmds.append(f"pip install -r {gpu_req_rel_path}")
                # aimet_onnx transitively re-installs onnxruntime via onnxruntime-extensions
                install_cmds.append("pip uninstall -y onnxruntime")
                install_cmds.append(
                    f"pip install --force-reinstall --no-deps "
                    f"$(grep '^onnxruntime-gpu' {gpu_req_rel_path})"
                )
            tasks.append(
                RunCommandsWithVenvTask(
                    group_name=f"Install GPU Dependencies For Model {model_name}",
                    venv=model_venv,
                    commands=[" && ".join(install_cmds)],
                    raise_on_failure=False,
                    ignore_return_codes=[5],
                    retries=2,
                )
            )

            for test_suite in test_suites:
                model_filename = (
                    f"{filename_parts[0]}-{test_suite}-{model_name}{filename_parts[1]}"
                )
                model_junit_xml_path = os.path.join(base_dir, model_filename)
                marker_expr = (
                    f"{test_suite} and nightly" if nightly_only else test_suite
                )
                options = f"-m '{marker_expr}' --junit-xml={model_junit_xml_path}"
                # Isolate each suite in a single spawned xdist worker to contain
                # CUDA OOM/leaks/crashes (issue #19607). Spawned (not forked)
                # workers avoid the CUDA re-init error; -n1 keeps module-scoped
                # fixtures cached; --max-worker-restart replaces a crashed worker
                # so remaining tests still run.
                isolation = "-n1 --dist load --max-worker-restart=4"
                # Reclaim the GPU from any orphaned worker left by a prior suite.
                guard = f"python {os.path.join('scripts', 'gpu_guard.py')}"
                tasks.append(
                    RunCommandsWithVenvTask(
                        group_name=f"Run {test_suite} Tests For Model {model_name}",
                        venv=model_venv,
                        commands=[
                            f"{common_command} && {guard} && pytest -v -s --capture=no {isolation} src/qai_hub_models/models/{model_name}/test.py {options}",
                        ],
                        raise_on_failure=False,
                        # Ignore "no tests collected" return code
                        ignore_return_codes=[5],
                    )
                )

            # Free disk between models: venv + HF cache + QAIHM store + TMPDIR.
            tasks.append(
                RunCommandsTask(
                    f"Cleanup After Model {model_name}",
                    f"rm -rf {model_venv}"
                    f" {home_dir}/.cache/huggingface/hub/models--*"
                    f" {home_dir}/.qaihm/models/*"
                    f" {tmp_dir}/*",
                )
            )

        super().__init__(
            # If a group name is used here, you get two groups per model
            # printed to console when running these tasks, one of which is empty.
            None,
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )


class PyTestModelTask(CompositeTask):
    """Run all tests for a single model."""

    def __init__(
        self,
        model_name: str,
        python_executable: str,
        venv: (
            str | None
        ) = None,  # If None, creates a fresh venv for each model instead of using 1 venv for all models.
        use_shared_cache: bool = False,  # If True, uses a shared cache rather than the global QAIHM cache.
        run_mypy: bool = False,
        run_general: bool = True,
        run_pre_quantize_compile: bool = False,
        run_link: bool = False,
        run_quantize: bool = False,
        run_compile: bool = True,
        run_profile: bool = False,
        run_inference: bool = False,
        run_compute_device_accuracy: bool = False,
        run_export: bool = False,
        run_llm_export: bool = False,
        run_trace: bool = True,
        install_deps: bool = True,
        raise_on_failure: bool = False,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
    ) -> None:
        tasks = []

        model_version_reqs = get_model_python_version_requirements(model_name)
        current_py_version = sys.version_info

        if model_version_reqs[0] and current_py_version < model_version_reqs[0]:
            tasks.append(  # greater than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too old (must be at least {model_version_reqs[0]})"',
                )
            )
        elif model_version_reqs[1] and current_py_version >= model_version_reqs[1]:
            tasks.append(  # less than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too new (must be less than {model_version_reqs[1]})"',
                )
            )
        elif model_needs_aimet(model_name) and not can_support_aimet():
            tasks.append(
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- AIMET is required, but AIMET is not supported on this platform."',
                )
            )
        else:
            # Create test environment
            needs_model_venv = venv is None
            setup_task: Task | None = None
            if needs_model_venv:
                model_venv = os.path.join(BUILD_ROOT, "test", "model_envs", model_name)
                tasks.append(CreateVenvTask(model_venv, python_executable))
                # Creates a new environment from scratch
                setup_task = SyncModelVenvTask(
                    model_name,
                    model_venv,
                    include_dev_deps=True,
                    qaihm_wheel_dir=qaihm_wheel_dir,
                    cli_wheel_dir=cli_wheel_dir,
                    junit_xml_path=junit_xml_path,
                )
                tasks.append(setup_task)
            else:
                model_venv = venv
                if install_deps:
                    # Only install requirements.txt into existing venv
                    tasks.append(
                        SyncModelRequirementsVenvTask(
                            model_name, model_venv, pip_force_install=False
                        )
                    )

            if check_code_gen_field(model_name, "skip_hub_tests_and_scorecard"):
                tasks.append(  # greater than this python version
                    RunCommandsTask(
                        f"Skip Model {model_name} Hub Tests",
                        f'echo "Skipping Tests For Model {model_name} -- skip_hub_tests_and_scorecard is set in code gen"',
                    )
                )
            elif (
                check_code_gen_field(model_name, "skip_scorecard") and not run_general
            ):  # For scorecard runs, run_general is set to False because it is a test_compile_all_models task rather than a precheckin task with hub tests.
                tasks.append(  # greater than this python version
                    RunCommandsTask(
                        f"Skip Model {model_name} Scorecard",
                        f'echo "Skipping Scorecard For Model {model_name} -- skip_scorecard is set in code gen"',
                    )
                )
            else:
                # Extras arguments
                extras_args = ["-s"]

                # Generate flags
                test_flags = []
                if run_general:
                    test_flags.append("unmarked")
                if run_pre_quantize_compile:
                    test_flags.append("pre_quantize_compile")
                if run_link:
                    test_flags.append("link")
                if run_compile:
                    test_flags.append("compile")
                if run_profile:
                    test_flags.append("profile")
                if run_inference:
                    test_flags.append("inference")
                if run_quantize:
                    test_flags.append("quantize")
                if run_trace:
                    test_flags.append("trace")
                if run_compute_device_accuracy:
                    test_flags.append("compute_device_accuracy")
                if run_export:
                    test_flags.append("export")
                if run_llm_export:
                    test_flags.append("llm_export")
                if test_flags:
                    extras_args += ["-m", f'"{" or ".join(test_flags)}"']

                    # Create temporary directory for storing cloned & downloaded test artifacts.
                    with TemporaryDirectory() as tmpdir:
                        env = os.environ.copy()
                        if not use_shared_cache:
                            env[STORE_ROOT_ENV_VAR] = tmpdir

                        # Standard Test Suite
                        model_dir = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name)
                        model_test = os.path.join(model_dir, "test.py")
                        generated_model_test = os.path.join(
                            model_dir, "test_generated.py"
                        )

                        if os.path.exists(model_test) or os.path.exists(
                            generated_model_test
                        ):
                            tasks.append(
                                PyTestTask(
                                    group_name=f"Model Tests: {model_name}",
                                    venv=model_venv,
                                    files_or_dirs=model_dir,
                                    parallel=False,
                                    extra_args=" ".join([*extras_args, "--no-header"]),
                                    env=env,
                                    raise_on_failure=not needs_model_venv,  # Do not raise on failure if a model venv was created, to make sure the venv is removed when the test finishes
                                    ignore_no_tests_return_code=True,
                                    include_pytest_cmd_in_status_message=False,
                                    junit_xml_path=junit_xml_path,
                                    prereqs=[setup_task] if setup_task else None,
                                )
                            )

                if run_mypy:
                    tasks.append(
                        RunCommandsWithVenvTask(
                            f"MyPy: {model_name}",
                            model_venv,
                            # MyPy errors on "unused #type: ignore" from unrelated model code, if run in a non-global environment.
                            # Therefore we run mypy only on the specific model folder for specific model environments.
                            [
                                f'cd "{PY_PACKAGE_INSTALL_ROOT}" && mypy --warn-unused-configs --config-file="{PY_PACKAGE_INSTALL_ROOT}/pyproject.toml" --package qai_hub_models.models.{model_name}'
                            ],
                            junit_xml_path=junit_xml_path,
                            junit_testsuite="pytest",
                            junit_name="mypy",
                            junit_classname=f"qai_hub_models.models.{model_name}",
                            raise_on_failure=False,
                            prereqs=[setup_task] if setup_task else None,
                        )
                    )

            if not venv:
                tasks.append(
                    RunCommandsTask(
                        f"Remove virtual environment at {model_venv}",
                        f"rm -rf {model_venv}",
                    )
                )

        super().__init__(
            # If a group name is used here, you get two groups per model
            # printed to console when running these tasks, one of which is empty.
            None,
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
            show_subtasks_in_failure_message=False,
        )


class PyTestModelsTask(CompositeTask):
    """Run tests for the provided set of models."""

    def __init__(
        self,
        python_executable: str,
        models_for_testing: Iterable[str],
        models_to_test_export: Iterable[str],
        base_test_venv: str | None = None,  # Env with QAIHM installed
        venv_for_each_model: bool = True,  # Create a fresh venv for each model instead of using the base test venv instead.
        use_shared_cache: bool = False,  # Use the global QAIHM cache rather than a temporary one for tests.
        test_trace: bool = True,
        run_mypy: bool = False,  # Only run mypy for model folders that don't use global requirements. Global reqs are covered by the precommit.
        run_general: bool = True,
        run_export_pre_quantize_compile: bool = False,
        run_export_link: bool = False,
        run_export_quantize: bool = False,
        run_export_compile: bool = True,
        run_export_profile: bool = False,
        run_export_inference: bool = False,
        run_compute_device_accuracy: bool = False,
        run_full_export: bool = False,
        run_llm_export: bool = False,
        exit_after_single_model_failure: bool = False,
        raise_on_failure: bool = True,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = os.environ.get("QAIHM_JUNIT_XML_PATH"),
    ) -> None:
        self.exit_after_single_model_failure = exit_after_single_model_failure

        if len(models_for_testing) == 0 and len(models_to_test_export) == 0:
            super().__init__("All Per-Model Tests (Skipped)", [])
            return
        tasks = []

        if not on_ci():
            for run_test, job_type in [
                (run_export_quantize, "quantize"),
                (run_export_compile, "compile"),
                (run_export_profile, "profile"),
                (run_export_inference, "inference"),
            ]:
                if run_test:
                    # Clean previous cached test jobs.
                    filename = os.getenv(
                        "QAIHM_TEST_ARTIFACTS_DOR",
                        os.path.join(
                            os.getcwd(), "qaihm_test_artifacts", f"{job_type}-jobs.yaml"
                        ),
                    )
                    tasks.append(
                        RunCommandsTask(
                            "Delete stored compile jobs from past test runs.",
                            f'if [ -f "{filename}" ]; then rm "{filename}"; fi',
                        )
                    )

        has_venv = base_test_venv is not None
        if not has_venv:
            # Create Venv
            base_test_venv = os.path.join(BUILD_ROOT, "test", "base_venv")
            tasks.append(CreateVenvTask(base_test_venv, python_executable))
            tasks.append(
                SyncLocalQAIHMVenvTask(
                    base_test_venv,
                    ["dev"],
                    qaihm_wheel_dir=qaihm_wheel_dir,
                    cli_wheel_dir=cli_wheel_dir,
                )
            )

        print(f"Tests to be run for models: {models_for_testing}")
        global_models = set()
        if not venv_for_each_model:
            for model_name in models_for_testing:
                if not check_code_gen_field(
                    model_name, "global_requirements_incompatible"
                ):
                    global_models.add(model_name)

            if len(global_models) > 0:
                tasks.append(InstallGlobalRequirementsTask(base_test_venv))

        # Sort models for ease of tracking how far along the tests are.
        # Do reverse order because whisper is slow to compile, so trigger earlier.
        export_models = models_to_test_export
        hub_quantized_models = []
        nonhub_quantized_models = []
        aot_models = []
        for model in models_for_testing:
            if get_is_hub_quantized(model) and model in export_models:
                hub_quantized_models.append(model)
            else:
                nonhub_quantized_models.append(model)
            if get_requires_aot_prepare(model) and model in export_models:
                aot_models.append(model)

        if run_export_link:
            # Only run AOT models when running link tests
            models_to_run = aot_models
        elif run_export_quantize:
            models_to_run = hub_quantized_models
        else:
            # Run hub quantized models last to give quantize job time to complete
            models_to_run = nonhub_quantized_models + hub_quantized_models
        for model_name in models_to_run:
            # Run standard test suite for this model.
            is_global_model = model_name in global_models

            # Create a model-specific JUnit XML path if base path is provided
            model_junit_xml_path = None
            if junit_xml_path:
                base_dir = os.path.dirname(junit_xml_path)
                base_filename = os.path.basename(junit_xml_path)
                filename_parts = os.path.splitext(base_filename)
                model_filename = f"{filename_parts[0]}-{model_name}{filename_parts[1]}"
                model_junit_xml_path = os.path.join(base_dir, model_filename)

            tasks.append(
                PyTestModelTask(
                    model_name,
                    python_executable,
                    venv=base_test_venv if is_global_model else None,
                    use_shared_cache=use_shared_cache,
                    install_deps=not is_global_model,
                    run_trace=test_trace,
                    run_mypy=run_mypy and not is_global_model,
                    run_general=run_general,
                    run_pre_quantize_compile=run_export_pre_quantize_compile
                    and model_name in export_models,
                    run_link=run_export_link and model_name in aot_models,
                    run_quantize=run_export_quantize and model_name in export_models,
                    run_compile=run_export_compile and model_name in export_models,
                    run_profile=run_export_profile and model_name in export_models,
                    run_inference=run_export_inference and model_name in export_models,
                    run_compute_device_accuracy=run_compute_device_accuracy
                    and model_name in export_models,
                    run_export=run_full_export and model_name in export_models,
                    run_llm_export=run_llm_export and model_name in export_models,
                    # Do not raise on failure; let PyTestModelsTask::run_tasks handle this
                    raise_on_failure=False,
                    qaihm_wheel_dir=qaihm_wheel_dir,
                    cli_wheel_dir=cli_wheel_dir,
                    junit_xml_path=model_junit_xml_path,
                )
            )

        if run_export_compile and not has_venv:
            # Cleanup venv
            tasks.append(RunCommandsTask(base_test_venv, f"rm -rf {base_test_venv}"))

        super().__init__(
            "All Per-Model Tests",
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )

    def run_task(self) -> bool:
        result: bool = True
        for task in self.tasks:
            try:
                task_result = task.run()
            except Exception:
                task_result = False
            if not task_result:
                if (
                    isinstance(task, PyTestModelTask)
                    and self.exit_after_single_model_failure
                ):
                    self.tasks[-1].run()  # cleanup venv
                    break
                if not self.continue_after_single_task_failure:
                    break
            result = result and task_result
        return result


class GenerateTestSummaryTask(RunCommandsTask):
    """Generate a test failure summary from JUnit XML files."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        title: str = "Test",
        name: str = "Combined",
    ) -> None:
        """
        Initialize the task.

        Parameters
        ----------
        input_dir
            Directory containing JUnit XML files (searched recursively).
        output_dir
            Output directory for combined-junit.xml and summary.md.
        title
            Title for the summary.
        name
            Name for the test section.
        """
        combined_xml = f"{output_dir}/combined-junit.xml"
        summary_md = f"{output_dir}/summary.md"
        super().__init__(
            group_name="Generate Test Failure Summary",
            commands=[
                f'python scripts/combine_junit_xml.py --junit-xml="{input_dir}" --output="{combined_xml}"',
                f'python -m qai_hub_models.scripts.generate_test_summary --title="{title}" --name="{name}" --junit-xml="{combined_xml}" --output="{summary_md}"',
            ],
        )


class CollectLLMPerfTask(CompositeTask):
    """Task to collect LLM performance numbers (TPS/TTFT) via pytest.

    Creates a per-model virtual environment for each LLM, installs the model's
    dependencies (including QDC wheel and GPU requirements), runs pytest with the
    llm_perf marker, then removes the venv to free disk space.

    Configuration is passed via environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs, or "all"
    - QAIHM_TEST_DEVICES: Comma-separated device names
    - QAIRT_SDK_PATH: Path to QAIRT SDK zip
    - QDC_API_TOKEN: QDC API token

    Pre-compiled genie bundles are fetched from each model's
    release-assets.yaml.
    """

    def __init__(
        self,
        venv: str | None,
        raise_on_failure: bool = False,
    ) -> None:
        home_dir = os.path.expanduser("~")
        tmp_dir = os.environ.get("TMPDIR") or os.path.join(home_dir, "tmp")

        models_env = os.environ.get("QAIHM_LLM_MODELS", "all")
        if models_env.strip().lower() == "all":
            models_to_test = [
                model_name
                for model_name in get_all_models()
                if _model_has_llm_perf_tests(model_name)
                and is_quantized_llm_model(model_name)
            ]
        else:
            models_to_test = [m.strip() for m in models_env.split(",") if m.strip()]

        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")

        tasks = []
        qdc_wheel_glob = os.path.join(REPO_ROOT, "qualcomm_device_cloud_sdk-*.whl")
        common_command = (
            f"mkdir -p {tmp_dir}"
            f" && rm -rf {home_dir}/.cache/huggingface/hub/models--*"
            f" {home_dir}/.qaihm/models/* {tmp_dir}/*"
        )

        for model_name in models_to_test:
            model_venv = os.path.join(home_dir, "model_envs", model_name)

            # Create per-model venv and install QAIHM + model requirements.
            tasks.append(CreateVenvTask(model_venv))
            tasks.append(
                SyncModelVenvTask(
                    model_name,
                    model_venv,
                    include_dev_deps=True,
                )
            )

            # Install the QDC wheel so the perf test can submit QDC jobs. No GPU
            # / AIMET requirements are needed: this task no longer compiles, it
            # only fetches the pre-compiled genie bundle and runs it on device.
            tasks.append(
                RunCommandsWithVenvTask(
                    group_name=f"Install QDC SDK For Model {model_name}",
                    venv=model_venv,
                    commands=[f"pip install $(ls {qdc_wheel_glob})"],
                    raise_on_failure=False,
                    ignore_return_codes=[5],
                    retries=2,
                )
            )

            # Build per-model JUnit XML path (matches GPUPyTestModelsTask pattern).
            model_junit_xml_path = None
            if junit_xml_path:
                base_dir = os.path.dirname(junit_xml_path)
                filename_parts = os.path.splitext(os.path.basename(junit_xml_path))
                model_filename = (
                    f"{filename_parts[0]}-llm_perf-{model_name}{filename_parts[1]}"
                )
                model_junit_xml_path = os.path.join(base_dir, model_filename)

            # Set up environment and clear caches before running tests.
            tasks.append(
                RunCommandsWithVenvTask(
                    group_name=f"Set Up Environment For Model {model_name}",
                    venv=model_venv,
                    commands=[common_command],
                    raise_on_failure=False,
                )
            )

            # -n 3 matches QDC's 3-slot per-user cap; perf.yaml writes are
            # serialized via FileLock.
            tasks.append(
                PyTestTask(
                    group_name=f"Run LLM Perf Tests For Model {model_name}",
                    venv=model_venv,
                    files_or_dirs=f"src/qai_hub_models/models/{model_name}/test.py",
                    extra_args="-s -m 'llm_perf' -n 3",
                    junit_xml_path=model_junit_xml_path,
                    raise_on_failure=False,
                    ignore_no_tests_return_code=True,
                )
            )

            # Free disk between models: venv + HF cache + QAIHM store + TMPDIR.
            tasks.append(
                RunCommandsTask(
                    f"Cleanup After Model {model_name}",
                    f"rm -rf {model_venv}"
                    f" {home_dir}/.cache/huggingface/hub/models--*"
                    f" {home_dir}/.qaihm/models/*"
                    f" {tmp_dir}/*",
                )
            )

        super().__init__(
            "Collect LLM Performance Numbers",
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )


class GradeLLMResponsesTask(CompositeTask):
    """Grade on-device LLM eval responses with a HuggingFace grader model.

    Discovers ``*_eval.json`` files (written by the llm_perf eval tests) under
    ``search_dir`` and grades each one into a sibling ``*_grade.json`` via
    ``qai_hub_models.scripts.llm.grade_responses``. The grader needs
    ``transformers>=5.2``, which conflicts with the LLM source-repo pins in the
    main qaihm-build venv, so callers should pass a dedicated grader ``--venv``.
    """

    def __init__(
        self,
        venv: str | None,
        search_dir: str | None = None,
        raise_on_failure: bool = False,
    ) -> None:
        search_dir = search_dir or os.getcwd()
        eval_jsons = sorted(glob.glob(os.path.join(search_dir, "*_eval.json")))

        tasks: list[Task] = []
        if not eval_jsons:
            tasks.append(
                RunCommandsTask(
                    "Grade LLM Responses",
                    f'echo "No *_eval.json files found in {search_dir}; nothing to grade."',
                )
            )
        for eval_json in eval_jsons:
            base = os.path.splitext(os.path.basename(eval_json))[0]
            out_json = os.path.join(search_dir, f"{base}_grade.json")
            tasks.append(
                RunCommandsWithVenvTask(
                    group_name=f"Grade {os.path.basename(eval_json)}",
                    venv=venv,
                    commands=[
                        f'python -m qai_hub_models.scripts.llm.grade_responses "{eval_json}" --output-json "{out_json}"',
                    ],
                    raise_on_failure=False,
                )
            )

        super().__init__(
            "Grade LLM Responses",
            tasks,
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )
