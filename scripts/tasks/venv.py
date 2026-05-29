# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import os
import re
import subprocess
from collections.abc import Iterable

from .constants import (
    DEFAULT_PYTHON,
    DEV_REQUIREMENTS_PATH,
    GLOBAL_REQUIREMENTS_PATH,
    PY_CLI_INSTALL_ROOT,
    PY_PACKAGE_INSTALL_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    REPO_ROOT,
    REQUIREMENTS_PATH,
)
from .task import CompositeTask, RunCommandsTask, RunCommandsWithVenvTask
from .util import get_code_gen_str_field, get_pip, has_cuda_gpu, uv_installed


@functools.cache
def get_package_version(
    package_name: str, requirements_path: str = REQUIREMENTS_PATH
) -> str | None:
    """Extract version constraint for a package from a requirements file."""
    if not os.path.exists(requirements_path):
        return None

    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments
            line = line.split("#")[0].strip()
            # Match package name followed by version specifier
            match = re.match(rf"^{re.escape(package_name)}([<>=!~\[].*)$", line)
            if match:
                version = match.group(1)
                # Remove extras (e.g., diffusers[torch] -> diffusers)
                version = re.sub(r"\[.*?\]", "", version)
                # Remove environment markers (e.g., "; python_version>='3.8'")
                return version.split(";")[0].strip()
    return None


def get_torch_cpu_install_command(
    extras: Iterable[str] = (),
    search_global_reqs: bool = False,
    packages: list[str] | None = None,
) -> str | None:
    """
    Build a pip install command for torch packages from the CPU index.
    Returns None if no CUDA GPU is available or no packages need to be installed.

    Parameters
    ----------
    extras
        Extras to check for model-specific torch version requirements.
    search_global_reqs
        Whether to also check global requirements
    packages
        List of torch-related packages to check for version requirements.

    Returns
    -------
    str | None
        A pip install command for the torch CPU packages with appropriate versions, or None if no installation is needed.
        If a package is not found in any requirements file, it will be skipped and not included in the install command.
    """
    if has_cuda_gpu():
        return None

    # Check for torch package versions in the provided requirements files.
    # Requirements files that are later / last in the list take highest priority.
    reqfiles_to_check = [REQUIREMENTS_PATH]
    if search_global_reqs:
        reqfiles_to_check.append(GLOBAL_REQUIREMENTS_PATH)
    for extra in extras:
        if extra == "dev":
            reqfiles_to_check.append(DEV_REQUIREMENTS_PATH)
        else:
            model_requirements_path = os.path.join(
                PY_PACKAGE_MODELS_ROOT, extra, "requirements.txt"
            )
            if os.path.exists(model_requirements_path):
                reqfiles_to_check.append(model_requirements_path)

    # Get versions for each package.
    packages_to_versions: dict[str, str | None] = dict.fromkeys(
        packages or ["torch", "torchvision", "torchaudio", "torchcodec"]
    )
    for reqfile in reqfiles_to_check[::-1]:
        packages_to_versions = {
            package: version if version else get_package_version(package, reqfile)
            for package, version in packages_to_versions.items()
        }

    # Convert to version strings.
    pgk_install_strings = []
    for package, version in packages_to_versions.items():
        if version is not None:
            pgk_install_strings.append(f"'{package}{version}'")

    return (
        f"{get_pip()} install {' '.join(pgk_install_strings)} --index-url https://download.pytorch.org/whl/cpu"
        if pgk_install_strings
        else None
    )


class CreateVenvTask(RunCommandsTask):
    def __init__(self, venv_path: str, python_executable: str | None = None) -> None:
        super().__init__(
            f"Creating virtual environment at {venv_path}",
            f"source {REPO_ROOT}/scripts/util/env_create.sh --python={python_executable or DEFAULT_PYTHON} --venv={venv_path} --no-sync",
        )


def is_package_installed(package_name: str, venv_path: str | None = None) -> bool:
    if venv_path is not None:
        if not os.path.exists(venv_path):
            return False
        command = f'. {venv_path}/bin/activate && python -c "import {package_name}"'
    else:
        command = f'python -c "import {package_name}"'

    try:
        subprocess.check_call(command, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


class GenerateGlobalRequirementsTask(RunCommandsWithVenvTask):
    # Global requirements change based on the python version,
    # and should therefore be regenerated before running any model tests.
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Generate Global Requirements",
            venv,
            ["python -m qai_hub_models.scripts.generate_global_requirements"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class AggregateScorecardResultsTask(RunCommandsWithVenvTask):
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Aggregate Scorecard Results",
            venv,
            ["python -m qai_hub_models.scripts.aggregate_scorecard_results"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class DownloadQDCWheelTask(RunCommandsWithVenvTask):
    # Needed to run tests relying on QDC (e.g. Genie exports)
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Download QDC Wheel",
            venv,
            [
                f"bash {REPO_ROOT}/scripts/ci/download-qdc-wheel.sh '{REPO_ROOT}'",
            ],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class DownloadQAIRTAutoSDKTask(RunCommandsWithVenvTask):
    # Downloads the QAIRT SDK for automotive devices from S3 (requires AWS credentials)
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Download QAIRT Auto SDK",
            venv,
            ["python -m qai_hub_models.scripts.download_auto_qairt"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class InstallGlobalRequirementsTask(RunCommandsWithVenvTask):
    def __init__(self, venv_path: str | None) -> None:
        commands: list[str] = []

        if torch_cpu_cmd := get_torch_cpu_install_command(search_global_reqs=True):
            commands.append(torch_cpu_cmd)

        commands.append(f'{get_pip()} install -r "{GLOBAL_REQUIREMENTS_PATH}" ')

        super().__init__(
            group_name="Install Global Requirements",
            venv=venv_path,
            commands=commands,
        )


# Path to the LLM grader requirements file, relative to REPO_ROOT.
GRADER_REQUIREMENTS_PATH = os.path.join(
    REPO_ROOT,
    "src",
    "qai_hub_models",
    "models",
    "_shared",
    "llm",
    "requirements-grader.txt",
)


class InstallLLMGraderRequirementsTask(RunCommandsWithVenvTask):
    """Install LLM grader deps + qai_hub_models (editable, no deps).

    The grader requires ``transformers>=5.2``, which conflicts with the
    older transformers pins used by LLM source repos. Pass ``--venv`` to
    install into a dedicated grader venv rather than the default one.

    The editable ``qai_hub_models`` install (``--no-deps``) lets the grader
    CLI import its support modules without pulling in the full repo
    dependency tree.
    """

    def __init__(self, venv_path: str | None) -> None:
        super().__init__(
            group_name="Install LLM Grader Requirements",
            venv=venv_path,
            commands=[
                f'{get_pip()} install -r "{GRADER_REQUIREMENTS_PATH}"',
                f'{get_pip()} install --no-deps -e "{PY_PACKAGE_INSTALL_ROOT}"',
            ],
        )


class InstallCLITask(RunCommandsWithVenvTask):
    """Install the qai_hub_models_cli package from a wheel or as editable."""

    def __init__(
        self,
        venv_path: str | None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "install_cli",
        junit_classname: str = "qai_hub_models_cli",
    ) -> None:
        if cli_wheel_dir is not None:
            commands = [
                f"{get_pip()} install $(ls {cli_wheel_dir}/qai_hub_models_cli-*.whl)",
            ]
            install_method = "wheel"
        else:
            commands = [
                f'{get_pip()} install -e "{PY_CLI_INSTALL_ROOT}"',
            ]
            install_method = "editable"
        super().__init__(
            group_name=f"Install CLI ({install_method})",
            venv=venv_path,
            commands=commands,
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
        )


class SyncLocalQAIHMVenvTask(CompositeTask):
    """Sync the provided environment with local QAIHM and the provided extras."""

    def __init__(
        self,
        venv_path: str | None,
        extras: Iterable[str] = [],
        flags: str | None = None,
        pre_install: str | None = None,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
    ) -> None:
        extras_str = f"[{','.join(extras)}]" if extras else ""

        no_build_isolation = flags and (
            "--use-pep517" in flags or "--no-build-isolation" in flags
        )
        if flags is not None and uv_installed():
            # use pep 517 is default behavior for UV, and therefore is not a valid arg.
            flags = flags.replace("--use-pep517", "")
        if flags is not None and not uv_installed():
            # This flag disables the `--use-pep517` behavior for uv. This is the default for pip, and is not a valid pip arg.
            flags = flags.replace("--no-build-isolation", "")

        commands: list[str] = []

        if torch_cpu_cmd := get_torch_cpu_install_command(extras):
            commands.append(torch_cpu_cmd)

        if no_build_isolation and (qaihm_wheel_dir is None or cli_wheel_dir is None):
            # No build isolation means pypi/uv won't install the minimum build deps to build the AI Hub Models wheel.
            # Install them manually instead.
            commands.append(
                f"{get_pip()} install 'setuptools-scm>=9,<10' 'setuptools>=80'"
            )

        if pre_install:
            commands.append(f"{get_pip()} install {pre_install}")

        if qaihm_wheel_dir is not None:
            # Find wheel file and install it (use relative path to work in both local and CI)
            commands.append(
                f"{get_pip()} install $(ls {qaihm_wheel_dir}/qai_hub_models-*.whl){extras_str} {flags or ''}"
            )
            install_method = "wheel"
        else:
            # Local development: Use editable install
            commands.append(
                f'{get_pip()} install -e "{PY_PACKAGE_INSTALL_ROOT}{extras_str}" {flags or ""}'
            )
            install_method = "editable"

        super().__init__(
            group_name=f"Install QAIHM{extras_str} ({install_method})",
            tasks=[
                InstallCLITask(venv_path, cli_wheel_dir),
                RunCommandsWithVenvTask(
                    group_name=f"Install QAIHM{extras_str} ({install_method})",
                    venv=venv_path,
                    commands=commands,
                ),
            ],
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
        )


class SyncModelVenvTask(SyncLocalQAIHMVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras needed for the model_name."""

    def __init__(
        self,
        model_name: str,
        venv_path: str | None,
        include_dev_deps: bool = False,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
    ) -> None:
        extras = []
        if include_dev_deps:
            extras.append("dev")
        if os.path.exists(
            os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt")
        ):
            extras.append(model_name)

        super().__init__(
            venv_path,
            extras,
            get_code_gen_str_field(model_name, "pip_install_flags"),
            get_code_gen_str_field(model_name, "pip_pre_build_reqs"),
            qaihm_wheel_dir,
            cli_wheel_dir=cli_wheel_dir,
            junit_xml_path=junit_xml_path,
            junit_testsuite="pytest",
            junit_name="environment_setup",
            junit_classname=f"qai_hub_models.models.{model_name}",
        )


class SyncModelRequirementsVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with requirements from model_name's requirements.txt.
    Will not re-install QAI Hub Models. Intended for speeding up CI compared to building an entirely new env for each model.
    """

    def __init__(
        self, model_name: str, venv_path: str | None, pip_force_install: bool = True
    ) -> None:
        requirements_txt = os.path.join(
            PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt"
        )
        extra_flags = get_code_gen_str_field(model_name, "pip_install_flags")
        pre_install = get_code_gen_str_field(model_name, "pip_pre_build_reqs")
        if os.path.exists(requirements_txt):
            commands = [
                f'{get_pip()} install {"--force-reinstall" if pip_force_install else None} -r "{requirements_txt}" {extra_flags or ""}'
            ]
            if pre_install:
                commands.insert(0, f"{get_pip()} install {pre_install}")
        else:
            commands = []

        super().__init__(
            group_name=f"Install Model Requirements for {model_name}",
            venv=venv_path,
            commands=commands,
        )
