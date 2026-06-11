# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, _patch, patch

import numpy as np
import pytest
import qai_hub as hub

from qai_hub_models import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.common import cleanup
from qai_hub_models.models._shared.llm.model import (
    LLM_AIMETOnnx,
    LLMBase,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    load_release_assets_for_model,
    update_perf_yaml,
)
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.utils.testing import patch_qai_hub
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.fetch_prerelease_assets import download_prerelease_asset
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.onnx.helpers import ONNXBundle

GENIE_BUNDLES_ROOT = "genie_bundles"


def _mock_from_pretrained(
    model_cls: type[LLM_AIMETOnnx], context_length: int, sequence_length: int
) -> Mock:
    model = MagicMock()
    model.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def from_bundle_path_patch(bundle_path: str | os.PathLike) -> ONNXBundle:
    return ONNXBundle(
        bundle_path=Path(bundle_path),
        onnx_graph_name="model.onnx",
        aimet_encodings_name="model.encodings",
    )


def split_onnx_patch(*args: Any, num_splits: int, **kwargs: Any) -> list[ONNXBundle]:
    return [from_bundle_path_patch(f"{i}") for i in range(num_splits)]


# reusable patching function
def _create_patches(
    model_cls: type[LLM_AIMETOnnx],
    base_name: str,
    context_length: int,
    sequence_length: int,
    tmp_path: Path,
) -> tuple[
    Mock,
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
]:
    mock_from_pretrained = _mock_from_pretrained(
        model_cls, context_length, sequence_length
    )
    mock_from_pretrained.return_value.sample_inputs.return_value = {
        "input0": [np.array([1.0, 2.0])]
    }
    # patching to not download from huggingface.
    patch_model = patch(
        f"qai_hub_models.models.{base_name}.Model.from_pretrained",
        mock_from_pretrained,
    )

    patch_fp_model = patch(
        f"qai_hub_models.models.{base_name}.FP_Model.from_pretrained",
        return_value=Mock(),
    )

    patch_onnx_checker = patch("onnx.checker.check_model")
    patch_onnx_load = patch("onnx.load")

    patch_split_onnx = patch(
        "qai_hub_models.models._shared.llm.split_onnx_utils.utils.split_onnx",
        side_effect=split_onnx_patch,
    )

    patch_onnx_files = patch.object(
        ONNXBundle, "from_bundle_path", side_effect=from_bundle_path_patch
    )

    patch_get_or_create_cached_model = patch(
        "qai_hub_models.models._shared.llm.export.get_or_create_cached_model",
        return_value=Mock(),
    )
    patch_tool_versions = patch(
        "qai_hub_models.configs.tool_versions.ToolVersions.from_job",
        return_value=ToolVersions(
            qairt=QAIRTVersion("2.34", validate_exists_on_ai_hub=False)
        ),
    )

    return (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    )


def test_cli_device_with_skips_unsupported_precision_device(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4A16",
        ]

        with pytest.raises(
            ValueError,
            match=r"The selected precision \(w4a16\) is not supported on this target device",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips_unsupported_context_length(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4",
        ]

        with pytest.raises(
            ValueError,
            match=rf"The {base_name}'s context length is too large to deploy on SA8295P\. Please set the context length to 1024 or lower\.",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if not skip_inferencing:
            sys.argv.append("--do-inferencing")
        if skip_profiling:
            sys.argv.append("--skip-profiling")

        export_main()  # Call the main function to submit the compile jobs

        # Compile is called parts * 2 times (num_splits token parts, num_splits prompt parts)
        assert mock_hub.submit_compile_job.call_count == parts * 2
        call_args_list = mock_hub.submit_compile_job.call_args_list

        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts

        call_args_list = mock_hub.submit_link_job.call_args_list

        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(call_args_list)
        )

        # Skip profile and inference combinations.
        if skip_inferencing:
            mock_hub.submit_inference_job.assert_not_called()
        else:
            mock_hub.submit_inference_job.assert_called()

        if skip_profiling:
            mock_hub.submit_profile_job.assert_not_called()
        else:
            mock_hub.submit_profile_job.assert_called()
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_multiple_context_lengths_link_jobs(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    """Link jobs should combine across both sequence and context lengths.

    With num_splits=N and M context lengths, there should still be exactly N
    link jobs (one per split), each combining all (seq_len, ctx_len) variants.
    """
    context_lengths = [1024, 4096]
    sequence_lengths = [1, 128]
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(
        model_cls, base_name, context_lengths[0], sequence_lengths[0], tmp_path
    )

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, max(context_lengths))
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            ",".join(str(cl) for cl in context_lengths),
            "--sequence-length",
            ",".join(str(sl) for sl in sequence_lengths),
            "--skip-profiling",
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]

        export_main()

        num_instantiations = len(context_lengths) * len(sequence_lengths)
        assert mock_hub.submit_compile_job.call_count == parts * num_instantiations

        # Regardless of how many context lengths are used, there must be exactly
        # `parts` link jobs — one per split, combining all (seq, ctx) variants.
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )

        # Each link job must receive compile-job outputs for every instantiation.
        for call in mock_hub.submit_link_job.call_args_list:
            linked_models = call.args[0]
            assert len(linked_models) == num_instantiations


def test_cli_chipset_with_options(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    chipset: str,
    context_length: int,
    sequence_length: int,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        compile_options = "compile_extra"
        profile_options = "profile_extra"
        link_options = "link_extra"

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",  # script name
            "--chipset",
            chipset,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--compile-options",
            compile_options,
            "--profile-options",
            profile_options,
            "--link-options",
            link_options,
            "--sequence-length",
            f"{sequence_length},1",
            "--context-length",
            str(context_length),
            "--target-runtime",
            target_runtime.value,
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
            "--do-inferencing",
        ]

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_profile_job.call_count == parts * 2
        assert mock_hub.submit_inference_job.call_count == parts * 2

        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            assert all(
                f"chipset:{chipset}" in call.kwargs["device"].attributes
                for call in mock.call_args_list
            )

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )
        mock_get_hub_link_options = (
            mock_from_pretrained.return_value.get_hub_link_options
        )
        assert mock_get_hub_link_options.call_count == parts
        assert all(
            call.args == (target_runtime, link_options)
            for call in mock_get_hub_link_options.call_args_list
        )

        mock_get_hub_compile_options = (
            mock_from_pretrained.return_value.get_hub_compile_options
        )

        assert mock_get_hub_compile_options.call_count == parts * 2
        for call in mock_get_hub_compile_options.call_args_list:
            assert call.args == (target_runtime, precision, compile_options)
            assert "context_graph_name" in call.kwargs
            assert (
                call.kwargs["context_graph_name"]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        # Profile parts * 2 times
        assert mock_hub.submit_profile_job.call_count == parts * 2
        mock_get_hub_profile_options = (
            mock_from_pretrained.return_value.get_hub_profile_options
        )
        assert mock_get_hub_profile_options.call_count == parts * 2
        for call in mock_get_hub_profile_options.call_args_list:
            assert len(call.args) == 3
            assert call.args[:2] == (target_runtime, profile_options)
            assert (
                call.args[2]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        assert mock_hub.submit_inference_job.call_count == parts * 2

        assert tmp_path.exists()
        assert tmp_path.is_dir()

        # TODO (#15224): Remove from_pretrained as part of inference?
        assert mock_from_pretrained.call_count == 4
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[1].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["sequence_length"]
            == sequence_length
        )
        # In instantiations list (160) from _shared/llm/export.py
        assert mock_from_pretrained.call_args_list[1].kwargs["sequence_length"] == 1


# for llama3 all components are tested, i.e. no option to select individual components 'part_1_of_5', 'part_2_of_5', 'part_3_of_5', 'part_4_of_5', 'part_5_of_5'
def test_cli_default_device_select_component(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
    decode_sequence_length: int,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    patch_torch_inference = patch("qai_hub_models.utils.compare.torch_inference")

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model as mock_get_or_create_cached_model,
        patch_torch_inference as mock_torch_inference,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        mock_torch_inference.return_value = [
            np.array([1.0, 2.0])
        ]  # return mock value for torch inference.

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--model-cache-mode",
            str(cache_mode.name.lower()),
            "--device",
            device.name,
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if skip_download:
            sys.argv.append("--skip-downloading")
        if skip_summary:
            sys.argv.append("--skip-summary")

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_link_job.call_count == parts
        assert mock_hub.submit_profile_job.call_count == parts * 2

        # Check names
        expected_model_name = f"{base_name}_{precision}"
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
        ]:
            for i, call in enumerate(mock.call_args_list):
                instantiation_name = (
                    f"ar{sequence_length}_cl{context_length}"
                    if i < parts
                    else f"ar{decode_sequence_length}_cl{context_length}"
                )
                assert (
                    call.kwargs["name"]
                    == f"{expected_model_name}_{instantiation_name}_{(i % parts) + 1}_of_{parts}"
                )

        assert mock_get_or_create_cached_model.call_count == parts * 2
        # Check cache mode is correct.
        for call in mock_get_or_create_cached_model.call_args_list:
            assert call.kwargs["model_name"] == expected_model_name
            assert call.kwargs["cache_mode"] == cache_mode

        # check compile jobs have correct device name.
        assert all(
            call.kwargs["device"].name == device.name
            for call in mock_hub.submit_compile_job.call_args_list
        )

        assert tmp_path.exists()
        assert tmp_path.is_dir()


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
    use_seq_mse: bool = False,
    use_dynamic_shapes: bool = False,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (
            (
                (Path(output_path) / "model_seqlen1_cl4096.onnx").exists()
                and (Path(output_path) / "model_seqlen128_cl4096.onnx").exists()
            )
            or (Path(output_path) / "model_dynamic.onnx").exists()
        )
    ):
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            allow_cpu_to_quantize=True,
            checkpoint=checkpoint,
            num_samples=num_samples,
            use_seq_mse=use_seq_mse,
            use_dynamic_shapes=use_dynamic_shapes,
        )
        cleanup()
    return output_path


# ---------------------------------------------------------------------------
# LLM performance collection helpers
# ---------------------------------------------------------------------------


def fetch_genie_bundle_for_perf(
    model_id: str,
    precision: Precision,
    chipset: str,
    output_dir: Path,
) -> Path:
    """Download and extract the pre-compiled genie bundle for this model.

    Looks up release-assets.yaml for (precision, chipset, GENIE), downloads
    the zip from S3, and extracts it into output_dir. Returns the extracted
    bundle directory.

    Raises a clear error if no matching asset exists.
    """
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIE)
    if asset is None:
        available_chipsets: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available_chipsets = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No genie release asset found in release-assets.yaml for "
            f"model_id={model_id!r}, precision={precision!s}, chipset={chipset!r}. "
            f"Available chipsets for this precision: {available_chipsets or '<none>'}. "
            "Build and update release-assets.yaml before running LLM perf collection."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIE, precision, chipset
    )
    if bundle_dir.exists():
        # Already fetched on a previous test in this session.
        return bundle_dir

    zip_path = download_prerelease_asset(
        asset,
        model_id=model_id,
        runtime=TargetRuntime.GENIE,
        precision=precision,
        chipset=chipset,
        output_folder=output_dir,
        verbose=True,
    )
    shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
    if not bundle_dir.exists():
        raise RuntimeError(
            f"Extracted genie bundle missing expected directory {bundle_dir}; "
            f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
        )
    return bundle_dir


def run_llm_perf_test(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path | str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Fetch the pre-compiled genie bundle, run QDC, update perf.yaml.

    The genie bundle is downloaded from S3 using release-assets.yaml.

    Returns (tokens_per_second, time_to_first_token_ms, prefill_tokens_per_second).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genie_bundle_path = fetch_genie_bundle_for_perf(
        model_id, precision, device.chipset, output_dir
    )

    metadata = ModelMetadata.from_json(genie_bundle_path / "metadata.json")
    assert metadata is not None and metadata.genie is not None
    context_lengths = metadata.genie.context_lengths

    # QDC run
    from qai_hub_models.utils.qdc.genie_jobs import (
        _USE_DEFAULT_PROMPTS,
        save_eval_results_json,
        submit_genie_bundle_to_qdc_device,
    )

    api_token = os.environ.get("QDC_API_TOKEN")
    if not api_token:
        raise ValueError("QDC_API_TOKEN environment variable is not set")

    run_eval = os.environ.get("QAIHM_RUN_EVAL", "true").lower() == "true"
    tps, prefill_tps, ttft, eval_results = submit_genie_bundle_to_qdc_device(
        api_token,
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=f"Genie {model_id} {precision}",
        qairt_sdk_path=qairt_sdk_path,
        eval_prompts=None if not run_eval else _USE_DEFAULT_PROMPTS,
        model_id=model_id,
    )

    # Update perf.yaml with only the max context length
    if not skip_perf_update and tps is not None and ttft is not None:
        update_perf_yaml(
            model_id,
            device.reference_device_name,
            precision,
            max(context_lengths),
            tps,
            ttft,
            prefill_tps,
        )

    # Save eval results as JSON in the working directory (not output_dir)
    # so the workflow artifact upload picks it up from $GITHUB_WORKSPACE.
    if eval_results:
        eval_json_path = Path(f"{model_id}_{device.chipset}_{precision}_eval.json")
        save_eval_results_json(eval_results, str(eval_json_path))

    return tps, ttft, prefill_tps
