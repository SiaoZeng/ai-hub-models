# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.envvars import FORCE_MANIFEST_ROOT_ENVVAR
from qai_hub_models_cli.proto.info_pb2 import (
    ModelDomain,
    ModelInfo,
    ModelLicense,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.manifest_pb2 import (
    ManifestModelEntry,
    ModelManifestUrls,
    ReleaseManifest,
)
from qai_hub_models_cli.proto.numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    FormFactor,
    FormFactorInfo,
    PlatformInfo,
    RuntimeInfo,
)
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime

# ── Test fixtures ─────────────────────────────────────────────────────

_RELEASE_VERSION = Version("0.45.0")
_DEV_VERSION = Version("0.45.0.dev1")


def _manifest() -> ReleaseManifest:
    return ReleaseManifest(
        version="0.45.0",
        platform_url="https://example.com/platform.pb",
        models=[
            ManifestModelEntry(
                id="mobilenet_v2",
                display_name="MobileNet V2",
                domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
                manifest_urls=ModelManifestUrls(
                    info="https://example.com/mobilenet_v2/info.pb",
                    perf="https://example.com/mobilenet_v2/perf.pb",
                    numerics="https://example.com/mobilenet_v2/numerics.pb",
                    release_assets="https://example.com/mobilenet_v2/release_assets.pb",
                ),
            ),
            ManifestModelEntry(
                id="whisper_small",
                display_name="Whisper Small",
                domain=ModelDomain.MODEL_DOMAIN_AUDIO,
                manifest_urls=ModelManifestUrls(
                    info="https://example.com/whisper_small/info.pb",
                    perf="https://example.com/whisper_small/perf.pb",
                    numerics="https://example.com/whisper_small/numerics.pb",
                    release_assets="https://example.com/whisper_small/release_assets.pb",
                ),
            ),
        ],
    )


def _model_info() -> ModelInfo:
    return ModelInfo(
        id="mobilenet_v2",
        name="MobileNet V2",
        domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
        use_case=ModelUseCase.MODEL_USE_CASE_IMAGE_CLASSIFICATION,
        tags=[ModelTag.MODEL_TAG_REAL_TIME],
        license_type=ModelLicense.MODEL_LICENSE_APACHE_2_0,
    )


def _model_perf() -> ModelPerf:
    return ModelPerf(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        supported_devices=["Samsung Galaxy S24"],
        supported_chipsets=["qualcomm-snapdragon-8-gen-3"],
        performance_metrics=[
            ModelPerf.PerformanceDetails(
                precision=Precision.PRECISION_FLOAT,
                device="Samsung Galaxy S24",
                runtime=Runtime.RUNTIME_TFLITE,
                metrics=ModelPerf.PerfMetrics(
                    inference_time_milliseconds=2.5,
                    primary_compute_unit="NPU",
                ),
            ),
        ],
    )


def _model_numerics() -> ModelNumerics:
    return ModelNumerics(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        metrics=[
            ModelNumerics.NumericsMetric(
                dataset_name="imagenet",
                metric_name="Top-1 Accuracy",
                metric_unit="%",
                partial_torch_metric=71.8,
            ),
        ],
    )


def _model_release_assets() -> ModelReleaseAssets:
    return ModelReleaseAssets(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_TFLITE,
                download_url="https://example.com/mobilenet_v2-tflite-float.zip",
            ),
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_QNN_CONTEXT_BINARY,
                chipset="qualcomm-snapdragon-8-gen-3",
                download_url="https://example.com/mobilenet_v2-qnn-float-8g3.zip",
            ),
        ],
    )


def _platform_info() -> PlatformInfo:
    return PlatformInfo(
        aihm_version="0.45.0",
        runtimes=[
            RuntimeInfo(
                runtime=Runtime.RUNTIME_TFLITE,
                website_runtime="TFLite",
                file_extension=".tflite",
                is_aot_compiled=False,
            ),
            RuntimeInfo(
                runtime=Runtime.RUNTIME_QNN_CONTEXT_BINARY,
                website_runtime="QNN",
                file_extension=".bin",
                is_aot_compiled=True,
            ),
        ],
        form_factors=[
            FormFactorInfo(
                form_factor=FormFactor.FORM_FACTOR_PHONE,
                display_name="Phone",
            ),
        ],
        devices=[
            DeviceInfo(
                name="Samsung Galaxy S24",
                chipset="qualcomm-snapdragon-8-gen-3",
                form_factor=FormFactor.FORM_FACTOR_PHONE,
            ),
        ],
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-3",
                marketing_name="Snapdragon 8 Gen 3",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _clear_lru_caches() -> Generator[None, None, None]:
    from qai_hub_models_cli.proto_helpers.info import get_model_info
    from qai_hub_models_cli.proto_helpers.manifest import get_manifest
    from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics
    from qai_hub_models_cli.proto_helpers.perf import get_model_perf
    from qai_hub_models_cli.proto_helpers.platform import get_platform
    from qai_hub_models_cli.proto_helpers.release_assets import (
        get_model_release_assets,
    )

    fns = (
        get_manifest,
        get_model_info,
        get_model_perf,
        get_model_numerics,
        get_model_release_assets,
        get_platform,
    )
    for fn in fns:
        fn.cache_clear()
    yield
    for fn in fns:
        fn.cache_clear()


_skip_version_check = patch(
    "qai_hub_models_cli.proto_helpers._common.verify_version_supported"
)
_not_dev_release = patch(
    "qai_hub_models_cli.proto_helpers._common.use_aihm_source", return_value=False
)


# ── _common.py ────────────────────────────────────────────────────────


class TestCommon:
    def test_read_write_proto_roundtrip(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import read_proto, write_proto

        proto = _manifest()
        path = tmp_path / "manifest.pb"
        write_proto(path, proto)
        loaded = read_proto(path, ReleaseManifest)
        assert loaded.version == "0.45.0"
        assert len(loaded.models) == 2

    def test_get_release_cache_dir_envvar_override(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import get_release_cache_dir

        with patch.dict("os.environ", {FORCE_MANIFEST_ROOT_ENVVAR: str(tmp_path)}):
            assert get_release_cache_dir(Version("0.45.0")) == tmp_path

    def test_use_aihm_source_true_for_dev(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with (
            patch(
                "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
                Version("0.45.0.dev1"),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.importlib.util.find_spec",
                return_value=MagicMock(),
            ),
        ):
            assert use_aihm_source(Version("0.45.0.dev1")) is True

    def test_use_aihm_source_false_for_release(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with patch(
            "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
            Version("0.45.0"),
        ):
            assert use_aihm_source(Version("0.45.0")) is False

    def test_use_aihm_source_false_package_not_installed(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with (
            patch(
                "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
                Version("0.45.0.dev1"),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.importlib.util.find_spec",
                return_value=None,
            ),
        ):
            assert use_aihm_source(Version("0.45.0.dev1")) is False

    def test_fetch_proto_uses_cache(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import fetch_proto

        cache_path = tmp_path / "manifest.pb"
        cache_path.write_bytes(_manifest().SerializeToString())
        with patch("qai_hub_models_cli.proto_helpers._common.download") as mock_dl:
            result = fetch_proto(
                "https://example.com/manifest.pb", cache_path, ReleaseManifest
            )
        mock_dl.assert_not_called()
        assert result.version == "0.45.0"


# ── manifest.py ───────────────────────────────────────────────────────


class TestGetManifest:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        path = tmp_path / "manifest.pb"
        path.write_bytes(_manifest().SerializeToString())
        result = get_manifest(local_path=path)
        assert result.version == "0.45.0"
        assert len(result.models) == 2

    def test_fetches_from_s3(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_path = tmp_path / "releases" / "v0.45.0" / "manifest.pb"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(_manifest().SerializeToString())

        with (
            _skip_version_check,
            _not_dev_release,
            patch(
                "qai_hub_models_cli.proto_helpers._common.get_release_cache_dir",
                return_value=tmp_path / "releases" / "v0.45.0",
            ),
        ):
            result = get_manifest(_RELEASE_VERSION)
        assert result.version == "0.45.0"


class TestGetManifestEntry:
    def test_lookup_by_id(self) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry

        with patch(
            "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
            return_value=_manifest(),
        ):
            entry = get_manifest_entry("mobilenet_v2", _RELEASE_VERSION)
        assert entry.id == "mobilenet_v2"
        assert entry.display_name == "MobileNet V2"

    def test_not_found(self) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_manifest_entry("nonexistent_model", _RELEASE_VERSION)


# ── info.py ───────────────────────────────────────────────────────────


class TestGetModelInfo:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        path = tmp_path / "info.pb"
        path.write_bytes(_model_info().SerializeToString())
        result = get_model_info("mobilenet_v2", local_path=path)
        assert result.id == "mobilenet_v2"
        assert result.name == "MobileNet V2"

    def test_fetches_from_s3(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        with (
            _not_dev_release,
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=False,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.fetch_proto",
                return_value=_model_info(),
            ),
        ):
            result = get_model_info("mobilenet_v2", _RELEASE_VERSION)
        assert result.id == "mobilenet_v2"

    def test_not_found_raises(self) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_model_info("fake_model", _RELEASE_VERSION)


# ── perf.py ───────────────────────────────────────────────────────────


class TestGetModelPerf:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.perf import get_model_perf

        path = tmp_path / "perf.pb"
        path.write_bytes(_model_perf().SerializeToString())
        result = get_model_perf("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert result.performance_metrics[0].metrics.inference_time_milliseconds == 2.5

    def test_not_found_raises(self) -> None:
        from qai_hub_models_cli.proto_helpers.perf import get_model_perf

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_model_perf("fake_model", _RELEASE_VERSION)


# ── numerics.py ───────────────────────────────────────────────────────


class TestGetModelNumerics:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics

        path = tmp_path / "numerics.pb"
        path.write_bytes(_model_numerics().SerializeToString())
        result = get_model_numerics("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert result.metrics[0].dataset_name == "imagenet"

    def test_not_found_raises(self) -> None:
        from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_model_numerics("fake_model", _RELEASE_VERSION)


# ── release_assets.py ─────────────────────────────────────────────────


class TestGetModelReleaseAssets:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_release_assets,
        )

        path = tmp_path / "release_assets.pb"
        path.write_bytes(_model_release_assets().SerializeToString())
        result = get_model_release_assets("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert len(result.assets) == 2

    def test_not_found_raises(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_release_assets,
        )

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_model_release_assets("fake_model", _RELEASE_VERSION)


class TestGetModelAssetDetails:
    def test_universal_lookup(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_model_release_assets",
                return_value=_model_release_assets(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_runtime_info",
                return_value=RuntimeInfo(is_aot_compiled=False),
            ),
        ):
            asset = get_model_asset_details(
                "mobilenet_v2", "tflite", "float", version=_RELEASE_VERSION
            )
        assert asset.runtime == Runtime.RUNTIME_TFLITE

    def test_chipset_required_for_aot(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_model_release_assets",
                return_value=_model_release_assets(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_runtime_info",
                return_value=RuntimeInfo(is_aot_compiled=True),
            ),
            pytest.raises(KeyError, match="Chipset is required"),
        ):
            get_model_asset_details(
                "mobilenet_v2",
                "qnn_context_binary",
                "float",
                version=_RELEASE_VERSION,
            )

    def test_chipset_lookup(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_model_release_assets",
                return_value=_model_release_assets(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.release_assets.get_runtime_info",
                return_value=RuntimeInfo(is_aot_compiled=True),
            ),
        ):
            asset = get_model_asset_details(
                "mobilenet_v2",
                "qnn_context_binary",
                "float",
                chipset="qualcomm-snapdragon-8-gen-3",
                version=_RELEASE_VERSION,
            )
        assert asset.chipset == "qualcomm-snapdragon-8-gen-3"


# ── platform.py ───────────────────────────────────────────────────────


class TestGetPlatform:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.platform import get_platform

        path = tmp_path / "platform.pb"
        path.write_bytes(_platform_info().SerializeToString())
        result = get_platform(local_path=path)
        assert result.aihm_version == "0.45.0"
        assert len(result.runtimes) == 2

    def test_fetches_from_s3(self) -> None:
        from qai_hub_models_cli.proto_helpers.platform import get_platform

        with (
            _not_dev_release,
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=False,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.platform.get_manifest",
                return_value=_manifest(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.fetch_proto",
                return_value=_platform_info(),
            ),
        ):
            result = get_platform(_RELEASE_VERSION)
        assert result.aihm_version == "0.45.0"


# ══════════════════════════════════════════════════════════════════════════
# Disk cache tests (dev-release path)
# Tests manifest (release-level) and info (per-model) as representatives.
# ══════════════════════════════════════════════════════════════════════════


def _enter_mocks(stack: contextlib.ExitStack, mocks: tuple) -> None:
    for m in mocks:
        stack.enter_context(m)


def _aihm_sys_modules(mock_cli: MagicMock) -> dict:
    mock_version_mod = MagicMock()
    mock_version_mod.__version__ = "0.45.0.dev1"
    mock_pkg = MagicMock()
    mock_pkg._version = mock_version_mod
    mock_pkg.cli = mock_cli
    return {
        "qai_hub_models": mock_pkg,
        "qai_hub_models._version": mock_version_mod,
        "qai_hub_models.cli": mock_cli,
    }


class TestManifestDiskCache:
    def _mocks(self, cache_dir: Path, mock_cli: MagicMock) -> tuple:
        return (
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=True,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.get_release_cache_dir",
                return_value=cache_dir,
            ),
            patch.dict("sys.modules", _aihm_sys_modules(mock_cli)),
        )

    def test_cache_miss_writes_file(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_dir = tmp_path / "releases" / "v0.45.0.dev1"
        mock_cli = MagicMock()
        mock_cli.get_manifest_proto = MagicMock(return_value=_manifest())

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_dir, mock_cli))
            result = get_manifest(_DEV_VERSION)

        mock_cli.get_manifest_proto.assert_called_once()
        assert (cache_dir / "manifest.pb").exists()
        assert result.version == "0.45.0"

    def test_cache_hit_skips_getter(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_dir = tmp_path / "releases" / "v0.45.0.dev1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.pb").write_bytes(_manifest().SerializeToString())

        mock_cli = MagicMock()
        mock_cli.get_manifest_proto = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_dir, mock_cli))
            result = get_manifest(_DEV_VERSION)

        mock_cli.get_manifest_proto.assert_not_called()
        assert result.version == "0.45.0"

    def test_lru_cache_prevents_repeated_access(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_dir = tmp_path / "releases" / "v0.45.0.dev1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.pb").write_bytes(_manifest().SerializeToString())
        mock_cli = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_dir, mock_cli))
            result1 = get_manifest(_DEV_VERSION)
            result2 = get_manifest(_DEV_VERSION)

        assert result1 is result2


class TestInfoDiskCache:
    def _mocks(self, cache_path: Path, mock_cli: MagicMock) -> tuple:
        return (
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=True,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.model_cache_path",
                return_value=cache_path,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            patch.dict("sys.modules", _aihm_sys_modules(mock_cli)),
        )

    def test_cache_miss_writes_file(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        cache_path = tmp_path / "info.pb"
        mock_cli = MagicMock()
        mock_cli.get_info_proto = MagicMock(return_value=_model_info())

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_path, mock_cli))
            result = get_model_info("mobilenet_v2", _DEV_VERSION)

        mock_cli.get_info_proto.assert_called_once_with("mobilenet_v2")
        assert cache_path.exists()
        assert result.id == "mobilenet_v2"

    def test_cache_hit_skips_getter(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        cache_path = tmp_path / "info.pb"
        cache_path.write_bytes(_model_info().SerializeToString())
        mock_cli = MagicMock()
        mock_cli.get_info_proto = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_path, mock_cli))
            result = get_model_info("mobilenet_v2", _DEV_VERSION)

        mock_cli.get_info_proto.assert_not_called()
        assert result.id == "mobilenet_v2"

    def test_lru_cache_prevents_repeated_access(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        cache_path = tmp_path / "info.pb"
        cache_path.write_bytes(_model_info().SerializeToString())
        mock_cli = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_path, mock_cli))
            result1 = get_model_info("mobilenet_v2", _DEV_VERSION)
            result2 = get_model_info("mobilenet_v2", _DEV_VERSION)

        assert result1 is result2
