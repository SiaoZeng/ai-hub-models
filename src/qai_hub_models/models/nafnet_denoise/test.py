# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import warnings

import numpy as np
import pytest

from qai_hub_models.models._shared.nafnet.app import NAFNetApp
from qai_hub_models.models.nafnet_denoise.demo import main as demo_main
from qai_hub_models.models.nafnet_denoise.model import (
    IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NafNetDeNoise,
)
from qai_hub_models.scorecard.utils.testing import skip_clone_repo_check
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/denoise_img.png"
)


@skip_clone_repo_check
def test_task() -> None:
    warnings.filterwarnings("ignore")
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    model = NafNetDeNoise.from_pretrained()
    app = NAFNetApp(model, model.get_input_spec())
    restored_image = app.restore_image(image)

    np.testing.assert_allclose(
        np.asarray(preprocess_PIL_image(restored_image), dtype=np.float32),
        np.asarray(preprocess_PIL_image(output_image), dtype=np.float32),
        atol=0.01,
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    model = NafNetDeNoise.from_pretrained()
    app = NAFNetApp(
        model.convert_to_torchscript(),
        model.get_input_spec(),
    )
    restored_image = app.restore_image(image)
    np.testing.assert_allclose(
        np.asarray(preprocess_PIL_image(restored_image), dtype=np.float32),
        np.asarray(preprocess_PIL_image(output_image), dtype=np.float32),
        rtol=0.2,
        atol=0.01,
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
