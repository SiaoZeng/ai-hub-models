# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
from skimage.data import astronaut

from qai_hub_models.models.sixd_repnet.app import SixDRepNetApp
from qai_hub_models.models.sixd_repnet.demo import main as demo_main
from qai_hub_models.models.sixd_repnet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SixDRepNet,
)
from qai_hub_models.scorecard.utils.testing import skip_clone_repo_check
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy

EXPECTED_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_output.npy"
)


@skip_clone_repo_check
def test_task() -> None:
    """Test that SixDRepNet predicts head pose angles matching the expected output."""
    image = PILImageModule.fromarray(astronaut())
    model = SixDRepNet.from_pretrained()
    app = SixDRepNetApp(
        face_detector=model.face_detector,
        pose_estimator=model.pose_estimator,
    )

    # raw_output=True returns a [pitch, yaw, roll] numpy array
    output = app.predict_pose(np.array(image), raw_output=True)
    assert not isinstance(output, PILImage)
    angles = output if isinstance(output, np.ndarray) else output[0]

    expected = load_numpy(EXPECTED_OUTPUT)
    np.testing.assert_allclose(angles, expected, rtol=0.01, atol=1.0)


@skip_clone_repo_check
def test_demo() -> None:
    """Test that the SixDRepNet demo runs without error."""
    demo_main(is_test=True)
