# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.yolor.app import YoloRDetectionApp
from qai_hub_models.models.yolor.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolor.demo import main as demo_main
from qai_hub_models.models.yolor.model import MODEL_ASSET_VERSION, MODEL_ID, YoloR
from qai_hub_models.scorecard.utils.testing import skip_clone_repo_check
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)

OUTPUT_HORSES = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolor_horses_outputs_v2.npz"
).fetch()


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    model = YoloR.from_pretrained(include_postprocessing=True)
    app = YoloRDetectionApp(model)
    boxes_list, scores_list, class_idx_list = app.predict_boxes_from_image(
        image, raw_output=True
    )
    boxes = boxes_list[0].numpy()
    scores = scores_list[0].numpy()
    class_idx = class_idx_list[0].numpy()

    with np.load(OUTPUT_HORSES) as data:
        boxes_saved = data["boxes"]
        scores_saved = data["scores"]
        class_idx_saved = data["class_idx"]

    np.testing.assert_allclose(boxes_saved, boxes, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(scores_saved, scores, rtol=1e-2, atol=1e-2)
    np.testing.assert_equal(class_idx_saved, class_idx)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
