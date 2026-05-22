# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import pytest
import torch
from ultralytics.models import YOLOE as ultralytics_YOLOE
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.yolo.model import yolo_segment_postprocess
from qai_hub_models.models.yoloe_seg.app import YoloESegmentationApp
from qai_hub_models.models.yoloe_seg.demo import IMAGE_ADDRESS
from qai_hub_models.models.yoloe_seg.demo import main as demo_main
from qai_hub_models.models.yoloe_seg.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloESegmentor,
)
from qai_hub_models.scorecard.utils.testing import (
    assert_most_close,
    skip_clone_repo_check,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image

WEIGHTS = DEFAULT_WEIGHTS
PROMPT_TEXT = "bus,person"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/out_bus_person_with_mask.png"
)


@skip_clone_repo_check
def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    qaihm_model = YoloESegmentor.from_pretrained(PROMPT_TEXT, WEIGHTS)
    qaihm_app = YoloESegmentationApp(qaihm_model, nms_score_threshold=0.25)

    # Load source model and set the same text prompts
    source_yoloe = ultralytics_YOLOE(WEIGHTS)
    source_yoloe.set_classes([t.strip() for t in PROMPT_TEXT.split(",")])
    source_model = cast(SegmentationModel, source_yoloe.model)

    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    processed_sample_image = qaihm_app.preprocess_input(processed_sample_image)

    with torch.no_grad():
        # Original model output: source_model returns ((predictions, protos), ...)
        # source_out[0] is the raw detector tensor of shape [batch, 4+num_classes+32, num_anchors]
        source_out = source_model(processed_sample_image)[0]
        source_out_postprocessed = yolo_segment_postprocess(
            source_out[0], qaihm_model.num_classes
        )

        # Qualcomm AI Hub Model output: returns (boxes, scores, mask_coeffs, class_idx, mask_protos)
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)

        # Compare boxes, scores, mask_coeffs, and class_idx (first 4 outputs)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(
                source_out_postprocessed[i].numpy(),
                qaihm_out_postprocessed[i].numpy(),
                rtol=1e-3,
                atol=1e-4,
            ), f"Output mismatch at index {i}"


@skip_clone_repo_check
@pytest.mark.trace
def test_trace() -> None:
    """Verify that the TorchScript trace of the model runs and produces valid output."""
    net = YoloESegmentor.from_pretrained(PROMPT_TEXT, WEIGHTS)
    input_spec = net.get_input_spec()
    trace = net.convert_to_torchscript(input_spec, check_trace=False)

    # Verify the traced model runs end-to-end without errors
    img = load_image(IMAGE_ADDRESS)
    app = YoloESegmentationApp(trace, nms_score_threshold=0.25)
    out_imgs = app.predict_segmentation_from_image_yoloe(img)

    expected_out = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_close(
        np.asarray(out_imgs[0], dtype=np.float32),
        np.asarray(expected_out, dtype=np.float32),
        0.005,
    )


@skip_clone_repo_check
def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
