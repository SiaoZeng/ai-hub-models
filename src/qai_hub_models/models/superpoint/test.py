# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from PIL.Image import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from qai_hub_models.models.superpoint.app import SuperPointApp
from qai_hub_models.models.superpoint.demo import main as demo_main
from qai_hub_models.models.superpoint.model import (
    HF_MODEL_ID,
    INPUT_IMAGE_ADDRESS_1,
    INPUT_IMAGE_ADDRESS_2,
    SuperPoint,
)
from qai_hub_models.scorecard.utils.testing import (
    assert_most_close,
    skip_clone_repo_check,
)
from qai_hub_models.utils.asset_loaders import load_image


def run_source_model(
    image1: Image,
    image2: Image,
    height: int,
    width: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run the HF SuperPoint model and return raw outputs for both images."""
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID, use_fast=True)
    hf_model = SuperPointForKeypointDetection.from_pretrained(HF_MODEL_ID)
    hf_model.eval()
    inputs = processor(
        [image1, image2],
        return_tensors="pt",
        size={"height": height, "width": width},
        do_grayscale=True,
    )
    with torch.no_grad():
        outputs = hf_model(**inputs)
    results = []
    for slot in range(2):
        mask = outputs.mask[slot].bool()
        kp_rel = outputs.keypoints[slot][mask]
        sc = outputs.scores[slot][mask]
        desc = outputs.descriptors[slot][mask]
        kp_abs = kp_rel * torch.tensor([width, height], dtype=kp_rel.dtype)
        results.extend([kp_abs.numpy(), sc.numpy(), desc.numpy()])
    return cast(
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        tuple(results),
    )


def _compare(
    kp: np.ndarray,
    sc: np.ndarray,
    desc: np.ndarray,
    exp_kp: np.ndarray,
    exp_sc: np.ndarray,
    exp_desc: np.ndarray,
    keypoint_threshold: float,
) -> None:
    our_valid = sc > keypoint_threshold
    hf_valid = exp_sc > keypoint_threshold
    kp, sc, desc = kp[our_valid], sc[our_valid], desc[our_valid]
    exp_kp, exp_sc, exp_desc = exp_kp[hf_valid], exp_sc[hf_valid], exp_desc[hf_valid]

    dists = np.linalg.norm(exp_kp[:, None, :] - kp[None, :, :], axis=2)
    matched_idx = dists.argmin(axis=1)
    matched_dist = dists[np.arange(len(exp_kp)), matched_idx]
    close = matched_dist <= 1.0

    exp_kp_m = exp_kp[close]
    exp_sc_m = exp_sc[close]
    exp_desc_m = exp_desc[close]
    kp_m = kp[matched_idx[close]]
    sc_m = sc[matched_idx[close]]
    desc_m = desc[matched_idx[close]]

    assert len(exp_kp_m) > 0, "No keypoints matched between our model and HF reference"

    assert_most_close(
        exp_kp_m.astype(np.float32),
        kp_m.astype(np.float32),
        diff_tol=0.05,
        rtol=0.0,
        atol=1.0,
    )
    assert_most_close(
        exp_sc_m.astype(np.float32),
        sc_m.astype(np.float32),
        diff_tol=0.05,
        rtol=0.0,
        atol=0.005,
    )
    assert_most_close(
        exp_desc_m.astype(np.float32),
        desc_m.astype(np.float32),
        diff_tol=0.05,
        rtol=0.0,
        atol=0.005,
    )


def _run_test(hub_model: SuperPoint, traced: bool = False) -> None:
    image1 = load_image(INPUT_IMAGE_ADDRESS_1)
    image2 = load_image(INPUT_IMAGE_ADDRESS_2)

    if traced:
        app = SuperPointApp(
            hub_model.convert_to_torchscript(),
            height=480,
            width=640,
        )
    else:
        app = SuperPointApp(
            hub_model,
            height=480,
            width=640,
        )

    raw = app.predict(image1, image2, raw_output=True)
    assert not isinstance(raw, Image)
    kp1, sc1, desc1, kp2, sc2, desc2 = raw
    exp_kp1, exp_sc1, exp_desc1, exp_kp2, exp_sc2, exp_desc2 = run_source_model(
        image1, image2, height=480, width=640
    )

    _compare(kp1, sc1, desc1, exp_kp1, exp_sc1, exp_desc1, hub_model.keypoint_threshold)
    _compare(kp2, sc2, desc2, exp_kp2, exp_sc2, exp_desc2, hub_model.keypoint_threshold)


@skip_clone_repo_check
def test_task() -> None:
    _run_test(SuperPoint.from_pretrained())


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    _run_test(SuperPoint.from_pretrained(), traced=True)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
