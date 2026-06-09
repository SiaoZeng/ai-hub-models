# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import gc

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from qai_hub_models.models.sam3.demo import IMAGE_ADDRESS
from qai_hub_models.models.sam3.demo import main as demo_main
from qai_hub_models.models.sam3.model import (
    SAM3,
    SAM3Head,
    SAM3Loader,
    SAM3VisionBackbone,
)
from qai_hub_models.models.sam3.model_patches import (
    SAM3Normalize,
    patch_decoder_rpb_device,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.testing import assert_most_close


def _preprocess_image(pil_image: Image.Image, img_h: int, img_w: int) -> torch.Tensor:
    """Resize and scale a PIL image to the backbone's [0, 1] float input."""
    tensor = (
        torch.from_numpy(np.asarray(pil_image).copy()).permute(2, 0, 1).unsqueeze(0)
    )
    transform = v2.Compose(
        [
            v2.Resize(size=(img_h, img_w)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return transform(tensor)


def test_e2e_numerical() -> None:
    """
    Verify the patched SAM3 produces numerically equivalent outputs to
    the unpatched upstream SAM3 model on a real image + text prompt.
    """
    img_h, img_w = SAM3VisionBackbone.get_input_spec()["image"][0][-2:]
    pil_image = load_image(IMAGE_ADDRESS)
    image = _preprocess_image(pil_image, img_h, img_w)
    text_prompts = ["cup"]

    # Reference: build a fresh upstream SAM3 with only the minimal
    # device-safety patch applied.
    sam3_raw = SAM3Loader._load_sam3("cpu")
    patch_decoder_rpb_device(sam3_raw.transformer.decoder)
    tokenizer = sam3_raw.backbone.language_backbone.tokenizer
    context_length = int(sam3_raw.backbone.language_backbone.context_length)
    tokenized = tokenizer(text_prompts, context_length=context_length).long()

    ref_vision_backbone = SAM3VisionBackbone(
        normalize=SAM3Normalize(),
        vision_model=sam3_raw.backbone.vision_backbone,
    )
    ref_head = SAM3Head(
        language_model=sam3_raw.backbone.language_backbone,
        transformer=sam3_raw.transformer,
        segmentation_head=sam3_raw.segmentation_head,
        dot_prod_scoring=sam3_raw.dot_prod_scoring,
        vision_pos_enc_2=SAM3Loader._compute_vision_pos_enc_2(
            sam3_raw, "cpu", img_h, img_w
        ),
    )

    with torch.no_grad():
        ref_vis_out = ref_vision_backbone(image)
        ref_outputs = ref_head(tokenized, *ref_vis_out)

    ref_vis_np = [t.cpu().numpy() for t in ref_vis_out]
    ref_out_np = [t.cpu().numpy() for t in ref_outputs]

    del ref_vision_backbone, ref_head, sam3_raw, ref_vis_out, ref_outputs
    gc.collect()

    # Patched: the full QNN-compatible model used in production.
    sam3 = SAM3.from_pretrained()
    with torch.no_grad():
        vis_out = sam3.vision_backbone(image)
        outputs = sam3.head(tokenized, *vis_out)

    # Vision backbone parity.
    for _name, patched, ref in zip(
        ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        vis_out,
        ref_vis_np,
        strict=True,
    ):
        assert_most_close(patched.cpu().numpy(), ref, 0.005, rtol=0.001, atol=0.005)

    # Head parity.
    for _name, patched, ref in zip(
        ["pred_boxes", "scores", "pred_masks"],
        outputs,
        ref_out_np,
        strict=True,
    ):
        assert_most_close(patched.cpu().numpy(), ref, 0.005, rtol=0.001, atol=0.005)


def test_demo() -> None:
    demo_main(is_test=True)
