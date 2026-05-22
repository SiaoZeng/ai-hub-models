# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

from PIL import Image

from qai_hub_models import TargetRuntime
from qai_hub_models.models.yoloe_seg.app import (
    YoloESegmentationApp,
)
from qai_hub_models.models.yoloe_seg.model import (
    COCO_LABELS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloESegmentor,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebAsset,
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    "yoloe_seg", MODEL_ASSET_VERSION, "test_images/bus.jpg"
)


def yolo_prompt_segmentation_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    default_score_threshold: float = 0.45,
    stride_multiple: int | None = None,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=[TargetRuntime.TFLITE], add_output_dir=True
    )
    image_help = "image file path or URL."
    if stride_multiple:
        image_help = f"{image_help} Image spatial dimensions (x and y) must be multiples of {stride_multiple}."

    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help=image_help,
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--model-classes",
        type=str,
        default=COCO_LABELS,
        help=(
            "Comma-separated list of ALL classes the model was exported with "
            "(e.g. 'bus,person'). Required when running on-device (--eval-mode "
            "on-device) and --prompt-text specifies only a subset of those classes. "
            "When omitted for local inference the class list is read directly from "
            "the loaded model."
        ),
    )
    # Override the default for --prompt-text to None so we can detect whether
    # the user explicitly passed it.  The from_pretrained default (COCO_LABELS)
    # is restored below before model loading so local inference is unaffected.
    parser.set_defaults(prompt_text=None)

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Remember whether the user explicitly supplied --prompt-text.
    # args.prompt_text is None when the argument was not passed on the CLI.
    user_prompt: str | None = args.prompt_text

    # Restore a valid default so that model loading / compilation always
    # receives a proper class list (from_pretrained does not accept None).
    if args.prompt_text is None:
        args.prompt_text = COCO_LABELS

    # Load image & model
    model = demo_model_from_cli_args(model_type, model_id, args)

    # Determine which class indices to keep after NMS.
    # - requested_classes : the classes the user wants to see
    # - model_classes     : the full ordered class list the model was exported with
    model_classes: list[str] | Any | None = None

    if hasattr(model, "prompt_text"):
        # Local inference: the model was loaded with exactly the requested classes,
        # so model.prompt_text IS the full class list for this run.
        model_classes = model.prompt_text
        requested_classes = model_classes
    else:
        # On-device inference: parse the full exported class list from
        # --model-classes (defaults to all COCO labels).
        if args.model_classes.endswith(".txt"):
            with open(args.model_classes) as f:
                lines = f.readlines()
            model_classes = [t.rstrip("\r\n").strip() for t in lines if t.strip()]
        else:
            model_classes = [
                t.strip() for t in args.model_classes.split(",") if t.strip()
            ]

        if user_prompt:
            # User explicitly requested a subset of classes via --prompt-text.
            requested_classes = [t.strip() for t in user_prompt.split(",") if t.strip()]
        else:
            # No explicit prompt detect all classes the model was exported with.
            requested_classes = model_classes
    app = YoloESegmentationApp(
        model,  # type: ignore[arg-type]
        args.score_threshold,
        args.iou_threshold,
        filter_prompt=requested_classes if requested_classes else None,
        model_classes=model_classes,
    )

    print("Model Loaded")

    image = load_image(args.image)
    image_annotated = app.predict_segmentation_from_image_yoloe(image)[0]
    assert isinstance(image_annotated, Image.Image)

    if not is_test:
        display_or_save_image(image_annotated, args.output_dir)


def main(is_test: bool = False) -> None:
    yolo_prompt_segmentation_demo(
        YoloESegmentor,
        MODEL_ID,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
