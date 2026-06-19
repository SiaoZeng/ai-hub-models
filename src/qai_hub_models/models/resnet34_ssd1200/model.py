# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import torch
from qai_hub.client import Device

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import OutputSpec
from qai_hub_models.models._shared.yolo.model import (
    Yolo,
)
from qai_hub_models.models.resnet34_ssd1200.external_repos.inference.vision.classification_and_detection.python.models.ssd_r34 import (
    SSD_R34,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_model import SerializationSettings
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import (
    BboxFormat,
    BboxMetadata,
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

MODEL_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ssd-resnet34.pth"
)
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000000785.png"
)


class Resnet34SSD(Yolo):
    def __init__(
        self,
        model: torch.nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(
            serialization_settings=SerializationSettings(use_pt2=False),
        )
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output

    @classmethod
    def from_pretrained(
        cls,
        ckpt: Path | CachedWebModelAsset = MODEL_PATH,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Resnet34SSD:
        """
        Load a pretrained Resnet34SSD model from a checkpoint.

        Parameters
        ----------
        ckpt
            Path to the model checkpoint file. Defaults to the fetched asset path.
        include_postprocessing
            It's defined to make it compatible with the YOLO abstraction.
        split_output
            It's defined to make it compatible with the YOLO abstraction.

        Returns
        -------
        Resnet34SSD
            An instance of the model wrapped in the BaseModel interface.
        """
        model = SSD_R34()
        state_dict = load_torch(ckpt)
        model.load_state_dict(state_dict)

        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run Resnet34SSD on `image`, and produce a predicted set of bounding boxes and associated class scores and labels.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            boxes
                Bounding box locations. Shape is [batch, num_boxes, 4] where 4 == (x1, y1, x2, y2)
                Each box has absolute pixel coordinates based on the input image size (img_w x img_h).
            scores
                Each row contains class confidence values in the range [0, 1]. Shape is [batch, num_preds]
            labels
                Shape is [batch, num_preds] where each value is an integer class ID in the range [0, 80].
        """
        boxes, labels, scores = self.model(normalize_image_torchvision(image))

        # Shift COCO labels (1-80) down by 1 because SSD uses 81 classes where:
        #   - class 0 = background
        #   - classes 180 correspond to COCO classes
        # COCO annotations do not include a background class, so we subtract 1 to align them.
        labels = [l - 1 for l in labels]
        # Handle both tensor and tuple cases
        img_h, img_w = image.shape[2], image.shape[3]
        for box in boxes:
            box[:, 0] *= img_w  # x1
            box[:, 2] *= img_w  # x2
            box[:, 1] *= img_h  # y1
            box[:, 3] *= img_h  # y2
        return boxes[0].unsqueeze(0), scores[0].unsqueeze(0), labels[0].unsqueeze(0)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        """
        Specify the expected input format for the model.

        Parameters
        ----------
        batch_size
            Batch size for the input tensor. Default is 1.

        Returns
        -------
        InputSpec
            A dictionary describing input shape and data type.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, 1200, 1200),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(io_type=IoType.TENSOR),
            "labels": TensorSpec(io_type=IoType.TENSOR),
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options
