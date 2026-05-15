# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import torch
from torchvision.models import vgg
from typing_extensions import Self

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.deepbox.external_repos.boundingbox_3d.torch_lib import (
    Model as TorchLibModel,
)
from qai_hub_models.models.yolov3.model import YoloV3
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

VGG_WEIGHTS_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "epoch_10.pkl"
)
DEFAULT_YOLO_WEIGHTS = "yolov3-tinyu.pt"


class Yolo2DDetection(YoloV3):
    """
    Exportable YoloV3 bounding box detector, end-to-end.

    Hand detection model. Input is an image, output is
    [bounding boxes & keypoints, box & keypoint scores]
    """

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return super()._sample_inputs_impl(input_spec or self.get_input_spec())


class VGG3DDetection(BaseModel):
    """
    Hand landmark detector model. Input is an image cropped to the hand. The hand must be upright
    and un-tilted in the frame. Returns [landmark_scores, prob_is_right_hand, landmarks]
    """

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "DEFAULT") -> Self:
        if ckpt_path == "DEFAULT":
            ckpt_path = str(VGG_WEIGHTS_ASSET.fetch())

        my_vgg = vgg.vgg19_bn(pretrained=True)
        vgg_model = TorchLibModel.Model(features=my_vgg.features, bins=2)
        checkpoint = load_torch(ckpt_path)
        vgg_model.load_state_dict(checkpoint["model_state_dict"])
        return cls(vgg_model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass on the VGG 3D detection model.

        Parameters
        ----------
        image
            RGB image of range [0, 1] and shape [1, 3, H, W].

        Returns
        -------
        orient : torch.Tensor
            Orientation prediction. Shape is [1, bins, 2], where bins=2 and
            2 represents (cos, sin) of the local orientation angle.
        conf : torch.Tensor
            Confidence prediction for each orientation bin. Shape is [1, bins],
            where bins=2.
        dim : torch.Tensor
            Dimension prediction (height, width, length offsets). Shape is [1, 3].
        """
        # The original implementation of DeepBox applies RGB torchvision constants to BGR input images.
        image_bgr = torch.flip(image, dims=[1])
        norm_image_bgr = normalize_image_torchvision(image_bgr)
        out = self.model(norm_image_bgr)
        return out[0], out[1], out[2]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["orient", "conf", "dim"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


@CollectionModel.add_component(Yolo2DDetection, "yolo_2d_detection")
@CollectionModel.add_component(VGG3DDetection, "vgg_3d_detection")
class DeepBox(PretrainedCollectionModel):
    def __init__(
        self, yolo_2d_det: Yolo2DDetection, vgg_3d_det: VGG3DDetection
    ) -> None:
        super().__init__(yolo_2d_det, vgg_3d_det)
        self.yolo_2d_det = yolo_2d_det
        self.vgg_3d_det = vgg_3d_det

    @classmethod
    def from_pretrained(
        cls,
        yolo_ckpt: str = DEFAULT_YOLO_WEIGHTS,
        vgg_ckpt_path: str = "DEFAULT",
    ) -> Self:
        yolo = Yolo2DDetection.from_pretrained(yolo_ckpt)
        vgg_net = VGG3DDetection.from_pretrained(vgg_ckpt_path)
        return cls(yolo, vgg_net)
