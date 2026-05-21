# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.datasets.sav import SaVDataset
from qai_hub_models.models._shared.sam2.model import (
    SAM2Encoder as SAM2EncoderBase,
)
from qai_hub_models.models._shared.sam2.model import (
    SAM2Loader as SAM2LoaderBase,
)
from qai_hub_models.models._shared.sam2.model import (
    SAM2VideoDecoder as SAM2VideoDecoderBase,
)
from qai_hub_models.models.edgetam.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.edgetam.external_repos.edgetam.sam2.build_sam import (
    build_sam2,
)
from qai_hub_models.models.edgetam.external_repos.edgetam.sam2.modeling.sam2_base import (
    SAM2Base as Sam2,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_MODEL_TYPE = "edgetam"

MODEL_REGISTERY = {
    DEFAULT_MODEL_TYPE: "edgetam.pt",
}

CONFIG_REGISTERY = {
    DEFAULT_MODEL_TYPE: "edgetam.yaml",
}


class EdgeTAMEncoder(SAM2EncoderBase):
    """Exportable EdgeTAM encoder (backbone + prompt encoding) for the first/conditioning frame."""

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        sam2 = EdgeTAMLoader._load_sam2(model_type)
        EdgeTAMLoader._patch_sam2_for_qnn_compatibility(sam2)
        return cls(sam2)


class EdgeTAMMemoryEncoder(BaseModel):
    """
    Encodes predicted mask + image features into a memory representation.

    Takes the low-resolution image features and high-resolution predicted mask,
    then produces compact memory features via the memory encoder and spatial
    perceiver for use in subsequent frame tracking.
    """

    def __init__(self, sam2: Sam2) -> None:
        super().__init__()
        self.memory_encoder = sam2.memory_encoder
        self.spatial_perceiver = sam2.spatial_perceiver
        self.sigmoid_scale = sam2.sigmoid_scale_for_mem_enc
        self.sigmoid_bias = sam2.sigmoid_bias_for_mem_enc
        self.no_obj_embed_spatial = sam2.no_obj_embed_spatial
        assert self.no_obj_embed_spatial is None, (
            "EdgeTAMMemoryEncoder only supports models with no_obj_embed_spatial=False; "
            "use the base SAM2 memory encoder for models that set it to True."
        )
        assert self.spatial_perceiver is not None, (
            "EdgeTAMMemoryEncoder requires spatial_perceiver to be set; "
            "use the base SAM2 memory encoder for models without a spatial perceiver."
        )

        # Precompute the constant maskmem positional encoding.
        # PerceiverResampler returns the same pos_enc for any input, so a
        # single dummy forward pass is sufficient.
        img_size = sam2.image_size
        feat_hw = img_size // sam2.backbone_stride
        dummy_feat = torch.zeros(1, sam2.hidden_dim, feat_hw, feat_hw)
        dummy_mask = torch.zeros(1, 1, img_size, img_size)
        maskmem_out = sam2.memory_encoder(
            dummy_feat, dummy_mask, skip_mask_sigmoid=True
        )
        if sam2.spatial_perceiver is not None:
            _, maskmem_pos_enc = sam2.spatial_perceiver(
                maskmem_out["vision_features"], maskmem_out["vision_pos_enc"][0]
            )
        else:
            maskmem_pos_enc = maskmem_out["vision_pos_enc"][0]
        self.maskmem_pos_enc: torch.Tensor = maskmem_pos_enc.detach().cpu()

    def forward(
        self,
        pix_feat: torch.Tensor,
        mask_for_mem: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode image features and predicted mask into memory.

        Parameters
        ----------
        pix_feat
          Raw backbone image features. Shape: [1, 256, 64, 64].
        mask_for_mem
          High-resolution mask already prepared for memory encoding.
          Shape: [1, 1, 1024, 1024], values in [0, 1].

        Returns
        -------
        torch.Tensor
          Memory features with shape [1, 512, 64] after spatial perceiver.
        """
        if self.sigmoid_scale != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale
        if self.sigmoid_bias != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias

        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        maskmem_features = maskmem_out["vision_features"]

        maskmem_features, _ = self.spatial_perceiver(
            maskmem_features, maskmem_out["vision_pos_enc"][0]
        )

        return maskmem_features

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "pix_feat": ((1, 256, 64, 64), "float32"),
            "mask_for_mem": ((1, 1, 1024, 1024), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["maskmem_features"]

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

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        sam2 = EdgeTAMLoader._load_sam2(model_type)
        EdgeTAMLoader._patch_sam2_for_qnn_compatibility(sam2)
        return cls(sam2)

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """Serialize maskmem_pos_enc to a binary .npy file in output_dir."""
        out_path = Path(output_dir) / "maskmem_pos_enc.npy"
        np.save(str(out_path), self.maskmem_pos_enc)
        metadata.supplementary_files[out_path.name] = (
            "Constant memory-encoder positional encoding (numpy .npy). "
            "Load with torch.from_numpy(numpy.load(...)) and pass as maskmem_pos_enc to EdgeTAMVideoApp."
        )


class EdgeTAMVideoDecoder(SAM2VideoDecoderBase):
    """EdgeTAM decoder for video tracking that also returns obj_ptr."""

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

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        sam2 = EdgeTAMLoader._load_sam2(model_type)
        EdgeTAMLoader._patch_sam2_for_qnn_compatibility(sam2)
        return cls(sam2)


class EdgeTAMLoader(SAM2LoaderBase):
    """Helper class for loading and preparing a QNN-compatible EdgeTAM model."""

    @classmethod
    def load(
        cls,
        model_type: str = DEFAULT_MODEL_TYPE,
    ) -> tuple[
        Sam2,
        EdgeTAMEncoder,
        EdgeTAMMemoryEncoder,
        EdgeTAMVideoDecoder,
    ]:
        sam2_patched = cls._load_sam2(model_type)
        cls._patch_sam2_for_qnn_compatibility(sam2_patched)
        encoder = EdgeTAMEncoder(sam2_patched)
        memory_encoder = EdgeTAMMemoryEncoder(sam2_patched)
        video_decoder = EdgeTAMVideoDecoder(sam2_patched)
        return sam2_patched, encoder, memory_encoder, video_decoder

    @classmethod
    def _load_sam2(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Sam2:
        """Get the EdgeTAM model described by the given model type."""
        config_dir = EXTERNAL_REPO_PATHS["edgetam"] / "sam2" / "configs"

        # Initialize Hydra config from the cloned repo
        cls._initialize_hydra_config(
            config_dir=config_dir,
            job_name="edgetam_inference",
        )

        if model_type not in MODEL_REGISTERY:
            raise RuntimeError(f"Weights not found for model type `{model_type}`.")

        asset = CachedWebModelAsset(
            "https://github.com/facebookresearch/EdgeTAM/raw/main/checkpoints/edgetam.pt",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{MODEL_REGISTERY[model_type]}",
        )
        asset.fetch()
        return build_sam2(
            CONFIG_REGISTERY[model_type], asset.local_cache_path, device="cpu"
        )


@CollectionModel.add_component(EdgeTAMEncoder, "encoder")
@CollectionModel.add_component(EdgeTAMMemoryEncoder, "memory_encoder")
@CollectionModel.add_component(EdgeTAMVideoDecoder, "video_decoder")
class EdgeTAM(PretrainedCollectionModel):
    """EdgeTAM video object tracking model."""

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return SaVDataset

    def __init__(
        self,
        sam2: Sam2,
        encoder: EdgeTAMEncoder,
        memory_encoder: EdgeTAMMemoryEncoder,
        video_decoder: EdgeTAMVideoDecoder,
    ) -> None:
        super().__init__(encoder, memory_encoder, video_decoder)
        self.sam2 = sam2
        self.encoder = encoder
        self.memory_encoder = memory_encoder
        self.video_decoder = video_decoder

    @staticmethod
    def calibration_dataset_name() -> str:
        return "sav"

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> EdgeTAM:
        return cls(*EdgeTAMLoader.load(model_type))

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        self.memory_encoder.write_supplementary_files(output_dir, metadata)
