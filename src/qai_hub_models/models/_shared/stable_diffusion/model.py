# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tokenizers import Tokenizer
from typing_extensions import Self

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
)

# isort: on

import json
from pathlib import Path

import diffusers
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx


from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from huggingface_hub import hf_hub_download
from onnxsim import simplify
from qai_hub.client import Device

from qai_hub_models.models._shared.stable_diffusion import utils
from qai_hub_models.models._shared.stable_diffusion.model_adaptation import (
    get_timestep_embedding,
    monkey_patch_model,
)
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.base_model import (
    BaseModel,
    IndependentComponentFromPretrainedMixin,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    FromPretrainedMixin,
    hf_repo_exists,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.qai_hub_helpers import ensure_hexagon_version


class TextEncoderBase(BaseModel, FromPretrainedMixin):
    seq_len: int

    @classmethod
    def adapt_torch_model(cls, model: torch.nn.Module) -> torch.nn.Module:
        model.config.return_dict = False  # type: ignore[union-attr]

        class TextEncoderWrapper(torch.nn.Module):
            """Return only the first output (cond and uncond embedding)"""

            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
                return self.model(*args, **kwargs)[0]

        return TextEncoderWrapper(model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "tokens": TensorSpec(shape=(batch_size, cls.seq_len), dtype="int32"),
        }

    def get_output_names(self) -> list[str]:
        return ["text_embedding"]


class TextEncoderQuantizableBase(AIMETOnnxQuantizableMixin, TextEncoderBase):
    """Exportable CLIP Text Encoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        TextEncoderBase.__init__(self, None)
        self.host_device = host_device

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, AIMETOnnxQuantizableMixin.forward(self, tokens))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        import aimet_onnx
        from aimet_onnx.common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
        from aimet_onnx.quantsim import load_encodings_to_sim

        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        onnx_model = bundle.load_onnx_model()

        # Model-specific onnx transformations
        num_erf = utils.count_op_type(onnx_model, "Erf")
        if num_erf > 0:
            print(f"Warning: Found {num_erf} Erf ops instead of Gelu")

        # fuse_qkv breaks attention's projection matmul -> add pattern
        if checkpoint != "DEFAULT":
            onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

        # Clip attention masks (containing -3.4e34 to reasonable values for
        # encodings to properly represent. Must run after simplify which
        # constant prop to create the attention mask tensor.
        onnx_model = utils.clip_extreme_values(onnx_model)

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int16,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            providers=cls.get_ort_providers(host_device),
        )
        if bundle.aimet_encodings_path:
            load_encodings_to_sim(
                quant_sim, str(bundle.aimet_encodings_path), strict=False
            )
        return cls(quant_sim, host_device=host_device, onnx_bundle=bundle)

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        from qai_hub_models.datasets.stable_diffusion_calib import (
            StableDiffusionCalibDatasetTextEncoder,
        )

        return StableDiffusionCalibDatasetTextEncoder


class UnetBase(BaseModel, FromPretrainedMixin):
    seq_len: int

    @classmethod
    def adapt_torch_model(
        cls, model: UNet2DConditionModel, on_device_opt: bool = True
    ) -> torch.nn.Module:
        """The torch model is used to generate data in addition to generating
        the onnx model
        """
        model.get_time_embed = get_timestep_embedding  # type: ignore[attr-defined]

        if on_device_opt:
            monkey_patch_model(model)  # type: ignore[arg-type]

        class UNet2DConditionModelWrapper(torch.nn.Module):
            """Call with return_dict=false and unpack the output tuple"""

            def __init__(self, model: UNet2DConditionModel) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                latent: torch.Tensor,
                timestep: torch.Tensor,
                text_emb: torch.Tensor,
            ) -> torch.Tensor:
                return self.model(latent, timestep, text_emb, return_dict=False)[0]  # type: ignore[operator]

        return UNet2DConditionModelWrapper(model)

    def forward(
        self, latent: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.model(latent, time_emb, text_emb)

    def get_channel_last_inputs(self) -> list[str]:
        return ["latent"]

    def get_channel_last_outputs(self) -> list[str]:
        return ["output_latent"]

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
        text_emb_dim: int = 1024,
    ) -> InputSpec:
        return {
            "latent": TensorSpec(shape=(batch_size, 4, 64, 64), dtype="float32"),
            "timestep": TensorSpec(shape=(batch_size, 1), dtype="float32"),
            "text_emb": TensorSpec(
                shape=(batch_size, cls.seq_len, text_emb_dim), dtype="float32"
            ),
        }

    def get_output_names(self) -> list[str]:
        return ["output_latent"]


class UnetQuantizableBase(AIMETOnnxQuantizableMixin, UnetBase):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        # model is None as we don't do anything with the torch model
        UnetBase.__init__(self, None)
        self.host_device = host_device

    def forward(
        self, latent: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            AIMETOnnxQuantizableMixin.forward(self, latent, time_emb, text_emb),
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        import aimet_onnx
        from aimet_onnx.common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
        from aimet_onnx.quantsim import load_encodings_to_sim

        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        onnx_model = bundle.load_onnx_model()

        # Don't run simplify on Unet which hangs
        # TODO (#12356): onnxsim cannot handle >2GB model

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int16,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            providers=cls.get_ort_providers(host_device),
        )
        if bundle.aimet_encodings_path:
            load_encodings_to_sim(
                quant_sim, str(bundle.aimet_encodings_path), strict=False
            )
        return cls(quant_sim, host_device=host_device, onnx_bundle=bundle)

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        from qai_hub_models.datasets.stable_diffusion_calib import (
            StableDiffusionCalibDatasetUnet,
        )

        return StableDiffusionCalibDatasetUnet


class VaeDecoderBase(BaseModel, FromPretrainedMixin):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)

    @classmethod
    def adapt_torch_model(cls, model: AutoencoderKL) -> torch.nn.Module:
        model.config.return_dict = False  # type: ignore[attr-defined]

        class AutoencoderKLDecoder(torch.nn.Module):
            def __init__(self, model: AutoencoderKL) -> None:
                super().__init__()
                self.model = model

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                z = z / self.model.config.scaling_factor  # type: ignore[attr-defined]
                z = self.model.post_quant_conv(z)  # type: ignore[attr-defined]
                image = self.model.decoder(z)  # type: ignore[attr-defined]
                # move output range from -1 ~ 1 to 0~1
                image = (image / 2 + 0.5).clamp(0, 1)
                # output in NHWC
                return image.permute(0, 2, 3, 1)

        return AutoencoderKLDecoder(model)

    def get_channel_last_inputs(self) -> list[str]:
        return ["latent"]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "latent": TensorSpec(shape=(batch_size, 4, 64, 64), dtype="float32"),
        }

    def get_output_names(self) -> list[str]:
        return ["image"]


class VaeDecoderQuantizableBase(AIMETOnnxQuantizableMixin, VaeDecoderBase):
    """Exportable VaeDecoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        VaeDecoderBase.__init__(self, None)
        self.host_device = host_device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        import aimet_onnx
        from aimet_onnx.common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
        from aimet_onnx.quantsim import load_encodings_to_sim

        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        onnx_model = bundle.load_onnx_model()

        if checkpoint != "DEFAULT":
            onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int16,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            providers=cls.get_ort_providers(host_device),
        )
        if bundle.aimet_encodings_path:
            load_encodings_to_sim(
                quant_sim, str(bundle.aimet_encodings_path), strict=False
            )
        return cls(quant_sim, host_device=host_device, onnx_bundle=bundle)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, AIMETOnnxQuantizableMixin.forward(self, latent))

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_hexagon_version(
            min_version=73,
            target_runtime=target_runtime,
            device=device,
            model_name="Stable Diffusion VaeDecoder",
        )

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        from qai_hub_models.datasets.stable_diffusion_calib import (
            StableDiffusionCalibDatasetVae,
        )

        return StableDiffusionCalibDatasetVae


def make_scheduler(
    checkpoint: CheckpointSpec,
    subfolder: str,
    revision: str | None = None,
) -> diffusers.SchedulerMixin:
    """
    Load and instantiate the scheduler from a Hugging Face repo or a local path.

    Parameters
    ----------
    checkpoint
        Hugging Face repo ID or local path.
    subfolder
        Subdirectory where scheduler_config.json is located.
    revision
        Git branch, tag, or commit (only used for HF repos).

    Returns
    -------
    scheduler : diffusers.SchedulerMixin
        A scheduler instance (subclass of SchedulerMixin).
    """
    if hf_repo_exists(str(checkpoint)):
        config_path = hf_hub_download(
            repo_id=str(checkpoint),
            filename=f"{subfolder}/{SCHEDULER_CONFIG_NAME}",
            revision=revision,
        )
    else:
        config_path = str(Path(checkpoint) / subfolder / SCHEDULER_CONFIG_NAME)

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cls_name = cfg.pop("_class_name")
    # Replace PNDMScheduler with EulerDiscreteScheduler for more creative /
    # faster generation
    if cls_name == "PNDMScheduler":
        cls_name = "EulerDiscreteScheduler"
    scheduler_cls = getattr(diffusers, cls_name)
    return scheduler_cls.from_config(cfg)


class StableDiffusionBase(
    IndependentComponentFromPretrainedMixin, PretrainedCollectionModel
):
    """Put glue modules here to aid app/demo code."""

    guidance_scale: float = 7.5
    default_num_steps: int = 20
    hf_repo_id: str = ""

    @staticmethod
    def make_tokenizer() -> Tokenizer:
        raise NotImplementedError()

    @classmethod
    def make_scheduler(
        cls, checkpoint: CheckpointSpec, subfolder: str = "scheduler"
    ) -> diffusers.SchedulerMixin:
        checkpoint = cls.handle_default_checkpoint(checkpoint)
        return make_scheduler(checkpoint, subfolder)

    @classmethod
    def handle_default_checkpoint(cls, checkpoint: CheckpointSpec) -> CheckpointSpec:
        """Convert DEFAULT checkpoint to HF_REPO id"""
        ckpt_type = CheckpointType.from_checkpoint(checkpoint)
        if ckpt_type in [CheckpointType.DEFAULT, CheckpointType.DEFAULT_UNQUANTIZED]:
            if cls.hf_repo_id == "":
                raise ValueError("hf_repo_id is not defined.")
            return cls.hf_repo_id
        return checkpoint
