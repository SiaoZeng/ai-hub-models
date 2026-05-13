# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen2.5-VL 7B Vision-Language Model - PreSplit-Part architecture.

Architecture:
- Qwen2_5_VL_7B_PreSplit (Singleton, FP): Manages full model + ONNX splitting
- Qwen2_5_VL_7B_QuantizablePreSplit (Singleton): Manages QuantSim + calibration
- Qwen2_5_VL_7B_PartBase -> Part1..Part4: Unified split inference
  (handles both FP and Quantizable modes based on precision)
- Qwen2_5_VL_7B_VisionEncoder: Vision encoder for on-device export (FP + quantized)
- Collection class for deploying as 5 text splits + 1 vision encoder
"""

from __future__ import annotations

import contextlib
import logging

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on
import gc
import json
import os
import shutil
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicQuantizablePreSplitMixin,
    SingleSlotCacheMixin,
    get_onnx_model,
)
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import split_onnx
from qai_hub_models.models._shared.qwen2_vl.model import (
    Qwen2VLTextBase,
    Qwen2VLTextBase_AIMETOnnx,
)
from qai_hub_models.models._shared.qwen2_vl.vision_encoder import (
    Qwen2VLVisionEncoder,
)
from qai_hub_models.models.common import (
    Precision,
    SampleInputsType,
    SourceModelFormat,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.llm_helpers import export_embedding_weights_from_tensor
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference

logger = logging.getLogger(__name__)

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)

# Model architecture constants (from Qwen2.5-VL-7B-Instruct)
NUM_LAYERS = 28
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 6
HIDDEN_SIZE = 3584
NUM_KEY_VALUE_HEADS = 4
NUM_ATTN_HEADS = 28

# Vision encoder configuration
VISION_HIDDEN_SIZE = 1280
VISION_OUT_HIDDEN_SIZE = 3584
VISION_DEPTH = 32
VISION_NUM_HEADS = 16
VISION_PATCH_SIZE = 14

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 80

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT: dict = {
    Precision.w4a16: "qwen2_5_vl_7b_instruct_w4a16_seqmse",
}

# Default image dimensions (must be divisible by patch_size * spatial_merge_size)
DEFAULT_IMAGE_HEIGHT = 336
DEFAULT_IMAGE_WIDTH = 504

SPLIT_MODEL_NAME = "Qwen2_5_VL_7B"


# ---------------------------------------------------------------------------
# Qwen2_5_VL_7B_PreSplit - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen2_5_VL_7B_PreSplit(SingleSlotCacheMixin, Qwen2VLTextBase):
    """
    FP PreSplit for Qwen2.5-VL-7B.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint. VLM uses split_embedding=False since inputs_embeds
    bypasses the embedding layer.
    """

    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision = DEFAULT_PRECISION,
    ) -> tuple[float | None, float]:
        # Some layers have per-layer scaling
        # defined in _shared/qwen2_vl/model.py.
        return (-250.0, 1.0)

    _hf_repo_name: str = HF_REPO_NAME

    def __init__(
        self,
        checkpoint: str | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)  # type: ignore[misc]
        self.onnx_splits: dict[int, ONNXBundle] = {}
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def __del__(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            with contextlib.suppress(Exception):
                self._temp_dir.cleanup()

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        text_config = self.llm_config
        if hasattr(self.llm_config, "text_config"):
            text_config = self.llm_config.text_config
        if not (
            text_config.num_hidden_layers == NUM_LAYERS
            and text_config.hidden_size == HIDDEN_SIZE
            and text_config.num_attention_heads == NUM_ATTN_HEADS
            and text_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Qwen2_5_VL_7B_PreSplit:
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            cached.sequence_length = sequence_length
            cached.context_length = context_length
            return cached

        attention_mask_min_clip, _ = cls.attention_mask_min_clip_and_multiplier()

        try:
            instance = cls(
                checkpoint=checkpoint,
                sequence_length=sequence_length,
                context_length=context_length,
                host_device=host_device,
                load_pretrained=True,
                attention_mask_min_clip=attention_mask_min_clip,
                _skip_optimizations=_skip_optimizations,
            )
        except Exception:
            cls.clear_cache()
            raise
        cls.cache_store(instance, cache_key)
        return instance

    @staticmethod
    def get_output_names() -> list[str]:
        return Qwen2VLTextBase._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_attention_heads": NUM_ATTN_HEADS,
            }
        return Qwen2VLTextBase._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", NUM_LAYERS),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config.get("hidden_size", HIDDEN_SIZE),
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", NUM_KEY_VALUE_HEADS
            ),
            num_attention_heads=llm_config.get("num_attention_heads", NUM_ATTN_HEADS),
            llm_io_type=llm_io_type,
        )

    def convert_to_onnx_and_split(
        self,
        part_id: int = 1,
    ) -> ONNXBundle:
        """Convert to ONNX and split into parts. Results are cached."""
        if part_id in self.onnx_splits:
            return self.onnx_splits[part_id]

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()

        temp_path = Path(self._temp_dir.name)

        # Export full ONNX model with dynamic shapes
        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            use_dynamic_shapes=True,
        )

        full_bundle = ONNXBundle.from_bundle_path(onnx_dir, "model")

        # Split (no embedding split for VLM - uses inputs_embeds)
        split_output_dir = temp_path / "splits_dynamic"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        split_bundles = split_onnx(
            onnxfile=full_bundle,
            modelname="Qwen2_5_VL_7B",
            num_splits=NUM_SPLITS,
            num_layers_per_split=NUM_LAYERS_PER_SPLIT,
            output_dir=str(split_output_dir),
            split_embedding=False,
        )

        for i, bundle in enumerate(split_bundles):
            self.onnx_splits[i + 1] = bundle

        return self.onnx_splits[part_id]

    def free_memory(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None  # type: ignore[assignment]
        self.onnx_splits.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Qwen2_5_VL_7B_QuantizablePreSplit - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen2_5_VL_7B_QuantizablePreSplit(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin["Qwen2_5_VL_7B_PreSplit"],
    Qwen2VLTextBase_AIMETOnnx,
):
    """
    Quantizable PreSplit for Qwen2.5-VL-7B.

    The S3 asset zip contains the FULL output of quantize.py (dynamic
    ONNX + weights + encodings + tokenizer + config + embedding_weights.raw),
    so DEFAULT resolution just downloads and extracts. No FP torch model
    is needed to load the quantized checkpoint.
    """

    FPModel = Qwen2_5_VL_7B_PreSplit  # type: ignore[assignment]
    _hf_repo_name: str = HF_REPO_NAME

    # DynamicQuantizablePreSplitMixin config
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    # DynamicPreSplitOnnxMixin config
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_embedding = False  # VLM uses inputs_embeds directly

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        # Some layers have per-layer scaling
        # defined in _shared/qwen2_vl/model.py.
        return (-250.0, 1.0)

    @staticmethod
    def get_output_names() -> list[str]:
        return Qwen2VLTextBase._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        return Qwen2_5_VL_7B_PreSplit.get_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: Qwen2_5_VL_7B_PreSplit | None = None,
    ) -> None:
        """Save calibrated checkpoint with ONNX, encodings, and embedding weights."""
        if fp_model is None:
            fp_model = Qwen2_5_VL_7B_PreSplit.from_pretrained(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        super().save_calibrated_checkpoint(output_checkpoint, fp_model)

        # VLM-specific: embedding table is needed for on-device LUT encoder
        # and for token-to-embedding conversion during evaluation.
        export_embedding_weights_from_tensor(
            fp_model.get_embedding_weights(), Path(output_checkpoint)
        )


# ---------------------------------------------------------------------------
# Vision Encoder Component
# ---------------------------------------------------------------------------


class Qwen2_5_VL_7B_VisionEncoder(Qwen2VLVisionEncoder):
    """
    Vision encoder for Qwen2.5-VL-7B (adapted VEG for on-device deployment).

    Supports both FP inference (via PyTorch VEG) and quantized inference
    (via AIMET-ONNX QuantSim). Used as a Collection component.

    During export, loads the pre-quantized ONNX from the checkpoint
    (vision_encoder.{onnx,data,encodings}) instead of re-exporting.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint: str | None = None
        self._precision: Precision = Precision.float
        self._quantized_session: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> Qwen2_5_VL_7B_VisionEncoder:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For quantized inference, resolve DEFAULT to the downloaded asset
        # checkpoint (which contains vision_encoder.onnx/encodings alongside
        # the text model artifacts). FP inference does not need the asset.
        if precision != Precision.float and (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            checkpoint = Qwen2_5_VL_7B_QuantizablePreSplit.fetch_default_checkpoint(
                precision
            )

        # Load FP VEG (provides model weights for FP, buffers for quantized)
        load_device = device if precision == Precision.float else torch.device("cpu")
        instance: Qwen2_5_VL_7B_VisionEncoder = super().from_pretrained(  # type: ignore[assignment]
            checkpoint=HF_REPO_NAME,
            device=load_device,
            image_height=image_height,
            image_width=image_width,
        )
        instance._checkpoint = str(checkpoint)
        instance._precision = precision

        if precision != Precision.float:
            instance._init_quantized_session(Path(str(checkpoint)), device)

        return instance

    def _init_quantized_session(
        self,
        ckpt_path: Path,
        device: torch.device,
    ) -> None:
        """Create an AIMET-ONNX QuantSim session for quantized inference.

        Loads the pre-quantized ONNX from *ckpt_path* and creates a
        QuantSim session. The FP VEG buffers (RoPE, attention masks)
        are already on ``self`` from ``from_pretrained``.
        """
        import logging

        from aimet_common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim

        veg_onnx = ckpt_path / "vision_encoder.onnx"
        veg_enc = ckpt_path / "vision_encoder.encodings"

        onnx_model = onnx.load(str(veg_onnx), load_external_data=True)

        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        quant_logger = logging.getLogger("Quant")
        prev_level = quant_logger.level
        quant_logger.setLevel(logging.WARNING)
        try:
            quant_sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme=QuantScheme.min_max,
                param_type="int8",
                activation_type="int16",
                providers=providers,
            )
            if veg_enc.exists():
                load_encodings_to_sim(quant_sim, str(veg_enc), strict=False)
        finally:
            quant_logger.setLevel(prev_level)

        self._quantized_session = quant_sim

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids_cos: torch.Tensor | None = None,
        position_ids_sin: torch.Tensor | None = None,
        window_attention_mask: torch.Tensor | None = None,
        full_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._is_quantized:
            return self._forward_quantized(pixel_values)
        return super().forward(
            pixel_values=pixel_values,
            position_ids_cos=position_ids_cos,
            position_ids_sin=position_ids_sin,
            window_attention_mask=window_attention_mask,
            full_attention_mask=full_attention_mask,
        )

    def _forward_quantized(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run inference through the AIMET-ONNX QuantSim session."""
        assert self._quantized_session is not None
        # Pass all 5 VEG inputs positionally — mock_torch_onnx_inference
        # maps them by position to ONNX input names.
        return mock_torch_onnx_inference(  # type: ignore[return-value]
            self._quantized_session.session,
            pixel_values,
            self._pos_emb_cos,  # type: ignore[arg-type]
            self._pos_emb_sin,  # type: ignore[arg-type]
            self._window_attention_mask,  # type: ignore[arg-type]
            self._full_attention_mask,  # type: ignore[arg-type]
        )

    @staticmethod
    def get_input_spec(
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
    ) -> InputSpec:
        return Qwen2VLVisionEncoder.get_input_spec(
            image_height=image_height,
            image_width=image_width,
            patch_size=VISION_PATCH_SIZE,
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["image_features"]

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX

    # ------------------------------------------------------------------
    # VEG Quantization Lifecycle (classmethods)
    # ------------------------------------------------------------------

    @classmethod
    def get_calibration_data(
        cls,
        num_samples: int,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
    ) -> list[np.ndarray]:
        """Load real images from imagenette for VEG calibration.

        Returns a list of pixel_values numpy arrays, each shaped
        (seq_len, patch_dim) as expected by the VEG ONNX model.
        """
        from PIL import Image
        from transformers import AutoProcessor

        from qai_hub_models.datasets.imagenette import IMAGENETTE_ASSET

        IMAGENETTE_ASSET.fetch(extract=True)
        img_root = IMAGENETTE_ASSET.extracted_path

        train_dir = img_root / "train"
        image_paths: list[Path] = []
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                image_paths.extend(
                    img_path
                    for img_path in sorted(class_dir.iterdir())
                    if img_path.suffix.lower() in (".jpeg", ".jpg", ".png")
                )
        if len(image_paths) < num_samples:
            raise RuntimeError(
                f"Imagenette has {len(image_paths)} images but "
                f"{num_samples} calibration samples requested."
            )

        step = max(1, len(image_paths) // num_samples)
        selected = image_paths[: step * num_samples : step]

        from qai_hub_models.models._shared.qwen2_vl.model import Qwen2VLTextBase

        proc = AutoProcessor.from_pretrained(HF_REPO_NAME)
        dummy_text = Qwen2VLTextBase.get_input_prompt_with_tags(
            user_input_prompt="", include_image=True
        )

        pixel_values_list: list[np.ndarray] = []
        for i, img_path in enumerate(selected):
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((image_width, image_height))
            processed = proc(
                text=[dummy_text], images=[img_resized], return_tensors="pt"
            )
            pv = processed["pixel_values"].detach().numpy().astype(np.float32)
            pixel_values_list.append(pv)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    Loaded calibration image {i + 1}/{num_samples}")

        return pixel_values_list

    @classmethod
    def create_quantsim(
        cls,
        veg_model: Qwen2_5_VL_7B_VisionEncoder,
        host_device: torch.device,
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Export VEG to ONNX and create an AIMET-ONNX QuantSim.

        Returns ``(quant_sim, fixed_inputs_np)`` where *fixed_inputs_np*
        contains the resolution-dependent inputs (RoPE, masks) needed
        during calibration.
        """
        import aimet_onnx.common.quantsim as qs
        import aimet_onnx.quantsim as quantsim_mod
        import onnx as onnx_lib
        import torch as _torch
        from aimet_common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel

        from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
        from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

        sample_inputs = veg_model.get_sample_inputs()
        input_names = list(sample_inputs.keys())
        dummy_args = tuple(v.to(host_device) for v in sample_inputs.values())

        tmp_dir = tempfile.TemporaryDirectory()
        onnx_path = str(Path(tmp_dir.name) / "vision_encoder.onnx")

        seq_len_dim = _torch.export.Dim("seq_len", min=1)
        dynamic_shapes = (
            {0: seq_len_dim},  # pixel_values
            {0: seq_len_dim},  # position_ids_cos
            {0: seq_len_dim},  # position_ids_sin
            {1: seq_len_dim, 2: seq_len_dim},  # window_attention_mask
            {1: seq_len_dim, 2: seq_len_dim},  # full_attention_mask
        )

        safe_torch_onnx_export(
            veg_model,
            dummy_args,
            onnx_path,
            input_names=input_names,
            output_names=["image_features"],
            opset_version=18,
            dynamo=True,
            optimize=True,
            dynamic_shapes=dynamic_shapes,
        )

        onnx_model = onnx_lib.load(onnx_path)
        tmp_dir.cleanup()

        default_config = get_aimet_config_path("default_config_llama")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if host_device.type != "cuda":
            providers = ["CPUExecutionProvider"]

        quantsim_mod.op_types_to_tie_qtzrs = ["Concat"]
        quantsim_mod._tie_qtzrs = True
        quantsim_mod.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type="int8",
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file=default_config,
            providers=providers,
        )

        Qwen2VLVisionEncoder._configure_quant_sim(quant_sim)

        fixed_inputs_np = {
            name: tensor.cpu().detach().numpy().astype(np.float32)
            for name, tensor in sample_inputs.items()
            if name != "pixel_values"
        }

        return quant_sim, fixed_inputs_np

    @classmethod
    def calibrate(
        cls,
        quant_sim: Any,
        calibration_data: list[np.ndarray],
        fixed_inputs: dict[str, np.ndarray],
    ) -> None:
        """Calibrate the QuantSim with real images."""
        num_samples = len(calibration_data)

        def _forward_pass(session: Any, _unused: Any) -> None:
            for i, pv in enumerate(calibration_data):
                feed = {"pixel_values": pv, **fixed_inputs}
                session.run(None, feed)
                if (i + 1) % 10 == 0:
                    print(f"    Calibration forward pass {i + 1}/{num_samples}")

        quant_sim.compute_encodings(_forward_pass, None)

    @classmethod
    def save_quantized_checkpoint(
        cls,
        quant_sim: Any,
        output_dir: str | Path,
    ) -> None:
        """Save calibrated VEG artifacts (ONNX + encodings) to *output_dir*."""
        from aimet_onnx.utils import save_model_with_external_weights

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = tempfile.TemporaryDirectory()
        tmp = Path(tmp_dir.name)

        quant_sim.export(str(tmp), "vision_encoder", export_model=False)

        shutil.copy2(
            str(tmp / "vision_encoder.encodings"),
            str(out_dir / "vision_encoder.encodings"),
        )

        with quant_sim._remove_quantization_nodes():
            save_model_with_external_weights(
                quant_sim.model.model,
                str(out_dir / "vision_encoder.onnx"),
                location="vision_encoder.data",
                all_tensors_to_one_file=True,
            )

        tmp_dir.cleanup()

    def _get_onnx_bundle(self) -> ONNXBundle:
        if self._checkpoint is None:
            raise ValueError("No checkpoint provided for VisionEncoder.")
        ckpt = Path(self._checkpoint)
        return ONNXBundle(
            bundle_path=ckpt,
            onnx_graph_name="vision_encoder.onnx",
            onnx_weights_name="vision_encoder.data"
            if (ckpt / "vision_encoder.data").exists()
            else None,
            aimet_encodings_name="vision_encoder.encodings"
            if (ckpt / "vision_encoder.encodings").exists()
            else None,
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        model_name = "Qwen2_5_VL_7B_VisionEncoder"

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_path) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return str(out_dir)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        spec = input_spec or self.get_input_spec()
        result: SampleInputsType = {}
        for name, (shape, dtype_str) in spec.items():
            np_dtype = np.float32 if dtype_str == "float32" else np.int64
            result[name] = [np.zeros(shape, dtype=np_dtype)]
        return result


# ---------------------------------------------------------------------------
# Unified Part Base & Concrete Parts
# ---------------------------------------------------------------------------


class Qwen2_5_VL_7B_PartBase(BaseModel):
    """
    Unified Part base: handles both FP and Quantizable modes based on precision.

    Each Part represents one split of the ONNX model for deployment.
    VLM Parts use input_embeds instead of input_ids.
    """

    part_id: int = 0

    def __init__(
        self,
        presplit: Qwen2_5_VL_7B_PreSplit | Qwen2_5_VL_7B_QuantizablePreSplit,
        precision: Precision = DEFAULT_PRECISION,
        context_lengths: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision
        self._context_lengths = context_lengths or [presplit.context_length]
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = False,
        context_lengths: list[int] | None = None,
        **kwargs: Any,
    ) -> Self:
        if precision == Precision.float:
            presplit: Qwen2_5_VL_7B_PreSplit | Qwen2_5_VL_7B_QuantizablePreSplit = (
                Qwen2_5_VL_7B_PreSplit.from_pretrained(
                    sequence_length=sequence_length,
                    context_length=context_length,
                    host_device=host_device,
                )
            )
        else:
            presplit = Qwen2_5_VL_7B_QuantizablePreSplit.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(presplit, precision=precision, context_lengths=context_lengths)

    def get_input_spec(self, **kwargs: Any) -> InputSpec:
        """Return input spec for this split part, read from ONNX model inputs."""
        return self._get_input_spec_for_instance()

    def _get_input_spec_for_instance(
        self,
        seq_len: int | None = None,
        context_length: int | None = None,
    ) -> InputSpec:
        """Get input spec for this specific Part instance.

        VLM Parts don't have a separate embedding split. Part 1 takes
        input_embeds + attention_mask + KV cache for its layers.
        Names are read from the actual split ONNX model.
        """
        if seq_len is None:
            seq_len = self._presplit.sequence_length
        if context_length is None:
            context_length = self._presplit.context_length
        head_dim = HIDDEN_SIZE // NUM_ATTN_HEADS
        kv_seq_len = context_length - seq_len

        onnx_input_names = self._get_onnx_input_names()
        spec: InputSpec = {}

        for name in onnx_input_names:
            if "past_key" in name:
                spec[name] = (
                    (NUM_KEY_VALUE_HEADS, 1, head_dim, kv_seq_len),
                    "float32",
                )
            elif "past_value" in name:
                spec[name] = (
                    (NUM_KEY_VALUE_HEADS, 1, kv_seq_len, head_dim),
                    "float32",
                )
            elif name == "attention_mask":
                spec[name] = (
                    (1, 1, seq_len, context_length),
                    "float32",
                )
            elif name == "inputs_embeds":
                spec[name] = (
                    (1, seq_len, HIDDEN_SIZE),
                    "float32",
                )
            elif name in ("position_ids_cos", "position_ids_sin"):
                embed_dim = head_dim // 2
                spec[name] = (
                    (1, 1, seq_len, embed_dim),
                    "float32",
                )
            else:
                # Intermediate hidden state from previous part
                spec[name] = (
                    (1, seq_len, HIDDEN_SIZE),
                    "float32",
                )

        return spec

    def _genie_graph_name(self, seq_len: int, context_length: int | None = None) -> str:
        """Build a QNN context graph name that genie-app can parse.

        Format: ``{token|prompt}_ar{seq_len}_cl{ctx_len}_{part}_of_{total}``
        """
        ctx_len = context_length or self._presplit.context_length
        prefix = "token" if seq_len == 1 else "prompt"
        return f"{prefix}_ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{NUM_SPLITS}"

    def get_compile_specs(self) -> list[tuple[InputSpec, str | None]]:
        """Return compile specs for token-generation (seq_len=1) and
        prompt-processing (seq_len=sequence_length) at each context length.

        Each spec gets a unique graph name so they are linked into a single
        multi-graph context binary.
        """
        prompt_seq_len = self._presplit.sequence_length
        specs: list[tuple[InputSpec, str | None]] = []

        for ctx_len in self._context_lengths:
            # Token generation: seq_len=1
            spec_tok = self._get_input_spec_for_instance(
                seq_len=1, context_length=ctx_len
            )
            specs.append((spec_tok, self._genie_graph_name(1, ctx_len)))

            # Prompt processing: seq_len=sequence_length (e.g. 128)
            spec_prompt = self._get_input_spec_for_instance(
                seq_len=prompt_seq_len, context_length=ctx_len
            )
            specs.append((spec_prompt, self._genie_graph_name(prompt_seq_len, ctx_len)))

        return specs

    def get_output_names(self) -> list[str]:
        return [
            name.replace("/", "_").replace(".", "_")
            for name in self._get_onnx_output_names()
        ]

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        full_inputs = self._presplit._sample_inputs_impl()
        onnx_input_names = self._get_onnx_input_names()

        result: SampleInputsType = {}
        seq_len = self._presplit.sequence_length

        for name in onnx_input_names:
            if name in full_inputs:
                result[name] = full_inputs[name]
            else:
                # Intermediate hidden state
                result[name] = [np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)]

        return result

    # -------------------------------------------------------------------
    # Methods that branch on self._is_quantized
    # -------------------------------------------------------------------

    def _get_onnx_input_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def _get_quant_sim(self) -> QuantizationSimModel:
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )
        onnx_model.ir_version = min(onnx_model.ir_version, 11)

        assert isinstance(self._presplit, Qwen2_5_VL_7B_QuantizablePreSplit)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")

        # Suppress verbose AIMET logs and prints during per-part QuantSim
        quant_logger = logging.getLogger("Quant")
        prev_level = quant_logger.level
        quant_logger.setLevel(logging.WARNING)
        import io
        import sys as _sys

        _real_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            # Use the same QuantSim creation as the full model (W4A16 config,
            # proper quant scheme, config file) instead of bare defaults (8-bit).
            self._quant_sim = Qwen2_5_VL_7B_QuantizablePreSplit.create_quantsim(
                onnx_model, host_device, self._precision
            )

            if onnx_bundle.aimet_encodings_path is not None:
                load_encodings_to_sim(
                    self._quant_sim,
                    str(onnx_bundle.aimet_encodings_path),
                    strict=False,
                )
        finally:
            _sys.stdout = _real_stdout
            quant_logger.setLevel(prev_level)

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()
        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Patch the file in-place (graph-only
        # load keeps external weight references intact).
        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        # Use a neutral name — the ONNX is exported with dynamic shapes,
        # so one upload serves both token and prompt compile specs.
        model_name = f"{self.part_id}_of_{NUM_SPLITS}"

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_path) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )

        return str(out_dir)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = DEFAULT_PRECISION,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return BaseModel.get_hub_compile_options(
            self,
            target_runtime=target_runtime,
            precision=precision,
            other_compile_options=other_compile_options,
            device=device,
            context_graph_name=context_graph_name,
        )

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        if self._is_quantized:
            return self._presplit.get_hub_profile_options(
                target_runtime=target_runtime,
                other_profile_options=other_profile_options,
                context_graph_name=context_graph_name,
            )
        return super().get_hub_profile_options(
            target_runtime=target_runtime,
            other_profile_options=other_profile_options,
            context_graph_name=context_graph_name,
        )


class Qwen2_5_VL_7B_Part1_Of_5(Qwen2_5_VL_7B_PartBase):
    """Part 1: Layers 0-5."""

    part_id = 1


class Qwen2_5_VL_7B_Part2_Of_5(Qwen2_5_VL_7B_PartBase):
    """Part 2: Layers 6-11."""

    part_id = 2


class Qwen2_5_VL_7B_Part3_Of_5(Qwen2_5_VL_7B_PartBase):
    """Part 3: Layers 12-17."""

    part_id = 3


class Qwen2_5_VL_7B_Part4_Of_5(Qwen2_5_VL_7B_PartBase):
    """Part 4: Layers 18-23."""

    part_id = 4


class Qwen2_5_VL_7B_Part5_Of_5(Qwen2_5_VL_7B_PartBase):
    """Part 5: Layers 24-27 + LM head."""

    part_id = 5


# ---------------------------------------------------------------------------
# Collection Class
# ---------------------------------------------------------------------------


@CollectionModel.add_component(Qwen2_5_VL_7B_VisionEncoder, "vision_encoder")
@CollectionModel.add_component(Qwen2_5_VL_7B_Part1_Of_5, "part1_of_5")
@CollectionModel.add_component(Qwen2_5_VL_7B_Part2_Of_5, "part2_of_5")
@CollectionModel.add_component(Qwen2_5_VL_7B_Part3_Of_5, "part3_of_5")
@CollectionModel.add_component(Qwen2_5_VL_7B_Part4_Of_5, "part4_of_5")
@CollectionModel.add_component(Qwen2_5_VL_7B_Part5_Of_5, "part5_of_5")
class Qwen2_5_VL_7B_Collection(PretrainedCollectionModel):
    """
    Unified Collection with 5 text Parts + 1 Vision Encoder for Qwen2.5-VL-7B.

    Supports both FP and Quantizable modes based on precision parameter.
    All Parts share the same PreSplit via class-level cache.
    """

    _checkpoint: str
    _hub_device: Any

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = False,
        context_lengths: list[int] | None = None,
        **kwargs: Any,
    ) -> Self:
        part_kwargs = dict(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            host_device=host_device,
            _skip_quantsim_creation=_skip_quantsim_creation,
            context_lengths=context_lengths,
        )
        # component_classes is a list on the current branch but becomes
        # dict[str, type] after rebasing onto dev/david/llm3.
        comp_classes = (
            cls.component_classes.values()
            if isinstance(cls.component_classes, dict)
            else cls.component_classes
        )
        components = [
            comp_cls.from_pretrained(**part_kwargs) for comp_cls in comp_classes
        ]
        instance = cls(*components)
        # Use the resolved checkpoint path (not the "DEFAULT" sentinel) so
        # downstream supplementary-file copies find tokenizer.json etc.
        resolved_checkpoint: str | Path = checkpoint
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            for comp in components:
                presplit = getattr(comp, "_presplit", None)
                ckpt = getattr(presplit, "checkpoint", None)
                if ckpt is not None:
                    resolved_checkpoint = ckpt
                    break
        instance._checkpoint = str(resolved_checkpoint)
        return instance

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """Write genie-app assets: genie config, embedding table, tokenizer, HTP config, app script."""
        from qai_hub_models.utils.llm_helpers import (
            create_genie_config,
            generate_genie_app_script,
            save_htp_config_for_genie_bundle,
        )

        output_dir = Path(output_dir)
        checkpoint_path = Path(self._checkpoint)

        # --- Embedding weights ---
        embed_src = checkpoint_path / "embedding_weights.raw"
        if embed_src.exists():
            shutil.copy(embed_src, output_dir / "embedding_weights.raw")
            print("Copied embedding table from checkpoint")
        else:
            fp_model = Qwen2_5_VL_7B_PreSplit.from_pretrained()
            export_embedding_weights_from_tensor(
                fp_model.get_embedding_weights(), output_dir
            )
        metadata.supplementary_files["embedding_weights.raw"] = (
            "Embedding table (float32) for token-to-embedding conversion."
        )

        # --- Tokenizer files ---
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
            src = checkpoint_path / name
            if src.exists():
                shutil.copy(src, output_dir / name)
                metadata.supplementary_files[name] = f"Model {name} from checkpoint."

        # --- Sample prompt (text-only; vision prompt is assembled at runtime) ---
        sample_prompt = Qwen2VLTextBase.get_input_prompt_with_tags(include_image=False)
        with open(output_dir / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample text-only prompt for standalone genie-t2t-run."
        )

        # --- HTP backend extension config ---
        hub_device = getattr(self, "_hub_device", None)
        if hub_device is not None:
            device_info: dict[str, str] = {}
            attrs = hub_device.attributes
            for attr in [attrs] if isinstance(attrs, str) else attrs:
                if ":" in attr:
                    key, value = attr.split(":", 1)
                    device_info[key] = value
            if save_htp_config_for_genie_bundle(device_info, output_dir):
                metadata.supplementary_files["htp_backend_ext_config.json"] = (
                    "HTP backend extension config for Genie."
                )

        # --- Genie config (text-dec-htp.json equivalent) ---
        # Get context_lengths, image_processor, and llm_config from a text Part's presplit
        context_lengths: list[int] = []
        image_processor = None
        llm_config = None
        for comp in self.components.values():
            if isinstance(comp, Qwen2_5_VL_7B_PartBase):
                presplit = comp._presplit
                context_lengths = comp._context_lengths
                image_processor = getattr(presplit, "_image_processor", None)
                # FP presplit stores the full VLM config; quantized presplit
                # only has text_config.  Either works for genie config generation.
                llm_config = getattr(
                    presplit, "_original_llm_config", presplit.llm_config
                )
                break

        # Quantized presplit doesn't cache the image_processor — load from HF.
        if image_processor is None:
            from transformers import AutoProcessor

            image_processor = AutoProcessor.from_pretrained(
                HF_REPO_NAME
            ).image_processor

        assert image_processor.patch_size == VISION_PATCH_SIZE, (
            f"HF image_processor.patch_size ({image_processor.patch_size}) "
            f"!= VISION_PATCH_SIZE ({VISION_PATCH_SIZE})"
        )

        # Build model_list from downloaded text part .bin files (exclude vision encoder)
        model_list = sorted(
            fn
            for fn in metadata.model_files
            if fn.startswith("part") and fn.endswith(".bin")
        )

        # Get text_config from the full VLM config
        assert llm_config is not None, "Could not retrieve llm_config from presplit"
        text_config = llm_config
        if hasattr(llm_config, "text_config"):
            text_config = llm_config.text_config

        # Build VLM MRoPE config from the HF config
        rope_scaling = getattr(text_config, "rope_scaling", None)
        vlm_rope_config: dict[str, Any] = {
            "rope-type": "qwen2vl-mrope",
            "time-step": 50,
        }
        vlm_rope_config["spatial-merge-size"] = image_processor.merge_size
        if rope_scaling is not None and "mrope_section" in rope_scaling:
            vlm_rope_config["mrope-section"] = rope_scaling["mrope_section"]

        if not context_lengths:
            context_lengths = [DEFAULT_CONTEXT_LENGTH]

        # text-generator.json: used by genie-app-script.txt (genie-app VLM pipeline)
        genie_config = create_genie_config(
            context_length=max(context_lengths),
            llm_config=text_config,
            embedding_type="rope",
            model_list=model_list,
            embedding_size=text_config.hidden_size,
            top_level_key="text-generator",
            embedding_lut_path="embedding_weights.raw",
            vlm_rope_config=vlm_rope_config,
        )
        with open(output_dir / "text-generator.json", "w") as f:
            json.dump(genie_config, f, indent=4)
        metadata.supplementary_files["text-generator.json"] = (
            "Genie SDK config for text decoder (VLM pipeline)."
        )

        # genie_config.json: same content with "dialog" key for genie-t2t-run
        dialog_config = create_genie_config(
            context_length=max(context_lengths),
            llm_config=text_config,
            embedding_type="rope",
            model_list=model_list,
            embedding_size=text_config.hidden_size,
            top_level_key="dialog",
            embedding_lut_path="embedding_weights.raw",
            vlm_rope_config=vlm_rope_config,
        )
        with open(output_dir / "genie_config.json", "w") as f:
            json.dump(dialog_config, f, indent=4)
        metadata.supplementary_files["genie_config.json"] = (
            "Genie SDK config for genie-t2t-run (text-only LLM testing)."
        )

        # --- Image encoder config (img-enc-htp.json equivalent) ---
        veg_bins = sorted(
            fn
            for fn in metadata.model_files
            if fn.startswith("vision_encoder") and fn.endswith(".bin")
        )
        img_enc_config = {
            "image-encoder": {
                "version": 1,
                "engine": {
                    "version": 1,
                    "mode": "image",
                    "backend": {
                        "version": 1,
                        "type": "QnnHtp",
                        "QnnHtp": {
                            "version": 1,
                            "spill-fill-bufsize": 0,
                            "use-mmap": False,
                            "allow-async-init": False,
                        },
                        "extensions": "htp_backend_ext_config.json",
                    },
                    "model": {
                        "version": 1,
                        "type": "binary",
                        "binary": {
                            "version": 1,
                            "ctx-bins": veg_bins,
                        },
                        "vision-param": {
                            "height": DEFAULT_IMAGE_HEIGHT
                            // image_processor.patch_size,
                            "width": DEFAULT_IMAGE_WIDTH // image_processor.patch_size,
                        },
                    },
                },
            }
        }
        with open(output_dir / "img-enc-htp.json", "w") as f:
            json.dump(img_enc_config, f, indent=4)
        metadata.supplementary_files["img-enc-htp.json"] = (
            "Genie SDK config for vision encoder."
        )

        # --- Text encoder config (LUT embedding lookup) ---
        text_enc_config = {
            "text-encoder": {
                "version": 1,
                "type": "lut",
                "lut": {
                    "version": 1,
                    "lut-path": "embedding_weights.raw",
                    "size": text_config.hidden_size,
                    "datatype": "float32",
                },
                "tokenizer": {"version": 1, "path": "tokenizer.json"},
            }
        }
        with open(output_dir / "text-encoder.json", "w") as f:
            json.dump(text_enc_config, f, indent=4)
        metadata.supplementary_files["text-encoder.json"] = (
            "Genie SDK config for text encoder (LUT embedding)."
        )

        # --- Genie metadata & genie-app-script.txt ---
        # Define pipeline topology once; use it for both metadata.genie
        # and the genie-app-script.txt that genie-app consumes at runtime.
        from qai_hub_models.configs.model_metadata import (
            GenieChatTemplate,
            GenieMetadata,
            GeniePipeline,
            GeniePipelineConnection,
            GenieSampleInput,
            GenieVisionPreprocessing,
        )

        chat_spec = Qwen2_5_VL_7B_PreSplit.get_chat_template()

        pipeline_nodes = {
            "imageEncoder": "img-enc-htp.json",
            "lutEncoder": "text-encoder.json",
            "textGenerator": "text-generator.json",
        }

        pipeline_connections = [
            GeniePipelineConnection(
                producer_node="imageEncoder",
                producer_node_io="GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT",
                consumer_node="textGenerator",
                consumer_node_io="GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
            ),
            GeniePipelineConnection(
                producer_node="lutEncoder",
                producer_node_io="GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT",
                consumer_node="textGenerator",
                consumer_node_io="GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
            ),
        ]

        sample_inputs = [
            GenieSampleInput(
                node="lutEncoder",
                node_io="GENIE_NODE_TEXT_ENCODER_TEXT_INPUT",
                file="sample_inputs/prompt_prefix.txt",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT",
                file="sample_inputs/pixel_values.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS",
                file="sample_inputs/position_ids_cos.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN",
                file="sample_inputs/position_ids_sin.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
                file="sample_inputs/window_attention_mask.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK",
                file="sample_inputs/full_attention_mask.raw",
            ),
            GenieSampleInput(
                node="lutEncoder",
                node_io="GENIE_NODE_TEXT_ENCODER_TEXT_INPUT",
                file="sample_inputs/prompt_suffix.txt",
            ),
        ]

        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**chat_spec),
            context_lengths=context_lengths,
            supports_streaming=True,
            supports_vision=True,
            supports_thinking=False,
            pipeline=GeniePipeline(
                nodes=pipeline_nodes,
                connections=pipeline_connections,
            ),
            sample_inputs=sample_inputs,
            vision_preprocessing=GenieVisionPreprocessing(
                image_width=DEFAULT_IMAGE_WIDTH,
                image_height=DEFAULT_IMAGE_HEIGHT,
                patch_size=image_processor.patch_size,
                temporal_patch_size=image_processor.temporal_patch_size,
                spatial_merge_size=image_processor.merge_size,
                normalize_mean=image_processor.image_mean,
                normalize_std=image_processor.image_std,
            )
            if image_processor is not None
            else None,
        )

        # Generate genie-app-script.txt from the same pipeline data.
        genie_script = generate_genie_app_script(
            pipeline_nodes, pipeline_connections, sample_inputs
        )
        with open(output_dir / "genie-app-script.txt", "w") as f:
            f.write(genie_script)
        metadata.supplementary_files["genie-app-script.txt"] = (
            "Genie-app pipeline script for VLM inference."
        )

        # --- Sample VEG inputs (inputs/ directory) ---
        self._write_sample_veg_inputs(output_dir)

    @staticmethod
    def _write_sample_veg_inputs(output_dir: str | os.PathLike) -> None:
        """Generate sample VEG input .raw files in inputs/ for genie-app."""
        from transformers import AutoProcessor

        inputs_dir = Path(output_dir) / "sample_inputs"
        inputs_dir.mkdir(exist_ok=True)

        # Fetch sample image from S3 asset store
        from qai_hub_models.utils.asset_loaders import load_image

        img = load_image(SAMPLE_IMAGE)
        img_resized = img.resize((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))

        # Patchify + normalize via HF processor
        from qai_hub_models.models._shared.qwen2_vl.model import Qwen2VLTextBase

        proc = AutoProcessor.from_pretrained(HF_REPO_NAME)
        dummy_text = Qwen2VLTextBase.get_input_prompt_with_tags(
            user_input_prompt="", include_image=True
        )
        processed = proc(text=[dummy_text], images=[img_resized], return_tensors="pt")

        # RoPE and attention masks from VisionEncoder
        from qai_hub_models.models.qwen2_5_vl_7b_instruct import VisionEncoder

        veg = VisionEncoder.from_pretrained(device=torch.device("cpu"))
        veg.eval()

        raw_files = {
            "pixel_values.raw": processed["pixel_values"],
            "position_ids_cos.raw": veg._pos_emb_cos.cpu().float(),
            "position_ids_sin.raw": veg._pos_emb_sin.cpu().float(),
            "window_attention_mask.raw": veg._window_attention_mask.cpu().float(),
            "full_attention_mask.raw": veg._full_attention_mask.cpu().float(),
        }
        for name, tensor in raw_files.items():
            tensor.detach().numpy().astype(np.float32).tofile(inputs_dir / name)
        del veg

        # Prompt text files (real newlines required for tokenizer)
        prompt_prefix = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|>"
        )
        prompt_suffix = (
            "<|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n"
        )
        (inputs_dir / "prompt_prefix.txt").write_text(prompt_prefix)
        (inputs_dir / "prompt_suffix.txt").write_text(prompt_suffix)

        print(f"Wrote VEG sample inputs to {inputs_dir}/")

    @classmethod
    def prepare_genie_assets(cls, **kwargs: Any) -> None:
        # All genie assets are produced by write_supplementary_files above.
        # The parent class would overwrite genie_config.json with "dialog"
        # key, but VLM pipeline requires "text-generator" key.
        pass
