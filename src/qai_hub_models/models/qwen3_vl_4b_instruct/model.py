# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import itertools
import json
import logging

from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphWorkbenchModelCollection,
)
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphWorkbenchModel,
)

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on
import os
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoProcessor
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
)
from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    GeniePipeline,
    GeniePipelineConnection,
    GenieSampleInput,
    GenieVisionPreprocessing,
    ModelMetadata,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.llm_helpers import (
    create_genie_config,
    export_embedding_weights_from_tensor,
    generate_genie_app_script,
    get_rope_scaling,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    LLMDynamic_AIMETOnnx,
    LLMPartBase,
    SingleSlotCacheMixin,
    SplitForwardMixin,
    get_onnx_model,
    get_tokenizer,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models._shared.qwen3_vl.model import (
    HubCompatibleQwen3VLGenerator,
    Qwen3VLDynamic_AIMETOnnx,
    Qwen3VLTextBase,
)
from qai_hub_models.models._shared.qwen3_vl.vision_encoder import (
    Qwen3VLVisionEncoder,
)
from qai_hub_models.models._shared.vlm.model import DEFAULT_IMAGE_SIZE
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import (
    BaseModel,
)
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)

# Model architecture constants (from Qwen3-VL-4B-Instruct)
NUM_LAYERS = 36
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 2560
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32
HEAD_DIM = 128
NUM_DEEPSTACK_LAYERS = 3

# Vision encoder configuration
VISION_HIDDEN_SIZE = 1280
VISION_OUT_HIDDEN_SIZE = 2560
VISION_DEPTH = 32
VISION_NUM_HEADS = 16
VISION_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-VL-4B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 40

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT: dict = {
    Precision.w4a16: "w4a16",
}

# Default image dimensions (must be divisible by patch_size * spatial_merge_size)
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 512


def num_visual_tokens_for_image_size(image_size: tuple[int, int]) -> int:
    """Post-merge visual token count for an image: (W/patch)*(H/patch)/merge^2.

    ``image_size`` is ``(width, height)`` to match the dataset/eval convention
    (PIL ``Image.resize`` takes ``(width, height)``).
    """
    width, height = image_size
    return (
        (height // VISION_PATCH_SIZE)
        * (width // VISION_PATCH_SIZE)
        // (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE)
    )


DEFAULT_NUM_VISUAL_TOKENS = num_visual_tokens_for_image_size(
    (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
)

SPLIT_MODEL_NAME = "Qwen3_VL_4B"


# ---------------------------------------------------------------------------
# Qwen3_VL_4B_PreSplit - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen3_VL_4B_PreSplit(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Qwen3VLTextBase
):
    """
    FP PreSplit for Qwen3-VL-4B.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint. VLM uses split_embedding=False since inputs_embeds
    bypasses the embedding layer.
    """

    GeneratorClass = HubCompatibleQwen3VLGenerator

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_embedding = False

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision = DEFAULT_PRECISION,
    ) -> tuple[float | None, float]:
        return (-250.0, 1.0)

    _hf_repo_name: str = HF_REPO_NAME

    def __init__(
        self,
        checkpoint: str | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

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
    ) -> Qwen3_VL_4B_PreSplit:
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
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
            cls.release()
            raise
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_spec(self) -> OutputSpec:
        return Qwen3VLTextBase._get_output_spec(NUM_LAYERS)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return self.get_static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type, image_size
        )

    @staticmethod
    def get_static_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_attention_heads": NUM_ATTN_HEADS,
            }
        return Qwen3VLTextBase._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", NUM_LAYERS),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config.get("hidden_size", HIDDEN_SIZE),
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", NUM_KEY_VALUE_HEADS
            ),
            num_attention_heads=llm_config.get("num_attention_heads", NUM_ATTN_HEADS),
            head_dim=llm_config.get("head_dim", HEAD_DIM),
            llm_io_type=llm_io_type,
            num_deepstack_layers=NUM_DEEPSTACK_LAYERS,
            num_visual_tokens=num_visual_tokens_for_image_size(image_size),
        )

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes."""
        from torch.export import Dim

        seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
        num_visual_tokens = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]

        extra_dynamic_shapes: dict[str, dict[int, Any]] = {
            "visual_pos_masks": {1: seq_len},
        }
        for i in range(NUM_DEEPSTACK_LAYERS):
            extra_dynamic_shapes[f"deepstack_visual_embeds_{i}"] = {
                0: num_visual_tokens
            }

        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=DEFAULT_CONTEXT_LENGTH,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            use_dynamic_shapes=True,
            extra_dynamic_shapes=extra_dynamic_shapes,
        )
        return ONNXBundle.from_bundle_path(onnx_dir, "model")


# ---------------------------------------------------------------------------
# Qwen3_VL_4B_QuantizablePreSplit - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen3_VL_4B_QuantizablePreSplit(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin["Qwen3_VL_4B_PreSplit"],
    Qwen3VLDynamic_AIMETOnnx,
):
    """
    Quantizable PreSplit for Qwen3-VL-4B.

    The S3 asset zip contains the FULL output of quantize.py (dynamic
    ONNX + weights + encodings + tokenizer + config + embedding_weights.raw).
    """

    FPModel = Qwen3_VL_4B_PreSplit  # type: ignore[assignment]
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
    split_embedding = False

    # SHA produces per-head q_norm/k_norm nodes in the ONNX graph.
    # Between block starts (input_layernorm): NUM_ATTN_HEADS q_norms
    # + NUM_KEY_VALUE_HEADS k_norms + 1 post_attention_layernorm = 41 intermediate ops
    ada_scale_num_rmsnorm_per_blk: int | None = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        return (-100, 1.0)

    def get_output_spec(self) -> OutputSpec:
        return Qwen3VLTextBase._get_output_spec(NUM_LAYERS)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return self.get_static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type, image_size
        )

    @classmethod
    def get_static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return cls.FPModel.get_static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
            image_size=image_size,
        )

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: Qwen3_VL_4B_PreSplit | None = None,
    ) -> None:
        """Save calibrated checkpoint with ONNX, encodings, and embedding weights."""
        if fp_model is None:
            fp_model = Qwen3_VL_4B_PreSplit.from_pretrained()
        super().save_calibrated_checkpoint(output_checkpoint, fp_model)

        # VLM-specific: embedding table is needed for on-device LUT encoder
        export_embedding_weights_from_tensor(
            fp_model.get_embedding_weights().float(), Path(output_checkpoint)
        )


# ---------------------------------------------------------------------------
# Vision Encoder Component
# ---------------------------------------------------------------------------


class Qwen3_VL_4B_VisionEncoder(Qwen3VLVisionEncoder):
    """
    Vision encoder for Qwen3-VL-4B (adapted VEG for on-device deployment).

    Returns multiple outputs: image_embeddings + deepstack features.
    Supports both FP inference and quantized inference (via AIMET-ONNX QuantSim).
    """

    DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    _hf_repo_name: str = HF_REPO_NAME

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
    ) -> Qwen3_VL_4B_VisionEncoder:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if precision != Precision.float and (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            checkpoint = Qwen3_VL_4B_QuantizablePreSplit.fetch_default_checkpoint(
                precision
            )

        load_device = device if precision == Precision.float else torch.device("cpu")
        instance: Qwen3_VL_4B_VisionEncoder = super().from_pretrained(  # type: ignore[assignment]
            checkpoint=cls._hf_repo_name,
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
        """Create an AIMET-ONNX QuantSim session for quantized inference."""
        import logging

        from aimet_onnx.common.defs import QuantScheme
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

    def component_precision(self) -> Precision:
        return self._precision

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
    ) -> tuple[torch.Tensor, ...]:
        if self._is_quantized:
            return self._forward_quantized(pixel_values)
        return super().forward(
            pixel_values=pixel_values,
            position_ids_cos=position_ids_cos,
            position_ids_sin=position_ids_sin,
            window_attention_mask=window_attention_mask,
            full_attention_mask=full_attention_mask,
        )

    def _forward_quantized(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run inference through the AIMET-ONNX QuantSim session."""
        assert self._quantized_session is not None
        results = mock_torch_onnx_inference(
            self._quantized_session.session,
            pixel_values,
            cast(torch.Tensor, self._pos_emb_cos),
            cast(torch.Tensor, self._pos_emb_sin),
            cast(torch.Tensor, self._window_attention_mask),
            cast(torch.Tensor, self._full_attention_mask),
        )
        if isinstance(results, torch.Tensor):
            return (results,)
        return tuple(results)

    def get_input_spec(
        self,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
    ) -> InputSpec:
        return self.get_static_input_spec(image_height, image_width)

    @staticmethod
    def get_static_input_spec(
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
    ) -> InputSpec:
        return Qwen3VLVisionEncoder.get_static_input_spec(
            image_height=image_height,
            image_width=image_width,
            patch_size=VISION_PATCH_SIZE,
        )

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

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = "Qwen3_VL_4B_VisionEncoder"

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_dir) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return out_dir

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


class Qwen3_VL_4B_PartBase(LLMPartBase, torch.nn.Module, MultiGraphWorkbenchModel):
    """
    Unified Part base: handles both FP and Quantizable modes based on precision.

    Spec derivation is inherited from ``LLMPartBase`` (head_dim attribute +
    ``_extra_graph_inputs`` hook); this class carries the family deploy/session
    plumbing (mirroring ``LlamaPartBase`` for text LLMs) plus the qwen3
    architecture constants and the deepstack graph-input override.
    """

    # Architecture dims (LLMPartBase attribute names; head_dim is explicit
    # because 2560 / 32 != 128).
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    part_id: int = 0

    def __init__(
        self,
        presplit: Qwen3_VL_4B_PreSplit | Qwen3_VL_4B_QuantizablePreSplit,
        precision: Precision = DEFAULT_PRECISION,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._graph_names: dict[str, tuple[int, int]] = {
            f"ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{NUM_SPLITS}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(
                DEFAULT_EXPORT_SEQUENCE_LENGTHS, DEFAULT_EXPORT_CONTEXT_LENGTHS
            )
        }

    @property
    def shared_source_model(self) -> bool:
        return True

    @property
    def graph_names(self) -> list[str]:
        return list(self._graph_names.keys())

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: Qwen3_VL_4B_PreSplit | Qwen3_VL_4B_QuantizablePreSplit = (
                Qwen3_VL_4B_PreSplit.from_pretrained(
                    host_device=host_device,
                )
            )
            precision = Precision.float
        else:
            precision = checkpoint_type.precision(
                DEFAULT_PRECISION, checkpoint=checkpoint
            )
            presplit = Qwen3_VL_4B_QuantizablePreSplit.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(presplit, precision=precision)

    def _extra_graph_inputs(
        self, name: str, sequence_length: int, context_length: int
    ) -> TensorSpec | None:
        # Deepstack-specific inputs (qwen3 VL only).
        if name == "visual_pos_masks":
            return TensorSpec(shape=(1, sequence_length), dtype="bool")
        if name.startswith("deepstack_visual_embeds_"):
            return TensorSpec(
                shape=(DEFAULT_NUM_VISUAL_TOKENS, HIDDEN_SIZE), dtype="float32"
            )
        return None

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

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    def _get_quant_sim(self) -> QuantizationSimModel:
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )
        onnx_model.ir_version = min(onnx_model.ir_version, 11)

        assert isinstance(self._presplit, Qwen3_VL_4B_QuantizablePreSplit)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim,
                str(onnx_bundle.aimet_encodings_path),
                strict=False,
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()
        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        precision_suffix = f"_{self._precision}" if self._is_quantized else ""
        out_dir = Path(output_dir) / f"{model_name}{precision_suffix}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )

        return out_dir


# Concrete Part classes
class Qwen3_VL_4B_Part1_Of_4(Qwen3_VL_4B_PartBase):
    part_id = 1


class Qwen3_VL_4B_Part2_Of_4(Qwen3_VL_4B_PartBase):
    part_id = 2


class Qwen3_VL_4B_Part3_Of_4(Qwen3_VL_4B_PartBase):
    part_id = 3


class Qwen3_VL_4B_Part4_Of_4(Qwen3_VL_4B_PartBase):
    part_id = 4


# ---------------------------------------------------------------------------
# Split-Forward Wrappers (for ONNX-based evaluation)
# ---------------------------------------------------------------------------


class _Qwen3VLSplitForwardMixin(SplitForwardMixin):
    def get_split_part_classes(self) -> list[type]:
        return [
            Qwen3_VL_4B_Part1_Of_4,
            Qwen3_VL_4B_Part2_Of_4,
            Qwen3_VL_4B_Part3_Of_4,
            Qwen3_VL_4B_Part4_Of_4,
        ]

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._exporting_onnx or torch.compiler.is_compiling():
            return super(SplitForwardMixin, self).forward(  # type: ignore[misc]
                input_tokens, attention_mask, *args
            )
        self._ensure_parts()
        assert self._parts is not None
        assert self._input_names_for_parts is not None

        full_names = list(
            self.get_input_spec(  # type: ignore[attr-defined]
                sequence_length=DEFAULT_SEQUENCE_LENGTH,
                context_length=DEFAULT_CONTEXT_LENGTH,
            ).keys()
        )
        # Total positional args = input_tokens + attention_mask + *args
        num_provided = 2 + len(args)
        num_expected = len(full_names)

        # Pad missing deepstack inputs with zeros using actual runtime shapes.
        # visual_pos_masks=0 means no visual tokens, so deepstack is a no-op.
        if num_provided < num_expected:
            actual_seq_len = input_tokens.shape[1]
            device = input_tokens.device
            extra = []
            for name in full_names[num_provided:]:
                if name == "visual_pos_masks":
                    extra.append(
                        torch.zeros(1, actual_seq_len, dtype=torch.bool, device=device)
                    )
                elif name.startswith("deepstack_visual_embeds_"):
                    extra.append(
                        torch.zeros(1, DEFAULT_NUM_VISUAL_TOKENS, device=device)
                    )
                else:
                    extra.append(torch.zeros(1, device=device))
            args = (*args, *extra)

        return self._split_forward(
            self._parts,
            self._input_names_for_parts,
            input_tokens,
            attention_mask,
            *args,
        )


class FPSplitModelWrapper(_Qwen3VLSplitForwardMixin, Qwen3_VL_4B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    _Qwen3VLSplitForwardMixin, Qwen3_VL_4B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class Qwen3_VL_4B_Collection(MultiGraphWorkbenchModelCollection):
    """
    Collection model for Qwen3-VL-4B deployment.

    Combines 6 text parts + 1 vision encoder for full VLM deployment.
    """

    _checkpoint: str

    def __init__(
        self,
        vision_encoder: Qwen3_VL_4B_VisionEncoder,
        part1: Qwen3_VL_4B_Part1_Of_4,
        part2: Qwen3_VL_4B_Part2_Of_4,
        part3: Qwen3_VL_4B_Part3_Of_4,
        part4: Qwen3_VL_4B_Part4_Of_4,
    ) -> None:
        super().__init__(
            {
                "vision_encoder": vision_encoder,
                "part1_of_4": part1,
                "part2_of_4": part2,
                "part3_of_4": part3,
                "part4_of_4": part4,
            }
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        **kwargs: Any,
    ) -> Qwen3_VL_4B_Collection:
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        precision = (
            checkpoint_type.precision(DEFAULT_PRECISION, checkpoint=checkpoint)
            if checkpoint_type.is_aimet_onnx()
            else Precision.float
        )

        part_kwargs: dict[str, Any] = dict(
            checkpoint=checkpoint,
            host_device=host_device,
        )
        parts: list[BaseModel | MultiGraphWorkbenchModel] = []
        for part_cls in [
            Qwen3_VL_4B_VisionEncoder,
            Qwen3_VL_4B_Part1_Of_4,
            Qwen3_VL_4B_Part2_Of_4,
            Qwen3_VL_4B_Part3_Of_4,
            Qwen3_VL_4B_Part4_Of_4,
        ]:
            if issubclass(part_cls, Qwen3_VL_4B_VisionEncoder):
                parts.append(
                    part_cls.from_pretrained(
                        checkpoint=checkpoint,
                        device=host_device,
                        precision=precision,
                    )
                )
            else:
                parts.append(part_cls.from_pretrained(**part_kwargs))  # type: ignore[attr-defined]
        instance = cls(*parts)  # type: ignore[arg-type]
        resolved_checkpoint: str | Path = checkpoint
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            for comp in parts:
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
        output_dir = Path(output_dir)
        checkpoint_path = Path(self._checkpoint)

        # --- Embedding weights ---
        embed_src = checkpoint_path / "embedding_weights.raw"
        if embed_src.exists():
            shutil.copy(embed_src, output_dir / "embedding_weights.raw")
            print("Copied embedding table from checkpoint")
        else:
            fp_model = Qwen3_VL_4B_PreSplit.from_pretrained()
            export_embedding_weights_from_tensor(
                fp_model.get_embedding_weights().float(), output_dir
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
        tokenizer = get_tokenizer(HF_REPO_NAME)
        sample_prompt = Qwen3VLTextBase.get_input_prompt_with_tags(
            include_image=False,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )
        with open(output_dir / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample text-only prompt for standalone genie-t2t-run."
        )

        # --- HTP backend extension config ---
        device_info: dict[str, str] = {}
        if metadata.chipset_attributes:
            ca = metadata.chipset_attributes
            if ca.htp_version is not None:
                device_info["hexagon"] = f"v{ca.htp_version}"
            if ca.soc_model is not None:
                device_info["soc-model"] = str(ca.soc_model)
        if save_htp_config_for_genie_bundle(device_info, output_dir):
            metadata.supplementary_files["htp_backend_ext_config.json"] = (
                "HTP backend extension config for Genie."
            )

        # --- Genie config (text-dec-htp.json equivalent) ---
        context_length: int = 0
        for file_meta in metadata.model_files.values():
            if "attention_mask" in file_meta.inputs:
                attn_shape = file_meta.inputs["attention_mask"].shape
                context_length = max(context_length, attn_shape[3])

        image_processor = None
        llm_config = None
        for comp in self.components.values():
            if isinstance(comp, Qwen3_VL_4B_PartBase):
                presplit = comp._presplit
                image_processor = getattr(presplit, "_image_processor", None)
                llm_config = getattr(
                    presplit, "_original_llm_config", presplit.llm_config
                )
                break

        if image_processor is None:
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

        # Build VLM MRoPE config from the HF config. transformers 5.x nests
        # rope settings (incl. mrope_section) under rope_parameters; get_rope_scaling
        # reads either layout.
        rope_scaling = get_rope_scaling(text_config)
        # Qwen3-VL uses *interleaved* MRoPE (mrope_interleaved=True), which Genie
        # implements only under "qwen3vl-mrope" (nsp-model.cpp). "qwen2vl-mrope"
        # applies a different, contiguous sectioning and would corrupt positions.
        vlm_rope_config: dict[str, Any] = {
            "rope-type": "qwen3vl-mrope",
            "time-step": 50,
        }
        vlm_rope_config["spatial-merge-size"] = image_processor.merge_size
        if rope_scaling is not None and "mrope_section" in rope_scaling:
            vlm_rope_config["mrope-section"] = rope_scaling["mrope_section"]

        # text-generator.json: used by genie-app-script.txt (genie-app VLM pipeline)
        genie_config = create_genie_config(
            context_length=context_length,
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
            context_length=context_length,
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

        # --- Image encoder config (img-enc-htp.json) ---
        veg_bins = sorted(
            fn
            for fn in metadata.model_files
            if fn.startswith("vision_encoder") and fn.endswith(".bin")
        )
        img_enc_config: dict[str, Any] = {
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
        chat_spec = Qwen3VLTextBase.get_chat_template()

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

        # Deepstack connection: a single GENIE_NODE_WILDCARD <-> GENIE_NODE_WILDCARD
        # connection. Genie has no dedicated deepstack node-IO enums; instead its
        # InjectiveConnector auto-routes every tensor whose name appears in BOTH
        # the producer's outputs and the consumer's inputs. The VEG outputs
        # ``deepstack_visual_embeds_{0..N-1}`` (+ ``visual_pos_masks``) and the
        # text generator consumes the same names, so one wildcard connection
        # carries all deepstack features by name. This must come AFTER the primary
        # EMBEDDING_OUTPUT->EMBEDDING_INPUT connection above (Genie requires a
        # primary connection before a wildcard), and only one wildcard per node.
        if NUM_DEEPSTACK_LAYERS > 0:
            pipeline_connections.append(
                GeniePipelineConnection(
                    producer_node="imageEncoder",
                    producer_node_io="GENIE_NODE_WILDCARD",
                    consumer_node="textGenerator",
                    consumer_node_io="GENIE_NODE_WILDCARD",
                )
            )

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
            context_lengths=[context_length],
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
            ),
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

        # --- Sample VEG inputs (sample_inputs/ directory) ---
        self._write_sample_veg_inputs(output_dir)

    @staticmethod
    def _write_sample_veg_inputs(output_dir: str | os.PathLike) -> None:
        """Generate sample VEG input .raw files in sample_inputs/ for genie-app."""
        inputs_dir = Path(output_dir) / "sample_inputs"
        inputs_dir.mkdir(exist_ok=True)

        img = load_image(SAMPLE_IMAGE)
        img_resized = img.resize((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))

        # Patchify + normalize via HF processor
        proc = AutoProcessor.from_pretrained(HF_REPO_NAME)
        tokenizer = get_tokenizer(HF_REPO_NAME)
        dummy_text = Qwen3VLTextBase.get_input_prompt_with_tags(
            user_input_prompt="",
            include_image=True,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )
        processed = proc(text=[dummy_text], images=[img_resized], return_tensors="pt")

        # Instantiate VEG to get pre-computed position/attention buffers
        veg = Qwen3_VL_4B_VisionEncoder.from_pretrained(device=torch.device("cpu"))
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

        # Prompt text files
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
