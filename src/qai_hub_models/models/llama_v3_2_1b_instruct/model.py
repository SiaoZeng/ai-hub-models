# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Llama 3.2 1B Instruct - PreSplit-Part architecture for LLM deployment.

Architecture:
- Llama3_2_1B_PreSplit (Singleton, FP): Manages full model + ONNX splitting
- Llama3_2_1B_QuantizablePreSplit (Singleton): Manages QuantSim + calibration
- Llama3_2_1B_PartBase -> Part1, Part2, Part3: Unified split inference
  (handles both FP and Quantizable modes based on precision)
- Collection class for deploying as 3 splits
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os

from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
from qai_hub.client import Device
from transformers import AutoConfig, AutoTokenizer
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    ModelMetadata,
)
from qai_hub_models.models._shared.llama3.model import (
    Llama3DynamicBase,
    Llama3DynamicBase_AIMETOnnx,
    LlamaDynamicQuantizablePreSplitMixin,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    LLMDynamic_AIMETOnnx,
    SingleSlotCacheMixin,
    SplitForwardMixin,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphCollectionModel
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.llm_helpers import (
    create_genie_config,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 6

# Model architecture constants (from Llama 3.2 1B)
NUM_LAYERS = 16
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 8
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging Face repo
HF_REPO_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 50

# Precision settings
DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4, Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4: "w4",
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_2_1B_1_of_3.onnx)
SPLIT_MODEL_NAME = "Llama3_2_1B"

# ---------------------------------------------------------------------------
# Llama3_2_1B_PreSplit - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Llama3_2_1B_PreSplit(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Llama3DynamicBase
):
    """
    FP PreSplit for Llama 3.2 1B.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.
    """

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    def __init__(
        self,
        checkpoint: str | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

    def _verify_ckpt(self) -> None:
        """Verify checkpoint compatibility."""
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = HF_REPO_NAME,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Llama3_2_1B_PreSplit:
        """
        Load or return a cached FP PreSplit.

        Uses dynamic shapes so sequence_length/context_length are not
        needed at construction time.
        """
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached

        instance = cls(
            checkpoint=checkpoint,
            host_device=host_device,
            load_pretrained=True,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_names(NUM_LAYERS)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return self.get_static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type
        )

    @staticmethod
    def get_static_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Model configuration dictionary.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        llm_io_type
            Input/output type for the LLM.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_attention_heads": NUM_ATTN_HEADS,
            }
        return Llama3DynamicBase._get_input_spec(
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


# ---------------------------------------------------------------------------
# Llama3_2_1B_QuantizablePreSplit - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Llama3_2_1B_QuantizablePreSplit(  # type: ignore[misc]
    LlamaDynamicQuantizablePreSplitMixin[Llama3_2_1B_PreSplit],
    Llama3DynamicBase_AIMETOnnx,
):
    """
    Quantizable PreSplit for Llama 3.2 1B.

    Manages QuantSim and calibration. Uses class-level cache keyed by
    checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.
    """

    FPModel = Llama3_2_1B_PreSplit  # type: ignore[assignment]

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

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_names(NUM_LAYERS)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        if bundle.aimet_encodings_path is not None:
            self._adapt_aimet_encodings(
                str(bundle.aimet_encodings_path),
                str(bundle.aimet_encodings_path),
                str(bundle.onnx_graph_path),
            )
        return super()._postprocess_full_onnx_bundle(bundle)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Model configuration dictionary.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        llm_io_type
            Input/output type for the LLM.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        return self.FPModel.get_static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )


# ---------------------------------------------------------------------------
# Unified Part Base & Concrete Parts
# ---------------------------------------------------------------------------


class Llama3_2_1B_PartBase(torch.nn.Module, MultiGraphWorkbenchModel):
    """
    Unified Part base: handles both FP and Quantizable modes based on precision.

    Each Part represents one split of the ONNX model for deployment.
    When precision is float, uses the FP PreSplit (ONNX ModelProto inference).
    When precision is quantized, uses the Quantizable PreSplit (ONNXBundle + encodings).
    """

    part_id: int = 0  # Override in subclasses (1-indexed)

    def __init__(
        self,
        presplit: Llama3_2_1B_PreSplit | Llama3_2_1B_QuantizablePreSplit,
        precision: Precision = DEFAULT_PRECISION,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._sequence_lengths = sequence_lengths
        self._context_lengths = context_lengths
        self._graph_names: dict[str, tuple[int, int]] = {
            f"{'token' if seq_len == 1 else 'prompt'}_ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{NUM_SPLITS}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(
                self._sequence_lengths, self._context_lengths
            )
        }

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
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: Llama3_2_1B_PreSplit | Llama3_2_1B_QuantizablePreSplit = (
                Llama3_2_1B_PreSplit.from_pretrained(host_device=host_device)
            )
            precision = Precision.float
        else:
            precision = checkpoint_type.precision(
                DEFAULT_PRECISION, checkpoint=checkpoint
            )
            presplit = Llama3_2_1B_QuantizablePreSplit.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(
            presplit,
            precision=precision,
            context_lengths=context_lengths,
            sequence_lengths=sequence_lengths,
        )

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        sequence_length, context_length = self._graph_names[graph_name]
        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": ((1, sequence_length), "int32")}
        head_dim = HIDDEN_SIZE // NUM_ATTN_HEADS
        embed_dim = head_dim // 2
        kv_seq_len = context_length - sequence_length

        # Read actual input names from the split ONNX model
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
                    (1, 1, sequence_length, context_length),
                    "float32",
                )
            elif "position_ids_cos" in name or "position_ids_sin" in name:
                spec[name] = (
                    (1, 1, sequence_length, embed_dim),
                    "float32",
                )
            else:
                # Intermediate hidden state from previous part
                # (found by process of elimination)
                spec[name] = (
                    (1, sequence_length, HIDDEN_SIZE),
                    "float32",
                )

        return spec

    def get_graph_output_names(self, graph_name: str) -> list[str]:
        """Get output names for this specific Part instance.

        Names are read from the actual split ONNX model at runtime.
        """
        # All graphs have the same outputs.
        return [
            name.replace("/", "_").replace(".", "_")
            for name in self._get_onnx_output_names()
        ]

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        """Get sample inputs for this specific part only.

        Uses actual ONNX input names read from the split model at runtime.
        When called from the multi-graph sample_inputs path, input_spec
        carries the per-graph shapes so we derive seq_len from it.
        """
        # Derive seq_len from input_spec when available (multi-graph path).
        if input_spec is not None and "input_ids" in input_spec:
            seq_len = input_spec["input_ids"][0][1]  # shape (1, seq_len)
        else:
            seq_len, _ = self._graph_names[graph_name]

        full_inputs = self._presplit._sample_inputs_impl()

        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": [np.zeros((1, seq_len), dtype=np.int32)]}

        # Parts 2+: read actual input names from ONNX and match them
        result: SampleInputsType = {}
        onnx_input_names = self._get_onnx_input_names()

        for name in onnx_input_names:
            if name in full_inputs:
                result[name] = full_inputs[name]
            else:
                # Intermediate hidden state (not in full model inputs)
                # found by process of elimination
                result[name] = [np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)]

        return result

    # -------------------------------------------------------------------
    # Methods that branch on self._is_quantized
    # -------------------------------------------------------------------

    def _get_onnx_input_names(self) -> list[str]:
        """Read actual input names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        """Read actual output names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        """Get ONNXBundle for this Part (works for both FP and quantized)."""
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def _get_quant_sim(self) -> QuantizationSimModel:
        """Get or create QuantSim for this specific part from its ONNXBundle."""
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()

        # Load ONNX model
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Clamp to keep QuantSim compatible.
        onnx_model.ir_version = min(onnx_model.ir_version, 10)

        assert isinstance(self._presplit, Llama3_2_1B_QuantizablePreSplit)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        # Use shared construction + activation config. We skip the
        # full-model _configure_quant_sim (lm_head, KV tying) since
        # those heuristics misfire on split models.
        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        # Load encodings if available
        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim, str(onnx_bundle.aimet_encodings_path), strict=False
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        """Get or create an ORT session for FP inference (cached)."""
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
        """Forward pass for this Part (FP or quantized based on precision)."""
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    @property
    def shared_source_model(self) -> bool:
        return True

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Export ONNX model for this Part."""
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        # Include precision in directory name to avoid cache collisions
        # between different precisions sharing the same output_path.
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

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super().get_graph_hub_compile_options(
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " --quantize_full_type w8a16 --quantize_io",
            device,
        )

    def get_graph_hub_profile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        if self._is_quantized:
            return self._presplit.get_hub_profile_options(
                target_runtime=target_runtime,
                other_profile_options=other_profile_options,
                context_graph_name=graph_name,
            )
        return super().get_graph_hub_profile_options(
            graph_name,
            target_runtime=target_runtime,
            other_profile_options=other_profile_options,
        )


class Llama3_2_1B_Part1_Of_3(Llama3_2_1B_PartBase):
    """Part 1: Embedding + first layers."""

    part_id = 1


class Llama3_2_1B_Part2_Of_3(Llama3_2_1B_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Llama3_2_1B_Part3_Of_3(Llama3_2_1B_PartBase):
    """Part 3: Final layers + LM head."""

    part_id = 3


class _Llama3SplitForwardMixin(SplitForwardMixin):
    """Llama-specific split-forward: returns the 3 concrete Part classes."""

    def get_split_part_classes(self) -> list[type]:
        return [
            Llama3_2_1B_Part1_Of_3,
            Llama3_2_1B_Part2_Of_3,
            Llama3_2_1B_Part3_Of_3,
        ]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    _Llama3SplitForwardMixin, Llama3_2_1B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""


class FPSplitModelWrapper(_Llama3SplitForwardMixin, Llama3_2_1B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""


# ---------------------------------------------------------------------------
# Collection Class
# ---------------------------------------------------------------------------


@MultiGraphCollectionModel.add_component(
    Llama3_2_1B_Part1_Of_3, "part1_of_3", cli_args_prefix=""
)
@MultiGraphCollectionModel.add_component(
    Llama3_2_1B_Part2_Of_3, "part2_of_3", cli_args_prefix=""
)
@MultiGraphCollectionModel.add_component(
    Llama3_2_1B_Part3_Of_3, "part3_of_3", cli_args_prefix=""
)
class Llama3_2_1B_Collection(MultiGraphCollectionModel):
    """
    Unified Collection with 3 Parts for Llama 3.2 1B.

    Supports both FP and Quantizable modes based on precision parameter.
    All Parts share the same PreSplit via class-level cache for memory efficiency.
    """

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
    ) -> Self:
        """
        Create Collection with all 3 Parts.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"``
            to create from HuggingFace.
        host_device
            Device for computation.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).
        sequence_lengths
            Sequence lengths to compile for.
        context_lengths
            Context lengths to compile for.

        Returns
        -------
        Self
            The Collection with all 3 Parts.
        """
        parts = [
            part_cls.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
                sequence_lengths=sequence_lengths,
                context_lengths=context_lengths,
            )
            for part_cls in cls.component_classes.values()
        ]
        return cls(*parts)

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        output_path = Path(output_dir)

        # Save tokenizer and config from HuggingFace (skip if already present)
        if not (output_path / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(HF_REPO_NAME)
            tokenizer.save_pretrained(output_path)
        if not (output_path / "config.json").exists():
            llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
            llm_config.save_pretrained(output_path)
        else:
            llm_config = AutoConfig.from_pretrained(str(output_path))

        # Derive context_length from the first part
        first_part = next(iter(self.components.values()))
        assert isinstance(first_part, Llama3_2_1B_PartBase)
        context_length: int = first_part._presplit.context_length

        # Build genie_config.json
        model_list = list(metadata.model_files.keys())
        config = create_genie_config(context_length, llm_config, "rope", model_list)
        with open(output_path / "genie_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Build htp_backend_ext_config.json from chipset attributes
        device_info: dict[str, str] = {}
        if metadata.chipset_attributes:
            ca = metadata.chipset_attributes
            if ca.htp_version is not None:
                device_info["hexagon"] = f"v{ca.htp_version}"
            if ca.soc_model is not None:
                device_info["soc-model"] = str(ca.soc_model)
        if save_htp_config_for_genie_bundle(device_info, output_path):
            metadata.supplementary_files["htp_backend_ext_config.json"] = (
                "HTP backend configuration for the target device."
            )

        # Write sample_prompt.txt for on-device genie-t2t-run
        tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        sample_prompt = Llama3_2_1B_PreSplit.get_input_prompt_with_tags(
            tokenizer=tokenizer
        )
        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

        chat_spec = Llama3_2_1B_PreSplit.get_chat_template()
        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**chat_spec)
            if chat_spec
            else GenieChatTemplate(),
            context_lengths=[context_length],
            supports_streaming=True,
            supports_vision=False,
            supports_thinking=False,
        )

        metadata.supplementary_files["genie_config.json"] = (
            "Genie SDK configuration for on-device LLM inference."
        )
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample prompt for on-device inference."
        )
        metadata.supplementary_files["tokenizer.json"] = (
            "Tokenizer for encoding/decoding text."
        )
