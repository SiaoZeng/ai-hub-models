# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
    LLMDynamic_AIMETOnnx,
)

if TYPE_CHECKING:
    from qai_hub.public_rest_api import DatasetEntries

    from qai_hub_models.utils.input_spec import InputSpec

logger = logging.getLogger(__name__)

# Fallback default calibration/export image size (width, height), matching the
# dataset/eval convention (PIL Image.resize takes (width, height)). Individual
# models author their own default image dimensions and pass them down; this is
# the generic default for callers that don't.
DEFAULT_IMAGE_SIZE: tuple[int, int] = (512, 512)


class VLMDynamic_AIMETOnnx(LLMDynamic_AIMETOnnx):
    """Dynamic-shape AIMET-ONNX base for vision-language models.

    Owns the model-agnostic calibration data pipeline. Subclasses provide
    ``get_input_spec`` (which must accept ``image_size`` and derive its
    visual-token count from it) so the prefill data matches the exported graph.
    """

    def _load_calibration_vision_model(self) -> torch.nn.Module | None:
        """Load the HF vision model for multimodal calibration samples."""
        try:
            from transformers import AutoModel

            hf_repo = getattr(self, "_hf_repo_name", None)
            if hf_repo is None and self.checkpoint is not None:
                hf_repo = self.checkpoint
            if hf_repo is None:
                hf_repo = self.llm_config._name_or_path

            hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            visual = hf_model.visual.eval()
            del hf_model
            return visual
        except Exception:
            logger.warning(
                "Failed to load vision model for calibration; "
                "multimodal samples will use text-only prefill.",
                exc_info=True,
            )
            return None

    def _prefill_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_inputs: int,
        seq_len: int,
        use_vision: bool = False,
        desc: str = "Pre-filling data",
    ) -> list[list[torch.Tensor | np.ndarray]]:
        """Shared prefill loop for calibration and weight optimization.

        Writes prefilled tensors to memory-mapped files to avoid RAM blow-up
        when accumulating large numbers of KV-cache entries. Uses one file
        per input with uniform entry shapes.

        Note: this assumes every prefilled sample yields the same per-input
        shapes (locked from the first sample). Callers must ensure that holds
        -- e.g. by forcing a fixed image size so vision inputs don't vary in
        token count across samples.
        """
        import tempfile

        from tqdm import tqdm

        from qai_hub_models.models._shared.llm.generator_factory import make_generator

        mmap_dir = tempfile.mkdtemp(prefix="vlm_prefill_")

        vision_model = self._load_calibration_vision_model() if use_vision else None

        generator = make_generator(
            self,
            sequence_length=seq_len,
            vision_model=vision_model,
            model_cls=self.FPModel,
        )

        num_entries = 0
        shapes: list[tuple] = []
        dtypes: list[np.dtype] = []
        files: list[Any] = []
        initialized = False

        device = generator.device
        with self.remove_quantization(), torch.no_grad():
            for sample in tqdm(dataloader, total=len(dataloader), desc=desc):
                input_ids, attention_mask, *rest = sample
                prefill_kwargs: dict[str, torch.Tensor | None] = dict(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                if use_vision:
                    pixel_values = rest[1] if len(rest) > 1 else None
                    prefill_kwargs["pixel_values"] = (
                        pixel_values.to(device) if pixel_values is not None else None
                    )
                    prefill_kwargs["image_grid_thw"] = (
                        rest[2] if len(rest) > 2 else None
                    )

                for prefilled_inputs in generator.prefill(**prefill_kwargs):  # type: ignore[arg-type]
                    arrays = [
                        tensor.cpu().numpy()
                        if isinstance(tensor, torch.Tensor)
                        else np.asarray(tensor)
                        for tensor in prefilled_inputs.values()
                    ]
                    if not initialized:
                        for i, arr in enumerate(arrays):
                            shapes.append(arr.shape)
                            dtypes.append(arr.dtype)
                            fpath = os.path.join(mmap_dir, f"input_{i}.bin")
                            # One long-lived handle per input; closed in bulk
                            # after the loop (not a with-block).
                            files.append(open(fpath, "wb"))  # noqa: SIM115
                            files[i].write(arr.tobytes())
                        initialized = True
                    else:
                        for i, arr in enumerate(arrays):
                            if arr.shape != shapes[i]:
                                raise ValueError(
                                    f"Calibration sample {num_entries} input #{i} has "
                                    f"shape {arr.shape}, expected {shapes[i]} (locked "
                                    "from the first sample). Vision inputs must have a "
                                    "fixed shape across samples -- force a fixed "
                                    "image_size."
                                )
                            files[i].write(arr.tobytes())
                    num_entries += 1
                del prefill_kwargs
                gc.collect()
                torch.cuda.empty_cache()

        for f in files:
            f.close()

        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(num_inputs)]
        for i in range(num_inputs):
            fpath = os.path.join(mmap_dir, f"input_{i}.bin")
            mm = np.memmap(
                fpath, dtype=dtypes[i], mode="r", shape=(num_entries, *shapes[i])
            )
            for j in range(num_entries):
                inputs[i].append(mm[j])

        return inputs

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_CALIBRATION_SEQ_LEN,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> DatasetEntries | None:
        """Get interleaved (wikitext + AOKVQA) calibration data for VLM.

        Images are resized to ``image_size`` so the per-sample vision inputs
        have a fixed token count matching the exported input spec.
        """
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor

        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.datasets.wikitext.interleaved_aokvqa_wikitext import (
            InterleavedAOKVQAWikitext,
        )
        from qai_hub_models.utils.base_dataset import DatasetSplit
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        hf_repo = getattr(self, "_hf_repo_name", None)
        if hf_repo is None and self.checkpoint is not None:
            hf_repo = self.checkpoint
        if hf_repo is None:
            hf_repo = self.llm_config._name_or_path
        processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)

        dataset = instantiate_dataset(
            InterleavedAOKVQAWikitext,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
            processor=processor,
            image_size=image_size,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
            image_size=image_size,  # type: ignore[call-arg]
        )
        assert input_spec is not None

        inputs = self._prefill_dataset(
            dataloader,
            num_inputs=len(input_spec),
            seq_len=sequence_length,
            use_vision=True,
            desc="Pre-filling calibration data (interleaved)",
        )
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def get_weight_optimization_data(
        self,
        num_samples: int = 0,
        sequence_length: int = DEFAULT_CALIBRATION_SEQ_LEN,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> DatasetEntries | None:
        """Get plain text (WikiText) data for seqMSE/AdaScale weight optimization."""
        from torch.utils.data import DataLoader

        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.datasets.wikitext import WikiText
        from qai_hub_models.utils.base_dataset import DatasetSplit
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        dataset = instantiate_dataset(
            WikiText,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
            image_size=image_size,  # type: ignore[call-arg]
        )
        assert input_spec is not None

        inputs = self._prefill_dataset(
            dataloader,
            num_inputs=len(input_spec),
            seq_len=sequence_length,
            use_vision=False,
            desc="Pre-filling weight optimization data (text-only)",
        )
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
