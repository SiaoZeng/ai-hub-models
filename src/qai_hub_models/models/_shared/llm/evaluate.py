# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache

from qai_hub_models import Precision
from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamic_AIMETOnnx,
    LLMDynamicBase,
)
from qai_hub_models.utils.args import (
    add_input_spec_args,
    get_model_cli_parser,
    get_model_kwargs,
)
from qai_hub_models.utils.base_dataset import (
    AugmentedLabelDataset,
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.checkpoint import (
    CheckpointType,
)


def get_dataset(
    model: torch.nn.Module,
    dataset_cls: type[BaseDataset],
    num_samples: int,
    processor: Any = None,
    image_size: tuple[int, int] | None = None,
) -> DataLoader[AugmentedLabelDataset]:
    extra_kwargs: dict[str, Any] = {}
    if processor is not None:
        extra_kwargs["processor"] = processor
    if image_size is not None:
        extra_kwargs["image_size"] = image_size
    dataset = instantiate_dataset(
        dataset_cls,
        DatasetSplit.TEST,
        input_spec=None,
        tokenizer=model.tokenizer,
        block_size=model.sequence_length,
        context_length=model.context_length,
        num_samples=num_samples,
        **extra_kwargs,
    )
    return DataLoader(
        dataset, shuffle=False, batch_size=1, collate_fn=dataset.collate_fn
    )


def evaluate(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    num_samples: int,
    dataset_cls: type[BaseDataset],
    kwargs: Mapping[str, Any],
    prompt_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    skip_fp_model_eval: bool = False,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
) -> tuple[float, str]:
    checkpoint_type = CheckpointType.from_checkpoint(kwargs["checkpoint"])
    if checkpoint_type == CheckpointType.INVALID:
        raise ValueError(
            f"Checkpoint '{kwargs['checkpoint']}' is not recognized "
            f"as a valid quantized checkpoint."
        )

    host_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if host_device.type == "cpu":
        print()
        print(
            "WARNING: Evaluation of this model (floating point or QuantSim) takes a long time on CPU. Doing it on a CUDA-enabled machine will be faster."
        )

    is_default = str(kwargs["checkpoint"]).startswith("DEFAULT")

    fp_kwargs: dict[str, Any] = {}
    if not issubclass(fp_model_cls, LLMDynamicBase):
        fp_kwargs["sequence_length"] = prompt_sequence_length
        fp_kwargs["context_length"] = context_length
    fp_model = fp_model_cls.from_pretrained(**fp_kwargs).to(torch.device("cpu"))

    model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN]

    final_kwargs: dict[str, Any] = dict(kwargs)
    if checkpoint_type == CheckpointType.GENIE_BUNDLE:
        model_cls = qnn_model_cls
        is_fp = False
    elif checkpoint_type.is_aimet_onnx():
        if is_default:
            final_kwargs["fp_model"] = fp_model
        final_kwargs["host_device"] = host_device
        model_cls = quantized_model_cls
        is_fp = False
    else:
        final_kwargs.pop("_skip_quantsim_creation", None)
        model_cls = fp_model_cls
        is_fp = True

    if final_kwargs["checkpoint"] in {"DEFAULT", "DEFAULT_UNQUANTIZED"}:
        del final_kwargs["checkpoint"]

    # For VLM models, load processor so multimodal datasets include images.
    # Also determine the VEG's expected image size for resizing.
    vlm_processor = None
    if vision_encoder_cls is not None and hf_repo_name is not None:
        from transformers import AutoProcessor

        vlm_processor = AutoProcessor.from_pretrained(
            hf_repo_name, trust_remote_code=True
        )
        if vlm_image_size is None:
            raise ValueError(
                "vlm_image_size must be provided when vision_encoder_cls is set."
            )

    eval_dataloader = get_dataset(
        fp_model,
        dataset_cls,
        num_samples,
        processor=vlm_processor,
        image_size=vlm_image_size,
    )
    evaluator = fp_model.get_evaluator(
        dataset_cls.dataset_name(),
        torch.device("cpu") if not is_fp else host_device,
    )

    embedding = None
    if skip_fp_model_eval and evaluator.is_distance_metric and not is_fp:
        # If it's a distance metric, we run the FP model and attach the outputs
        # to the ground truth of the eval data loader.
        assert fp_model_cls.EmbeddingClass is not None
        embedding = fp_model_cls.EmbeddingClass(
            max_length=context_length,
            config=fp_model.llm_config,
        )

        fp_generator = LLM_Generator(
            [fp_model.to(host_device)],
            fp_model.tokenizer,
            embedding,
            accumulate_logits_on_cpu=True,
        )

        fp_logits_list = []
        for input_ids, attention_mask, *_ in eval_dataloader:
            input_ids = input_ids.to(host_device)
            attention_mask = attention_mask.to(host_device)
            fp_logits = fp_generator(input_ids, attention_mask, DynamicCache()).logits
            fp_logits_list.append(fp_logits.detach().cpu())

        # Augment dataloader
        dataset = AugmentedLabelDataset(eval_dataloader.dataset, fp_logits_list)
        eval_dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=eval_dataloader.batch_size,
            collate_fn=eval_dataloader.collate_fn,
        )

    if not is_fp:
        fp_model.to(torch.device("cpu"))
        if "fp_model" not in final_kwargs:
            del fp_model
            gc.collect()
        torch.cuda.empty_cache()

        if issubclass(model_cls, LLMDynamic_AIMETOnnx):
            final_kwargs.pop("sequence_length", None)
            final_kwargs.pop("context_length", None)
        model = model_cls.from_pretrained(**final_kwargs).to(host_device)
    else:
        model = fp_model.to(host_device)

    model.eval()

    if eval_dataloader is None:
        eval_dataloader = get_dataset(model, dataset_cls, num_samples)

    if embedding is None:
        assert fp_model_cls.EmbeddingClass is not None
        embedding = fp_model_cls.EmbeddingClass(
            max_length=context_length,
            config=model.llm_config,
        )

    # Load vision encoder for VLM evaluation
    vision_encoder = None
    if vision_encoder_cls is not None:
        from qai_hub_models import Precision

        veg_kwargs: dict[str, Any] = {}
        if vlm_image_size is not None:
            veg_kwargs["image_width"] = vlm_image_size[0]
            veg_kwargs["image_height"] = vlm_image_size[1]

        if is_fp:
            veg_kwargs["precision"] = Precision.float
        else:
            checkpoint_path = kwargs.get("checkpoint")
            if checkpoint_path is not None:
                veg_kwargs["checkpoint"] = checkpoint_path

        print("Loading vision encoder for evaluation...")
        vision_encoder = vision_encoder_cls.from_pretrained(
            device=host_device,
            **veg_kwargs,
        )

    generator = LLM_Generator(
        [model],
        model.tokenizer,
        embedding,
        accumulate_logits_on_cpu=True,
        vision_encoder=vision_encoder,
        hf_repo_name=hf_repo_name,
    )

    evaluator.add_from_dataset(
        model=generator,
        data=eval_dataloader,
        eval_iterations=len(eval_dataloader),
    )
    model.to("cpu")
    del model
    return evaluator.get_accuracy_score(), evaluator.formatted_accuracy()


def llm_evaluate(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    supported_precisions: list[Precision],
    default_calibration_seqlen: int = 2048,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
) -> None:
    parser = get_model_cli_parser(
        quantized_model_cls,
        suppress_help_arguments=["--host-device", "--fp-model", "--precision"],
    )
    parser = add_input_spec_args(quantized_model_cls, parser)
    parser.add_argument(
        "--task",
        type=str,
        default="wikitext",
        choices=[d.dataset_name() for d in fp_model_cls.get_eval_dataset_classes()],
        help="Tasks for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to be used for evaluation.",
    )
    if vision_encoder_cls is not None:
        parser.add_argument(
            "--image-height",
            type=int,
            default=None,
            help="VEG image height (must be divisible by patch_size * spatial_merge_size).",
        )
        parser.add_argument(
            "--image-width",
            type=int,
            default=None,
            help="VEG image width (must be divisible by patch_size * spatial_merge_size).",
        )

    parser.set_defaults(sequence_length=default_calibration_seqlen)
    args = parser.parse_args()

    kwargs = dict(get_model_kwargs(quantized_model_cls, vars(args)))

    checkpoint_type = CheckpointType.from_checkpoint(kwargs["checkpoint"])
    if checkpoint_type == CheckpointType.GENIE_BUNDLE:
        # The NPU does not support the higher sequence length we use on GPU
        args.sequence_length = min(args.sequence_length, DEFAULT_SEQUENCE_LENGTH)

    # Collect VLM image size: CLI args override the caller-provided default
    if vision_encoder_cls is not None:
        img_h = getattr(args, "image_height", None)
        img_w = getattr(args, "image_width", None)
        if img_h is not None and img_w is not None:
            vlm_image_size = (img_w, img_h)

    eval_dataset_classes = fp_model_cls.get_eval_dataset_classes()
    task_dataset_cls = next(
        ds for ds in eval_dataset_classes if ds.dataset_name() == args.task
    )

    _, formatted_accuracy = evaluate(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        num_samples=args.num_samples,
        dataset_cls=task_dataset_cls,
        kwargs=kwargs,
        vision_encoder_cls=vision_encoder_cls,
        hf_repo_name=hf_repo_name,
        vlm_image_size=vlm_image_size,
        prompt_sequence_length=args.sequence_length,
        context_length=args.context_length,
    )

    print(formatted_accuracy)
