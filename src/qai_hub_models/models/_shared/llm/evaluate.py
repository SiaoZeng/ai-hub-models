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
from transformers import AutoProcessor

from qai_hub_models import Precision
from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.evaluators.llm_evaluator import LLMEvaluator
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.generator_factory import make_generator
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
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    processor: Any = None,
    image_size: tuple[int, int] | None = None,
) -> DataLoader:
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
        block_size=sequence_length,
        context_length=context_length,
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
    prompt_sequence_length: int | list[int] = DEFAULT_SEQUENCE_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
    task_kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Top-level evaluate that routes to the appropriate implementation.

    Models that define GeneratorClass use the new make_generator path.
    Legacy models fall back to the LLM_Generator path.
    """
    if hasattr(fp_model_cls, "GeneratorClass"):
        return _evaluate_impl(
            quantized_model_cls=quantized_model_cls,
            fp_model_cls=fp_model_cls,
            qnn_model_cls=qnn_model_cls,
            num_samples=num_samples,
            dataset_cls=dataset_cls,
            kwargs=kwargs,
            prompt_sequence_length=prompt_sequence_length,
            context_length=context_length,
            vision_encoder_cls=vision_encoder_cls,
            hf_repo_name=hf_repo_name,
            vlm_image_size=vlm_image_size,
            task_kwargs=task_kwargs,
        )
    return _legacy_evaluate_impl(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        kwargs=kwargs,
        prompt_sequence_length=prompt_sequence_length,
        context_length=context_length,
        vision_encoder_cls=vision_encoder_cls,
        hf_repo_name=hf_repo_name,
        vlm_image_size=vlm_image_size,
        task_kwargs=task_kwargs,
    )


def _evaluate_impl(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    num_samples: int,
    dataset_cls: type[BaseDataset],
    kwargs: Mapping[str, Any],
    prompt_sequence_length: int | list[int] = DEFAULT_SEQUENCE_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
    task_kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Evaluate using make_generator (for models with GeneratorClass)."""
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
            "WARNING: Evaluation of this model (floating point or QuantSim) takes a "
            "long time on CPU. Doing it on a CUDA-enabled machine will be faster."
        )

    is_default = str(kwargs["checkpoint"]).startswith("DEFAULT")

    fp_model = fp_model_cls.from_pretrained().to(torch.device("cpu"))

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
    vlm_processor = None
    if vision_encoder_cls is not None and hf_repo_name is not None:
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
        sequence_length=max(prompt_sequence_length)
        if isinstance(prompt_sequence_length, list)
        else prompt_sequence_length,
        context_length=context_length,
        processor=vlm_processor,
        image_size=vlm_image_size,
    )
    evaluator = fp_model.get_evaluator(
        dataset_cls.dataset_name(),
        torch.device("cpu") if not is_fp else host_device,
        **(task_kwargs or {}),
    )
    assert isinstance(evaluator, LLMEvaluator)

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

    # Load vision encoder for VLM evaluation
    vision_model = None
    if vision_encoder_cls is not None:
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
                # Load the quantized VEG (vision_encoder.onnx/.encodings) from
                # the checkpoint instead of FP weights. Precision is recovered
                # from the checkpoint's args.json; if absent it stays float.
                veg_kwargs["precision"] = checkpoint_type.precision(
                    Precision.float, checkpoint=checkpoint_path
                )

        print("Loading vision encoder for evaluation...")
        vision_model = vision_encoder_cls.from_pretrained(
            device=host_device,
            **veg_kwargs,
        )

    generator = make_generator(
        model,
        sequence_length=prompt_sequence_length,
        context_length=context_length,
        vision_model=vision_model,
        model_cls=fp_model_cls,
        device=torch.device("cpu") if not is_fp else host_device,
    )

    evaluator.add_from_dataset(
        model=generator,
        data=eval_dataloader,
        eval_iterations=len(eval_dataloader),
    )

    model.release()
    del model
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    score = evaluator.get_accuracy_score()
    formatted = evaluator.formatted_accuracy()
    return score, formatted


def _legacy_evaluate_impl(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    num_samples: int,
    dataset_cls: type[BaseDataset],
    kwargs: Mapping[str, Any],
    prompt_sequence_length: int | list[int] = DEFAULT_SEQUENCE_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
    task_kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Evaluate using LLM_Generator (legacy path for models without GeneratorClass)."""
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
        sequence_length=max(prompt_sequence_length)
        if isinstance(prompt_sequence_length, list)
        else prompt_sequence_length,
        context_length=context_length,
        processor=vlm_processor,
        image_size=vlm_image_size,
    )
    evaluator = fp_model.get_evaluator(
        dataset_cls.dataset_name(),
        torch.device("cpu") if not is_fp else host_device,
        **(task_kwargs or {}),
    )
    # Every LLM eval task resolves to an LLMEvaluator, which tells the generator
    # whether to accumulate logits on CPU (forward-only metrics) or keep them on
    # the model device (autoregressive generation). See LLMEvaluator.
    assert isinstance(evaluator, LLMEvaluator)

    embedding = None
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
        accumulate_logits_on_cpu=evaluator.accumulate_logits_on_cpu,
        vision_encoder=vision_encoder,
        hf_repo_name=hf_repo_name,
    )

    evaluator.add_from_dataset(
        model=generator,
        data=eval_dataloader,
        eval_iterations=len(eval_dataloader),
    )
    # Tear down the eval model BEFORE scoring. Every evaluator computes its
    # score from state accumulated during add_from_dataset (no evaluator needs
    # the live model at scoring time), and the response evaluator's score step
    # spawns a separate grader process that loads a large model on the same GPU
    # — leaving the eval model resident here causes the grader to OOM.
    #
    # Tear down explicitly: generator holds refs to `model` via self.models and
    # self.selected_model, so `del model` alone leaves the AIMET ORT session
    # (and its CUDA arena) alive across parametrized test cases.
    model.release()
    del model
    generator.release()
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    score = evaluator.get_accuracy_score()
    formatted = evaluator.formatted_accuracy()
    return score, formatted


def llm_evaluate(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    supported_precisions: list[Precision],
    default_calibration_seqlen: int = 2048,
    vision_encoder_cls: Any = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
    end_tokens: set[str] | None = None,
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

    prompt_group = parser.add_argument_group(
        "prompt-based tasks",
        "Options that only apply to the prompt-generation tasks "
        "('prompts', 'multimodal_prompts'); ignored by all other tasks.",
    )
    prompt_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write generated responses and grader summary. "
            "Required for the 'prompts' and 'multimodal_prompts' tasks."
        ),
    )
    prompt_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate per prompt.",
    )
    prompt_group.add_argument(
        "--grader-venv",
        type=str,
        default=None,
        help=(
            "Path to a venv whose python runs the response grader (separate "
            "transformers version). If omitted, a venv named qaihm-dev-grader "
            "is searched for under both the home dir and the repo root."
        ),
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

    task_kwargs: dict[str, Any] | None = None
    if args.task in {"prompts", "multimodal_prompts"}:
        if args.output_dir is None:
            parser.error(f"--output-dir is required for the '{args.task}' task.")
        task_kwargs = {
            "output_dir": args.output_dir,
            "max_new_tokens": args.max_new_tokens,
            "end_tokens": end_tokens,
            "grader_venv": args.grader_venv,
        }

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
        task_kwargs=task_kwargs,
    )

    print(formatted_accuracy)
