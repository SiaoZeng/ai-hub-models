# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models._shared.llm.quantize import quantize, save_command_args
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    MODEL_ID,
    SAMPLE_IMAGE,
    SUPPORTED_PRECISIONS,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)
from qai_hub_models.utils.args import get_quantize_action_with_default


def quantize_vision_encoder(
    output_dir: str,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    num_calibration_samples: int = 100,
) -> None:
    """Quantize the VEG (Vision Embedding Generator).

    Produces vision_encoder.{onnx,data,encodings} in *output_dir*.
    """
    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = Qwen2_5_VL_7B_VisionEncoder

    print(f"  Loading {num_calibration_samples} calibration images...")
    calibration_data = cls.get_calibration_data(
        num_calibration_samples, image_height, image_width
    )

    print("  Loading VEG from pretrained...")
    veg_model = Qwen2_5_VL_7B_VisionEncoder.from_pretrained(
        device=host_device,
        image_height=image_height,
        image_width=image_width,
    )
    veg_model.eval()

    print("  Exporting VEG to ONNX and creating QuantSim...")
    quant_sim, fixed_inputs = cls.create_quantsim(veg_model, host_device)

    print(f"  Calibrating with {num_calibration_samples} images...")
    cls.calibrate(quant_sim, calibration_data, fixed_inputs)

    print(f"  Saving VEG to: {output_dir}")
    cls.save_quantized_checkpoint(quant_sim, output_dir)

    del veg_model
    del quant_sim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Pass 2 (VEG) completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2.5-VL-7B model",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )

    parser.add_argument(
        "--calibration-sequence-length",
        type=int,
        default=DEFAULT_CALIBRATION_SEQ_LEN,
        help="Sequence length to be used during calibration.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to export the ONNX model and encodings.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Input directory with custom weights.",
    )

    parser.add_argument(
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Add to apply Sequential MSE.",
    )

    parser.add_argument(
        "--use-ada-scale",
        action="store_true",
        default=False,
        help="Add to apply AdaScale.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to be used for calibration.",
    )

    parser.add_argument(
        "--seq-mse-num-samples",
        type=int,
        default=None,
        help="Number of samples for sequential MSE.",
    )

    parser.add_argument(
        "--ada-scale-num-samples",
        type=int,
        default=None,
        help="Number of samples for AdaScale.",
    )

    parser.add_argument(
        "--ada-scale-num-iterations",
        type=int,
        default=None,
        help="Number of iterations for AdaScale.",
    )

    parser.add_argument(
        "--precision",
        default=Precision.parse(SUPPORTED_PRECISIONS[0]),
        action=get_quantize_action_with_default(SUPPORTED_PRECISIONS[0]),
        choices=[str(p) for p in SUPPORTED_PRECISIONS],
        help="Pick the precision with which the model must be quantized.",
    )

    parser.add_argument(
        "--skip-veg",
        action="store_true",
        default=False,
        help="Skip vision encoder (VEG) quantization (Pass 2).",
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip LLM text model quantization (Pass 1).",
    )

    parser.add_argument(
        "--veg-num-samples",
        type=int,
        default=100,
        help="Number of calibration samples for VEG quantization.",
    )

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    # Pass 1: LLM text model
    if not args.skip_llm:
        print("=" * 60)
        print("Pass 1: LLM Text Model Quantization")
        print("=" * 60)
        quantize(
            quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
            fp_model_cls=Qwen2_5_VL_7B_PreSplit,
            context_length=args.context_length,
            seq_len=args.calibration_sequence_length,
            precision=args.precision,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            checkpoint=args.checkpoint,
            use_seq_mse=args.use_seq_mse,
            use_ada_scale=args.use_ada_scale,
            allow_cpu_to_quantize=True,
            seq_mse_num_samples=args.seq_mse_num_samples,
            ada_scale_num_samples=args.ada_scale_num_samples,
            ada_scale_num_iterations=args.ada_scale_num_iterations,
            use_dynamic_shapes=True,
        )
    else:
        print("Skipping Pass 1 (LLM) as requested.")

    # Pass 2: Vision Encoder (VEG)
    if not args.skip_veg:
        print()
        print("=" * 60)
        print("Pass 2: Vision Encoder (VEG) Quantization")
        print("=" * 60)
        quantize_vision_encoder(
            output_dir=args.output_dir,
            num_calibration_samples=args.veg_num_samples,
        )
    else:
        print("Skipping Pass 2 (VEG) as requested.")

    save_command_args(Path(args.output_dir) / "args.json", args, cli_args)

    print()
    print("All quantization passes completed.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy "
        "of the model config to the output checkpoint folder."
    )
    print()
    sample_image = SAMPLE_IMAGE.fetch()
    print("Demo:")
    print(
        f"    python -m qai_hub_models.models.{MODEL_ID}.demo "
        f"--checkpoint {args.output_dir} --image {sample_image} "
        "--prompt 'Describe this image'"
    )
    print()
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{MODEL_ID}.export "
        f"--checkpoint {args.output_dir} --device 'Snapdragon 8 Elite QRD' "
        "--skip-profiling --skip-inferencing --output-dir output"
    )


if __name__ == "__main__":
    main()
