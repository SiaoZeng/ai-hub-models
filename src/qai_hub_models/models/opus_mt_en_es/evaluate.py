# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.opus_mt_en_es import MODEL_ID, App, Model
from qai_hub_models.models.opus_mt_en_es.export import export_model
from qai_hub_models.utils.args import evaluate_parser, get_model_kwargs
from qai_hub_models.utils.base_app import CollectionAppEvaluateProtocol
from qai_hub_models.utils.evaluate import evaluate_on_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec


def main() -> None:
    warnings.filterwarnings("ignore")
    eval_dataset_classes = Model.get_eval_dataset_classes()
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.VOICE_AI,
        ],
    }

    parser = evaluate_parser(
        model_cls=Model,
        supported_dataset_classes=eval_dataset_classes,
        supported_precision_runtimes=supported_precision_runtimes,
        uses_quantize_job=False,
        default_device="Samsung Galaxy S25 (Family)",
    )
    args = parser.parse_args()

    model_kwargs = get_model_kwargs(Model, vars(args))

    if len(eval_dataset_classes) == 0:
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_model(
            device=args.device,
            target_runtime=args.target_runtime,
            skip_downloading=True,
            skip_profiling=True,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
            **model_kwargs,
        )
        return

    assert isinstance(App, CollectionAppEvaluateProtocol), (
        "App must implement CollectionAppEvaluateProtocol, when eval_datasets is specified"
    )

    if args.use_dataset_cache:
        raise ValueError("Collection models do not support use_dataset_cache.")

    collection_model = Model.from_pretrained(**model_kwargs)
    num_components = len(collection_model.component_names)
    input_spec: InputSpec | None = None
    torch_model_list = list(collection_model.components.values())
    model_executors: dict[str, CollectionAppEvaluateProtocol] = {}
    on_device_model_list: list[AsyncOnDeviceModel] = []
    if not args.skip_torch_accuracy:
        model_executors["torch"] = App.from_components(torch_model_list)
        input_spec = torch_model_list[0].get_input_spec()

    if not args.skip_device_accuracy:
        if args.hub_model_id is not None:
            hub_model_id = args.hub_model_id.split(",")
            assert len(hub_model_id) == num_components, (
                f"Number of hub_model_ids ({len(hub_model_id)}) must equal "
                f"number of components ({num_components})"
            )
            compiled_model_list = [hub.get_model(model_id) for model_id in hub_model_id]
        else:
            compiled_model_list = compile_model_from_args(
                MODEL_ID,
                args,
                model_kwargs,
            )
            assert isinstance(compiled_model_list, list)
        for compiled_model in compiled_model_list:
            if compiled_model.get_producer() is None:
                raise ValueError(
                    "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
                )
            on_device_model_list.append(
                AsyncOnDeviceModel(
                    model=compiled_model,
                    input_names=None,
                    device=args.device,
                    inference_options=args.profile_options,
                )
            )
        if not args.skip_device_accuracy:
            model_executors["on-device"] = App.from_components(on_device_model_list)

        input_spec = on_device_model_list[0].get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    evaluate_on_dataset(
        evaluator_func=collection_model.get_evaluator,
        dataset_cls=args.dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        use_cache=args.use_dataset_cache,
    )


if __name__ == "__main__":
    main()
