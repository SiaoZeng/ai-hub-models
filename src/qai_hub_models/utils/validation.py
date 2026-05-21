# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import inspect

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphCollectionModel
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


def _is_valid_dataset_class(dataset_cls: type) -> bool:
    return (
        isinstance(dataset_cls, type)
        and issubclass(dataset_cls, BaseDataset)
        and not inspect.isabstract(dataset_cls)
    )


def _quantized_precision_names(code_gen: QAIHMModelCodeGen) -> list[str]:
    return [str(p) for p in code_gen.supported_precisions if p != Precision.float]


def validate_io_names(instance: BaseModel) -> list[str]:
    """
    Validate channel-last declarations match actual I/O names
    and that names don't contain dashes.

    Parameters
    ----------
    instance
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages for each failing check.
    """
    input_spec = instance.get_input_spec()
    output_names = instance.get_output_names()

    errors = [
        f"Channel-last input '{name}' not found in input spec: {list(input_spec.keys())}"
        for name in instance.get_channel_last_inputs()
        if name not in input_spec
    ]
    errors.extend(
        f"Channel-last output '{name}' not found in output names: {output_names}"
        for name in instance.get_channel_last_outputs()
        if name not in output_names
    )
    errors.extend(
        f"Input name '{name}' contains '-'. "
        "QNN converts dashes to underscores, causing name mismatches."
        for name in input_spec
        if "-" in name
    )
    errors.extend(
        f"Output name '{name}' contains '-'. "
        "QNN converts dashes to underscores, causing name mismatches."
        for name in output_names
        if "-" in name
    )
    return errors


def validate_io_names_collection(
    model: CollectionModel | MultiGraphCollectionModel,
) -> list[str]:
    """
    Run I/O name validation on each component of a collection model.

    Parameters
    ----------
    model
        The collection model to validate.

    Returns
    -------
    list[str]
        Error messages for each failing check, prefixed with the component name.
    """
    errors: list[str] = []
    for comp_name, component in model.components.items():
        if not isinstance(component, BaseModel):
            continue
        errors.extend(
            f"[component '{comp_name}'] {err}" for err in validate_io_names(component)
        )
    return errors


def validate_eval_datasets(
    model: BaseModel | PretrainedCollectionModel,
) -> list[str]:
    """
    Validate that all dataset classes returned by get_eval_dataset_classes() are valid.

    Parameters
    ----------
    model
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages for each invalid dataset class.
    """
    return [
        f"get_eval_dataset_classes() includes '{ds_cls.dataset_name()}', which is not "
        "a valid BaseDataset subclass."
        for ds_cls in model.get_eval_dataset_classes()
        if not _is_valid_dataset_class(ds_cls)
    ]


def validate_eval_datasets_have_evaluator(
    model: BaseModel,
) -> list[str]:
    """
    Validate that models with eval datasets implement get_evaluator().

    Parameters
    ----------
    model
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages if get_eval_dataset_classes() is non-empty but
        get_evaluator() is not overridden.
    """
    if not model.get_eval_dataset_classes():
        return []
    if model.get_evaluator is BaseModel.get_evaluator:
        return [
            "get_eval_dataset_classes() is non-empty but get_evaluator() is not implemented."
        ]
    return []


def _litemp_implemented(model: BaseModel, precision: Precision) -> bool:
    try:
        model.get_hub_litemp_percentage(precision)
    except NotImplementedError:
        return False
    return True


def validate_mixed_precision_litemp(
    model: BaseModel,
    code_gen: QAIHMModelCodeGen,
) -> list[str]:
    """
    Validate that models with mixed-precision support implement
    get_hub_litemp_percentage().

    Parameters
    ----------
    model
        The model instance to validate.
    code_gen
        The model's code-gen.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each mixed precision missing litemp support.
    """
    mixed_precisions = [
        p
        for p in code_gen.supported_precisions
        if isinstance(p, Precision) and p.override_type is not None
    ]
    return [
        f"Precision {p} uses mixed precision (override_type) "
        "but get_hub_litemp_percentage() raises NotImplementedError."
        for p in mixed_precisions
        if not _litemp_implemented(model, p)
    ]


def validate_labels_file(model: BaseModel) -> list[str]:
    """
    Validate that the labels file declared by the model exists on disk.

    Parameters
    ----------
    model
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages if the labels file is missing.
    """
    labels_name = model.get_labels_file_name()
    if labels_name is None:
        return []
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / labels_name
    if not labels_path.exists():
        return [
            f"get_labels_file_name() returns '{labels_name}', "
            f"but {labels_path} does not exist."
        ]
    return []


def _component_precision_implemented(component: BaseModel) -> bool:
    try:
        component.component_precision()
    except NotImplementedError:
        return False
    return True


def validate_component_precision(
    model: CollectionModel | MultiGraphCollectionModel,
    code_gen: QAIHMModelCodeGen,
) -> list[str]:
    """
    Validate that components implement component_precision() when the
    collection model declares mixed or mixed_with_float precision,
    and that components whose per-component precision uses mixed precision
    also implement get_hub_litemp_percentage().

    Parameters
    ----------
    model
        The collection model to validate.
    code_gen
        The model's code-gen.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each component missing component_precision()
        or litemp support.
    """
    has_mixed = any(
        p in [Precision.mixed, Precision.mixed_with_float]
        for p in code_gen.supported_precisions
    )
    if not has_mixed:
        return []

    errors: list[str] = []
    for comp_name, component in model.components.items():
        if not isinstance(component, BaseModel):
            continue
        if not _component_precision_implemented(component):
            errors.append(
                f"[component '{comp_name}'] Collection model declares mixed precision "
                "but component does not implement component_precision()."
            )
            continue
        comp_precision = component.component_precision()
        if (
            isinstance(comp_precision, Precision)
            and comp_precision.override_type is not None
            and not _litemp_implemented(component, comp_precision)
        ):
            errors.append(
                f"[component '{comp_name}'] Component precision {comp_precision} "
                "uses mixed precision (override_type) "
                "but get_hub_litemp_percentage() raises NotImplementedError."
            )
    return errors


def perform_runtime_model_validation(
    model_cls: type[BaseModel | PretrainedCollectionModel | MultiGraphCollectionModel],
    model_id: str,
    app_cls: type | None = None,
) -> None:
    """
    Run all static validation checks on a model's configuration.

    Raises AssertionError with all collected failures.

    Parameters
    ----------
    model_cls
        The model class to validate.
    model_id
        The model identifier used to load code-gen.yaml.
    app_cls
        For collection models, the App class so calibration checks
        can verify CollectionAppProtocol compliance. Passing ``None``
        is safe for models without quantized precisions; for models
        with quantized precisions, ``None`` will produce an error
        indicating the missing App.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    code_gen = QAIHMModelCodeGen.from_model(model_id)
    errors: list[str] = []

    model = model_cls.from_pretrained()
    if isinstance(model, MultiGraphCollectionModel):
        errors.extend(validate_io_names_collection(model))
        errors.extend(validate_component_precision(model, code_gen))
        # MultiGraphCollectionModel doesn't support new dataset API yet, skip dataset validation
    elif isinstance(model, PretrainedCollectionModel):
        errors.extend(validate_io_names_collection(model))
        errors.extend(validate_component_precision(model, code_gen))
        errors.extend(validate_eval_datasets(model))
    elif isinstance(model, CollectionModel):
        errors.extend(validate_io_names_collection(model))
        errors.extend(validate_component_precision(model, code_gen))
        # Other CollectionModel types don't support dataset validation yet
    else:
        errors.extend(validate_io_names(model))
        errors.extend(validate_mixed_precision_litemp(model, code_gen))
        errors.extend(validate_labels_file(model))
        errors.extend(validate_eval_datasets_have_evaluator(model))
        errors.extend(validate_eval_datasets(model))

    if errors:
        header = (
            f"Model validation failed for '{model_id}' with {len(errors)} error(s):"
        )
        details = "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(f"{header}\n{details}")
