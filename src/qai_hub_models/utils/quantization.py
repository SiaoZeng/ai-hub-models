# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.base_app import CollectionAppProtocol
from qai_hub_models.utils.base_model import BaseModel, PretrainedCollectionModel
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.input_spec import (
    InputSpec,
    get_batch_size,
    is_input_spec,
    is_input_spec_dict,
)
from qai_hub_models.utils.private_asset_loaders import UnfetchableDatasetError
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


def get_calibration_data(
    model: BaseModel | PretrainedCollectionModel,
    input_spec_arg: InputSpec | dict[str, InputSpec] | None = None,
    num_samples: int | None = None,
    component_name: str | None = None,
    dataset_options: dict | None = None,
    app: Any = None,
) -> DatasetEntries:
    """
    Produces a numpy dataset to be used for calibration data of a quantize job.

    If the model has a calibration dataset name set, it will use that dataset.
    Otherwise, it returns the model's sample inputs.

    Parameters
    ----------
    model
        The model or collection model for which to get calibration data.
    input_spec_arg
        For a single model: an InputSpec or None.
        For a collection model: a dict mapping component names to InputSpecs.
    num_samples
        Number of data samples to use. If not specified, uses
        default specified on dataset.
    component_name
        For collection models, the name of the component being calibrated.
    dataset_options
        Additional options to pass to the dataset constructor.
    app
        The model's app used with collection models to fetch calibration data
        via app.get_calibration_data() if it is instance of CollectionAppProtocol.

    Returns
    -------
    calibration_dataset : DatasetEntries
        Dataset compatible with the format expected by AI Hub Workbench.
    """
    # Resolve input_spec_arg into input_spec (single) and input_spec_dict (per-component)
    input_spec: InputSpec | None = None
    input_spec_dict: dict[str, InputSpec] | None = None
    if is_input_spec_dict(input_spec_arg):
        input_spec_dict = input_spec_arg
    elif is_input_spec(input_spec_arg):
        input_spec = input_spec_arg

    if isinstance(model, PretrainedCollectionModel):
        assert component_name is not None, (
            "component_name is required for collection models"
        )
        if isinstance(app, CollectionAppProtocol):
            return app.get_calibration_data(
                model,
                component_name,
                input_spec_dict,
                num_samples,
            )
        if input_spec_dict:
            input_spec = input_spec_dict.get(component_name)
        model = model.components[component_name]

    assert isinstance(model, BaseModel)
    calibration_dataset_name = model.calibration_dataset_name()
    if calibration_dataset_name is None:
        assert num_samples is None, (
            "Cannot set num_samples if model doesn't have calibration dataset."
        )
        print(
            "WARNING: Model will be quantized using only a single sample for calibration. "
            "The quantized model should be only used for performance evaluation, and is unlikely to "
            "produce reasonable accuracy without additional calibration data."
        )
        return model.sample_inputs(input_spec, use_channel_last_format=False)
    input_spec = input_spec or model.get_input_spec()
    batch_size = get_batch_size(input_spec) or 1
    dataset_options = dataset_options or {}

    try:
        dataset = get_dataset_from_name(
            calibration_dataset_name,
            split=DatasetSplit.TRAIN,
            input_spec=input_spec,
            **dataset_options,
        )
    except UnfetchableDatasetError as e:
        if e.installation_steps is None:
            raise ValueError(
                f"The calibration dataset ({e.dataset_name}) for this model is not publicly available. If you are running `export.py`, run export again and add `--fetch-static-assets`. This will fetch a pre-quantized model file, and skips the step that requires fetching this dataset."
            ) from None
        raise
    num_samples = num_samples or dataset.default_num_calibration_samples()

    # Round num samples to largest multiple of batch_size less than number requested
    num_samples = (num_samples // batch_size) * batch_size
    print(f"Loading {num_samples} calibration samples.")
    torch_dataset = sample_dataset(dataset, num_samples)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size)
    inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(len(input_spec))]
    for sample_input, _ in dataloader:
        if isinstance(sample_input, (tuple, list)):
            for i, tensor in enumerate(sample_input):
                inputs[i].append(tensor)
        else:
            inputs[0].append(sample_input)
    return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
