# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import inspect
from typing import Any

from qai_hub_models.utils.input_spec import InputSpec

from .common import BaseDataset, DatasetSplit

__all__ = ["BaseDataset", "DatasetSplit", "instantiate_dataset"]


def instantiate_dataset(
    dataset_cls: type[BaseDataset],
    split: DatasetSplit,
    input_spec: InputSpec | None = None,
    **kwargs: Any,
) -> BaseDataset:
    if (
        input_spec is not None
        and "input_spec" in inspect.signature(dataset_cls.__init__).parameters
    ):
        kwargs["input_spec"] = input_spec

    # Filter kwargs to only those accepted by the dataset constructor,
    # so callers can pass VLM-specific args (processor, image_size) without
    # breaking datasets that don't accept them.
    init_params = inspect.signature(dataset_cls.__init__).parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
    )
    if not has_var_keyword:
        kwargs = {k: v for k, v in kwargs.items() if k in init_params}

    return dataset_cls(split=split, **kwargs)
