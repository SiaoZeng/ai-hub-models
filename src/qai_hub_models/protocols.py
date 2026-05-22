# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
This file defines type helpers. Specifically, those helpers are python Protocols.

Protocols are helpful for defining interfaces that must be implemented for specific functions.

For example, a function may take any class that implements FromPretrained.
The parameter would be typed "FromPretrainedProtocol", as defined in this file.

Protocols may also be inherited to declare that a class must implement said protocol.

These are type checked at compile time.
"""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from typing_extensions import Self

from qai_hub_models import (
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator, _DataLoader

__all__ = [
    "AIMETQuantizableModelProtocol",
    "EvaluatableModelProtocol",
    "ExecutableModelProtocol",
    "FromPrecompiledProtocol",
    "FromPrecompiledTypeVar",
    "FromPretrainedProtocol",
    "FromPretrainedTypeVar",
    "QuantizableModelProtocol",
]

FromPretrainedTypeVar = TypeVar("FromPretrainedTypeVar", bound="FromPretrainedProtocol")
FromPrecompiledTypeVar = TypeVar(
    "FromPrecompiledTypeVar", bound="FromPrecompiledProtocol"
)


@runtime_checkable
class QuantizableModelProtocol(Protocol):
    """Minimum required methods to export a model that can be quantized."""


@runtime_checkable
class AIMETQuantizableModelProtocol(QuantizableModelProtocol, Protocol):
    """Methods required for a model to be quantizable."""

    def quantize(
        self,
        data: _DataLoader,
        num_samples: int | None = None,
        device: str = "cpu",
        requantize_model_weights: bool = False,
        data_has_gt: bool = False,
    ) -> None:
        """
        Compute quantization encodings for this model with the given dataset and model evaluator.

        This model will be updated with a new set of quantization parameters. Future calls to
        forward() and export_...() will take these quantization parameters into account.

        Parameters
        ----------
        data
            Data loader for the dataset to use for evaluation.
                If an evaluator is __NOT__ provided (see "evaluator" parameter), the iterator must return
                    inputs: Collection[torch.Tensor] | torch.Tensor

                otherwise, if an evaluator __IS__ provided, the iterator must return
                    tuple(
                      inputs: Collection[torch.Tensor] | torch.Tensor,
                      ground_truth: Collection[torch.Tensor] | torch.Tensor]
                    )

        num_samples
            Number of samples to use for evaluation. One sample is one iteration from iter(data).
            If none, defaults to the number of samples in the dataset.

        device
            Name of device on which inference should be run.

        requantize_model_weights
            If a weight is quantized, recompute its quantization parameters.

        data_has_gt
            Set to true if the data loader passed in also provides ground truth data.
            The ground truth data will be discarded for quantization.
        """
        ...

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format preferred for conversion on AI Hub Workbench."""
        ...


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ExecutableModelProtocol(Protocol, Generic[T_co]):
    """Models follow this protocol if they can be quantized using AI Hub Models."""

    def __call__(self, *args: Any, **kwargs: Any) -> T_co:
        """Execute the model and return its output."""
        ...


@runtime_checkable
class FromPretrainedProtocol(Protocol):
    """Models follow this protocol if they can be initiated from a pretrained model."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Utility function that helps users get up and running with a default
        pretrained model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_pretrained()` and always have it return something reasonable.
        """
        ...


class FromPrecompiledProtocol(Protocol):
    """Models follow this protocol if they can be initiated from a precompiled model."""

    @classmethod
    def from_precompiled(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Utility function that helps users get up and running with a default
        precompiled model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_precompiled()` and always have it return something reasonable.
        """
        ...


@runtime_checkable
class EvaluatableModelProtocol(ExecutableModelProtocol, Protocol):
    """Models follow this protocol if they can be evaluated using AI Hub Models."""

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        """Returns list of dataset classes on which this model can be evaluated."""
        ...

    def get_evaluator(self) -> BaseEvaluator:
        """Gets a class for evaluating output of this model."""
        ...
