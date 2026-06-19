# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from qai_hub.client import Device

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata, OutputSpec
from qai_hub_models.protocols import (
    EvaluatableModelProtocol,
    FromPrecompiledProtocol,
    FromPretrainedProtocol,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    InputSpec,
    make_torch_inputs,
)
from qai_hub_models.utils.kwarg_helpers import cli_friendly_class_name
from qai_hub_models.utils.qai_hub_helpers import (
    build_compile_options,
    build_link_options,
    build_profile_options,
    build_quantize_options,
    expand_to_batch_size,
    make_sample_inputs,
)
from qai_hub_models.utils.transpose_channel import (
    transpose_channel_first_to_last,
)

__all__ = [
    "BaseModel",
    "BasePrecompiledModel",
    "SerializationSettings",
    "WorkbenchModel",
]


class SerializationSettings(NamedTuple):
    use_pt2: bool = True
    check_trace: bool = True


def _model_cls_name(cls_instance: Any) -> str:
    """Model name."""
    # Return the cls_instance ID. Match exactly: qai_hub_models.models.<model_id>.<module>
    parts = type(cls_instance).__module__.split(".")
    if len(parts) == 4 and parts[:2] == ["qai_hub_models", "models"]:
        return parts[2]
    # Class defined outside qai_hub_models.models
    return cli_friendly_class_name(type(cls_instance).__name__)


class WorkbenchModel:
    """Base interface for AI Hub Workbench models."""

    # -- Subclasses must implement these --
    def get_input_spec(self, *args: Any, **kwargs: Any) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        raise NotImplementedError

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Convert to an AI Hub Workbench source model appropriate for the export method."""
        raise NotImplementedError

    def get_output_spec(self) -> OutputSpec:
        """
        Returns a map from `{output_name -> TensorSpec}` with semantic metadata
        for each output tensor (e.g. io_type, bbox format, description).

        Override in subclasses to provide output metadata for the model.
        """
        return {}

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """
        Write supplementary files required by the model during inference.
        These files will be packaged alongside the model when deployed.

        Parameters
        ----------
        output_dir
            Directory where the supplementary files should be written.
        metadata
            The metadata for the compiled models.
            metadata.supplementary_files will be populated with the files written.
        """
        return

    # -- Subclasses may override these --
    @property
    def name(self) -> str:
        """Model name / identifier."""
        return _model_cls_name(self)

    @property
    def context_graph_name(self) -> str:
        """The default name used for the graph context when compiling to a QNN Context Binary. May be overriden in the parameters of get_compile_options."""
        return self.name

    def get_channel_last_inputs(self) -> list[str]:
        """
        A list of input names that should be transposed to channel-last format
        for the on-device model in order to improve performance.
        """
        return []

    def get_channel_last_outputs(self) -> list[str]:
        """
        A list of output names that should be transposed to channel-last format
        for the on-device model in order to improve performance.
        """
        return []

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        """Report the reason if any combination of runtime and device isn't supported."""
        return None

    def get_hub_litemp_percentage(self, precision: Precision) -> float | None:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.

        This method should be implemented for models that support mixed precision quantization.
        """
        return None

    def component_precision(self) -> Precision:
        """
        If this is a component in a collection model, the parent model may declare
        a "variable" precision, where different components use different precisions.

        Returns
        -------
        Precision
            The precision to which this model should be quantized when the parent
            collection model uses "variable" precision.
        """
        raise NotImplementedError()

    # -- Less likely, but subclasses may override these --

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str = ""
    ) -> str:
        """AI Hub Workbench quantize options recommended for the model."""
        litemp_percentage = (
            self.get_hub_litemp_percentage(precision)
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(precision, litemp_percentage, other_options)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench compile options recommended for the model."""
        return build_compile_options(
            target_runtime,
            precision,
            list(self.get_output_spec()),
            self.get_channel_last_inputs(),
            self.get_channel_last_outputs(),
            context_graph_name or self.context_graph_name,
            other_compile_options,
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        """AI Hub Workbench link options recommended for the model."""
        return build_link_options(target_runtime, other_link_options)

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench profile options recommended for the model."""
        return build_profile_options(
            target_runtime, context_graph_name, other_profile_options
        )

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> SampleInputsType:
        """
        Returns a set of sample inputs for the model.

        For each input name in the model, a list of numpy arrays is provided.
        If the returned set is batch N, all input names must contain exactly N numpy arrays.

        Subclasses should NOT override this. They should instead override _sample_inputs_impl.

        This function will invoke _sample_inputs_impl and then apply any required channel
        format transposes.
        """
        input_spec = input_spec or self.get_input_spec()
        inputs = self._sample_inputs_impl(input_spec, **kwargs)
        inputs = expand_to_batch_size(inputs, input_spec)
        if use_channel_last_format and self.get_channel_last_inputs():
            return transpose_channel_first_to_last(
                self.get_channel_last_inputs(), inputs
            )
        return inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        Default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        return make_sample_inputs(input_spec)


class BaseModel(
    WorkbenchModel,
    EvaluatableModelProtocol,
    FromPretrainedProtocol,
    torch.nn.Module,
):
    """A pre-trained PyTorch model with helpers for submission to AI Hub Workbench."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        serialization_settings: SerializationSettings | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.eval()
        self.model = cast(torch.nn.Module, model)
        self.serialization_settings = serialization_settings or SerializationSettings()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
        If this model is being trained, calling `model.train()` will reverse all of these.
        """
        if isinstance(value, torch.nn.Module) and not self.training:
            value.eval()
        torch.nn.Module.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        If a model is in eval mode (which equates to self.training == False),
        we don't want to compute gradients when doing the forward pass.
        """
        context_fn = nullcontext if self.training else torch.no_grad
        with context_fn():
            return torch.nn.Module.__call__(self, *args, **kwargs)

    # -- Subclasses must implement these --

    # get_input_spec (inherited from WorkbenchModel)
    # get_output_spec (inherited from WorkbenchModel)

    # -- Subclasses may override these --

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
        """Dataset class used for calibration when quantizing the model."""
        return None

    def get_evaluator(self) -> BaseEvaluator:
        """Gets a class for evaluating output of this model."""
        raise NotImplementedError("No evaluator is supported for this model.")

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """Converts the torch module to a torchscript trace."""
        input_spec = input_spec or self.get_input_spec()
        self.eval()
        return torch.jit.trace(
            self, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize this model to disk. The serialized model will be uploaded to AI Hub Workbench during export."""
        if self.serialization_settings.use_pt2:
            if torch.torch_version.TorchVersion(torch.__version__) < "2.9":
                raise RuntimeError(
                    f"PT2 serialization (use_pt2=True) requires torch>=2.9, but found {torch.__version__}. "
                    "Upgrade torch or set SerializationSettings(use_pt2=False)."
                )
            input_spec = input_spec or self.get_input_spec()
            output_path = Path(output_dir) / f"{self.name}.pt2"
            self.to("cpu").eval()
            with torch.no_grad():
                exported = torch.export.export(
                    self, tuple(make_torch_inputs(input_spec))
                )
            torch.export.save(exported, output_path)
        else:
            output_path = Path(output_dir) / f"{self.name}.pt"
            input_spec = input_spec or self.get_input_spec()
            torch.jit.save(
                self.convert_to_torchscript(
                    input_spec, check_trace=self.serialization_settings.check_trace
                ),
                output_path,
            )
        return output_path


class BasePrecompiledModel(WorkbenchModel, FromPrecompiledProtocol):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str) -> None:
        self.target_model_path = target_model_path

    # -- Subclasses may override these --

    def get_target_model_path(self) -> str:
        return self.target_model_path

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(self.target_model_path)
