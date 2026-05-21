# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata, OutputSpec
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.common import (
    Precision,
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.protocols import (
    EvaluatableModelProtocol,
    FromPrecompiledProtocol,
    FromPretrainedProtocol,
    QuantizableModelProtocol,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export_result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    InputSpec,
    broadcast_data_to_multi_batch,
    get_batch_size,
    make_torch_inputs,
)
from qai_hub_models.utils.kwarg_helpers import (
    filter_kwargs,
    filter_per_component_kwargs,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT
from qai_hub_models.utils.qai_hub_helpers import (
    build_compile_options,
    build_link_options,
    build_profile_options,
    build_quantize_options,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

__all__ = [
    "BaseModel",
    "BasePrecompiledModel",
    "CollectionModel",
    "IndependentComponentFromPretrainedMixin",
    "PrecompiledCollectionModel",
    "PretrainedCollectionModel",
    "WorkbenchModel",
]


class WorkbenchModel:
    """Base interface for AI Hub Workbench models."""

    # -- Subclasses must implement these --

    def get_input_spec(self, *args: Any, **kwargs: Any) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        raise NotImplementedError

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str | None:
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

    def get_output_names(self) -> list[str]:
        """
        List of output names. If there are multiple outputs, the order of the names
        should match the order of tuple returned by the model.
        """
        outputs = self.get_output_spec().keys()
        assert outputs, "get_output_spec() is not defined!"
        return list(outputs)

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
            self.get_output_names(),
            self.get_channel_last_inputs(),
            self.get_channel_last_outputs(),
            context_graph_name,
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
        sample_inputs = self._sample_inputs_impl(input_spec, **kwargs)
        if input_spec is not None:
            batch_size = get_batch_size(input_spec)
            if batch_size is not None and batch_size > 1:
                sample_inputs = broadcast_data_to_multi_batch(input_spec, sample_inputs)
        if use_channel_last_format and self.get_channel_last_inputs():
            return transpose_channel_first_to_last(
                self.get_channel_last_inputs(), sample_inputs
            )
        return sample_inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """
        Default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs_dict = {}
        inputs_list = make_torch_inputs(input_spec)
        for i, input_name in enumerate(input_spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        return inputs_dict


class BaseModel(
    WorkbenchModel,
    QuantizableModelProtocol,
    EvaluatableModelProtocol,
    FromPretrainedProtocol,
    torch.nn.Module,
):
    """A pre-trained PyTorch model with helpers for submission to AI Hub Workbench."""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        torch.nn.Module.__init__(self)
        self.eval()
        self.model = cast(torch.nn.Module, model)

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
    # get_output_names (inherited from WorkbenchModel)

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

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format preferred for conversion on AI Hub Workbench."""
        return SourceModelFormat.TORCHSCRIPT

    @classmethod
    def get_labels_file_name(cls) -> str | None:
        """
        Returns the name of the labels file for this model.

        The labels file should exist in qai_hub_models/labels/ directory.
        """
        return None

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
        if labels_file_name := self.get_labels_file_name():
            out_path = Path(output_dir) / "labels.txt"
            labels_path = QAIHM_PACKAGE_ROOT / "labels" / labels_file_name
            shutil.copyfile(labels_path, out_path)
            metadata.supplementary_files["labels.txt"] = (
                "Mapping of model prediction indices -> string labels."
            )

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        # Torchscript should never be trained, so disable gradients for all parameters.
        # Need to do this on a model copy, in case the original model is being trained.
        model_copy = deepcopy(self)
        for param in model_copy.parameters():
            param.requires_grad = False
        return torch.jit.trace(
            model_copy, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str | None:
        """Convert to an AI Hub Workbench source model appropriate for the export method."""
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        return prepare_compile_zoo_model_to_hub(
            self,
            source_model_format=SourceModelFormat.TORCHSCRIPT,
            target_runtime=target_runtime,
            output_path=output_path,
            input_spec=input_spec,
            check_trace=check_trace,
            external_onnx_weights=external_onnx_weights,
            output_names=output_names or self.get_output_names(),
        )


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

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str | None:
        return self.target_model_path


ComponentT = TypeVar("ComponentT")
CollectionModelT = TypeVar("CollectionModelT", bound="CollectionModel[Any]")


class CollectionModel(Generic[ComponentT]):
    """
    Model that glues together several BaseModels or BasePrecompiledModel.

    Generic over the component type: CollectionModel[BaseModel] for pretrained,
    CollectionModel[BasePrecompiledModel] for precompiled.

    See test_base_model.py for usage examples.
    """

    COMPONENT_BASE_TYPES: ClassVar[type | tuple[type, ...]] = (
        BaseModel,
        BasePrecompiledModel,
    )
    component_classes: dict[str, type[ComponentT]] = {}
    component_class_names: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make copies for each subclass so they don't share state
        cls.component_classes = dict(cls.component_classes)
        cls.component_class_names = list(cls.component_class_names)

    def __init__(self, *args: ComponentT) -> None:
        component_names = type(self).component_class_names
        self.components: dict[str, ComponentT] = {}

        # Process positional arguments.
        if len(args) != len(component_names):
            raise ValueError(
                f"CollectionModel has {len(component_names)} ordered arguments, "
                "each of which should correspond with a single component."
            )
        for name, arg in zip(component_names, args, strict=False):
            expected_class = type(self).component_classes[name]
            if not isinstance(arg, expected_class):
                raise TypeError(
                    f"Expected component '{name}' to be an instance "
                    f"of {expected_class.__name__}, got {type(arg).__name__}"
                )
            self.components[name] = arg

    @classmethod
    def add_component(
        cls,
        component_class: type[ComponentT],
        component_name: str,
        cli_args_prefix: str | None = None,
        subfolder_hf: str | None = None,
    ) -> Callable[[type[CollectionModelT]], type[CollectionModelT]]:
        """
        Decorator to add a component (a subclass of BaseModel or
        BasePrecompiledModel) to a CollectionModel.  The component is
        inserted at the beginning of the class-level dictionary `component_classes`,
        so that the outer decorator appears first.

        See test_base_model.py for usage examples.

        Parameters
        ----------
        component_class
            Component class to add to the CollectionModel.
        component_name
            Name the component.
        cli_args_prefix
            Name of the CLI argument prefix, with underscores (uses component name if None).
        subfolder_hf
            By default the same as component_name. Specify this
            only when Huggingface uses a different subfolder name than the desired
            component_name. For example, in ControlNet the ControlNet model is not in
            any subfolder on HF, so subfolder_hf = "" even though we want
            to name our component "controlnet".

        Returns
        -------
        callable : Callable[[type[CollectionModelT]], type[CollectionModelT]]
            Decorator function that registers the component on the CollectionModel subclass.
        """

        def decorator(subclass: type[CollectionModelT]) -> type[CollectionModelT]:
            name = component_name
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name), (
                f"component_name must be lowercase snake_case, got: {name!r}"
            )
            assert issubclass(component_class, cls.COMPONENT_BASE_TYPES), (
                f"component_class must be a subclass of BaseModel or "
                f"BasePrecompiledModel, got: {component_class!r}"
            )
            if name in subclass.component_classes:
                raise ValueError(f"Component with name {name} already registered")
            # prepend — outer decorators should appear first
            subclass.component_classes = {
                name: component_class,
                **subclass.component_classes,
            }
            subclass.component_class_names.insert(0, name)

            subfolder = subfolder_hf if subfolder_hf is not None else name
            component_class.default_subfolder = component_name  # type: ignore[attr-defined]
            component_class.cli_args_prefix = (  # type: ignore[attr-defined]
                cli_args_prefix if cli_args_prefix is not None else component_name
            )
            component_class.default_subfolder_hf = subfolder  # type: ignore[attr-defined]
            return subclass

        return decorator

    @staticmethod
    def reset_components() -> Callable[[type[CollectionModel]], type[CollectionModel]]:
        """
        Decorator to erase all components set on a CollectionModel.
        Useful when subclassing a CollectionModel and overriding component classes.

        See test_base_model.py for usage examples.
        """

        def decorator(subclass: type[CollectionModel]) -> type[CollectionModel]:
            subclass.component_classes = {}
            subclass.component_class_names = []
            return subclass

        return decorator

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

    def _filter_kwargs_for_each_component(
        self,
        func_name: str,
        per_component_kwargs: ComponentGroup[dict[str, Any]] | None,
        global_kwargs: dict[str, Any],
    ) -> ComponentGroup[dict[str, Any]]:
        funcs = ComponentGroup(
            {
                name: getattr(component, func_name)
                for name, component in self.components.items()
            }
        )
        return filter_per_component_kwargs(funcs, per_component_kwargs, global_kwargs)


class PretrainedCollectionModel(CollectionModel[BaseModel], FromPretrainedProtocol):
    COMPONENT_BASE_TYPES = BaseModel

    # -- Subclasses may override these --

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
        """Dataset class used for calibration when quantizing collection components."""
        return None

    # -- Per-component getters (delegate to the component) --

    def get_component_input_spec(self, component_name: str, **kwargs: Any) -> InputSpec:
        return self.components[component_name].get_input_spec(**kwargs)

    def get_component_unsupported_reason(
        self, component_name: str, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        return self.components[component_name].get_unsupported_reason(
            target_runtime, device
        )

    def get_component_hub_quantize_options(
        self, component_name: str, precision: Precision, other_options: str = ""
    ) -> str:
        return self.components[component_name].get_hub_quantize_options(
            precision, other_options
        )

    def get_component_hub_compile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return self.components[component_name].get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name=component_name,
        )

    def get_component_hub_link_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        return self.components[component_name].get_hub_link_options(
            target_runtime, other_link_options
        )

    def get_component_hub_profile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        return self.components[component_name].get_hub_profile_options(
            target_runtime, other_profile_options
        )

    def get_component_sample_inputs(
        self,
        component_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        return self.components[component_name].sample_inputs(
            input_spec, use_channel_last_format
        )

    def get_component_precisions(
        self,
        precision: Precision,
    ) -> dict[str, Precision]:
        if precision not in [Precision.mixed, Precision.mixed_with_float]:
            return dict.fromkeys(self.component_class_names, precision)

        return {
            name: component.component_precision()
            for name, component in self.components.items()
        }

    # -- All-component getters --

    def get_input_spec(
        self,
        per_component_kwargs: ComponentGroup[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ComponentGroup[InputSpec]:
        per_component_kwargs = self._filter_kwargs_for_each_component(
            "get_input_spec", per_component_kwargs, kwargs
        )
        return ComponentGroup(
            {
                name: self.get_component_input_spec(
                    name, **per_component_kwargs.get(name, {})
                )
                for name in self.component_class_names
            }
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        for name in self.component_class_names:
            if reason := self.get_component_unsupported_reason(
                name, target_runtime, device
            ):
                return f"Component {name}: {reason}"
        return None

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_options: str = "",
        per_component_quantize_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_quantize_options = (
            per_component_quantize_options or ComponentGroup()
        )
        return ComponentGroup(
            {
                name: self.get_component_hub_quantize_options(
                    name,
                    precision,
                    other_options + f" {per_component_quantize_options.get(name, '')}",
                )
                for name in self.component_class_names
            }
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        per_component_compile_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_compile_options = (
            per_component_compile_options or ComponentGroup()
        )
        return ComponentGroup(
            {
                name: self.get_component_hub_compile_options(
                    name,
                    target_runtime,
                    precision,
                    other_compile_options
                    + f" {per_component_compile_options.get(name, '')}",
                    device,
                )
                for name in self.component_class_names
            }
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
        per_component_link_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_link_options = per_component_link_options or ComponentGroup()
        return ComponentGroup(
            {
                name: self.get_component_hub_link_options(
                    name,
                    target_runtime,
                    other_link_options + f" {per_component_link_options.get(name, '')}",
                )
                for name in self.component_class_names
            }
        )

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        per_component_profile_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_profile_options = (
            per_component_profile_options or ComponentGroup()
        )
        return ComponentGroup(
            {
                name: self.get_component_hub_profile_options(
                    name,
                    target_runtime,
                    other_profile_options
                    + f" {per_component_profile_options.get(name, '')}",
                )
                for name in self.component_class_names
            }
        )

    def sample_inputs(
        self,
        input_specs: ComponentGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
    ) -> ComponentGroup[SampleInputsType]:
        specs = input_specs or self.get_input_spec()
        return ComponentGroup(
            {
                name: self.get_component_sample_inputs(
                    name, specs.get(name), use_channel_last_format
                )
                for name in self.component_class_names
            }
        )


class PrecompiledCollectionModel(
    CollectionModel[BasePrecompiledModel], FromPrecompiledProtocol
):
    COMPONENT_BASE_TYPES = BasePrecompiledModel

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    @classmethod
    def from_precompiled(cls, **kwargs: Any) -> Self:
        """
        Instantiate the CollectionModel by instantiating all registered components
        using their own from_precompiled() methods.
        """
        components = []
        for component_cls in cls.component_classes.values():
            if not (
                hasattr(component_cls, "from_precompiled")
                and callable(component_cls.from_precompiled)
            ):
                raise AttributeError(
                    f"Component '{component_cls.__name__}' does not have "
                    "a callable from_precompiled method"
                )
            components.append(component_cls.from_precompiled())
        return cls(*components)

    def get_component_hub_profile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        return self.components[component_name].get_hub_profile_options(
            target_runtime, other_profile_options
        )

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        per_component_profile_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_profile_options = (
            per_component_profile_options or ComponentGroup()
        )
        return ComponentGroup(
            {
                name: self.get_component_hub_profile_options(
                    name,
                    target_runtime,
                    other_profile_options
                    + f" {per_component_profile_options.get(name, '')}",
                )
                for name in self.component_class_names
            }
        )


class IndependentComponentFromPretrainedMixin:
    """Mixin that builds a collection by calling each component's from_pretrained independently."""

    component_classes: dict[str, type[BaseModel]]

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        **kwargs: Any,
    ) -> Self:
        base_kwargs: dict[str, Any] = {
            "checkpoint": checkpoint,
            "host_device": host_device,
            **kwargs,
        }
        components = []
        for component_cls in cls.component_classes.values():
            supported = filter_kwargs(component_cls.from_pretrained, base_kwargs)
            components.append(component_cls.from_pretrained(**supported))
        return cls(*components)
