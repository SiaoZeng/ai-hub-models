# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Utility Functions for parsing input args for export and other customer facing scripts."""

from __future__ import annotations

import argparse
import copy
import inspect
import sys
from collections.abc import Callable, Mapping
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path
from pydoc import locate
from typing import Any, TypeVar

import qai_hub as hub
from numpydoc.docscrape import FunctionDoc

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.protocols import (
    FromPrecompiledProtocol,
    FromPrecompiledTypeVar,
    FromPretrainedProtocol,
    FromPretrainedTypeVar,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import (
    BaseModel,
    BasePrecompiledModel,
    CollectionModel,
    PretrainedCollectionModel,
    WorkbenchModel,
)
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphCollectionModel,
    MultiGraphWorkbenchModel,
)
from qai_hub_models.utils.envvars import DevModeEnvvar
from qai_hub_models.utils.evaluate import EvalMode
from qai_hub_models.utils.inference import OnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import filter_kwargs, get_params
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    raise_if_fp_is_unsupported,
)
from qai_hub_models.utils.version_helpers import QAIHMVersion


class ParseEnumAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        enum_type: type[Enum],
        **kwargs: Any,
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.enum_type = enum_type

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        assert isinstance(values, str)
        setattr(namespace, self.dest, self.enum_type[values.upper().replace("-", "_")])


ParserT = TypeVar("ParserT", bound=argparse.ArgumentParser)


def _get_non_float_precision(
    supported_precisions: set[Precision] | None,
) -> Precision | None:
    if not supported_precisions:
        return None

    for p in supported_precisions:
        if p != Precision.float:
            return p

    return None


def get_quantize_action_with_default(
    default_quantized_precision: Precision,
) -> type[argparse.Action]:
    """
    Get an action that:

    Returns default_quantized_precision if "--quantize" is passed with no arg.

    Returns a parsed precision object if "--quantize <value> " is passed.
    """

    class ParsePrecisionAction(argparse.Action):
        def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
            super().__init__(option_strings, dest, **kwargs)

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Precision | None,
            option_string: str | None = None,
        ) -> None:
            if values:
                if isinstance(values, Precision):
                    val = values
                else:
                    assert isinstance(values, str)
                    val = Precision.parse(values)
            else:
                val = default_quantized_precision

            setattr(namespace, self.dest, val)

    return ParsePrecisionAction


class QAIHMHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """
    Argparse formatter that combnines:
      * allowing raw text (eg. newlines) in help messages
      * including defaults in help messages (except for boolean args)
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """
        Default value for booleans in CLI help can be misleading.
        This overridden function will print just the help message for boolean args
        and print help message along with the default value for all other args.
        """
        # Don't print "(default: <value>)" in the CLI help if the value is a bool
        # or something "non-truthy" (e.g. "", None, [])
        if isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ) or (not action.default):
            return action.help
        return super()._get_help_string(action)


class QAIHMArgumentParser(argparse.ArgumentParser):
    """
    An ArgumentParser that sets device from the appropriate options.
    This isn't implemented as a `type` argument to `add_argument` because the
    device/chipset can be modified by device_os.
    """

    def __init__(
        self,
        model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar] | None = None,
        supported_precision_runtimes: (
            dict[Precision, list[TargetRuntime]] | None
        ) = None,
        default_device: str | None = None,
        default_chipset: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.supported_precision_runtimes = supported_precision_runtimes or {}
        self.default_device = default_device
        self.default_chipset = default_chipset
        self.model_cls = model_cls
        self._dataset_name_to_cls: dict[str, type[BaseDataset]] = {}
        super().__init__(*args, **kwargs)

    def set_supported_dataset_classes(
        self, dataset_classes: list[type[BaseDataset]]
    ) -> None:
        self._dataset_name_to_cls = {
            ds_cls.dataset_name(): ds_cls for ds_cls in dataset_classes
        }

    @staticmethod
    def get_hub_device(
        device: str | None = None, chipset: str | None = None, device_os: str = ""
    ) -> hub.Device | None:
        """
        Get a hub.Device given a device name and/or chipset name.
        If neither is specified, the function returns None.
        """
        if chipset or device:
            return hub.Device(
                name=device or "",
                attributes=f"chipset:{chipset}" if chipset else [],
                os=device_os,
            )
        return None

    def parse_args(
        self, args: list[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace or argparse.Namespace())
        parsed.device = self.get_hub_device(
            getattr(parsed, "device_str", None),
            getattr(parsed, "chipset", None),
            getattr(parsed, "device_os", ""),
        )
        if parsed.device is None:
            parsed.device = self.get_hub_device(
                self.default_device, self.default_chipset
            )

        if getattr(parsed, "quantize", None):
            parsed.precision = parsed.quantize

        # Resolve default target_runtime based on the chosen precision.
        if getattr(parsed, "target_runtime", None) is None:
            precision = getattr(parsed, "precision", None)
            if precision is None and self.supported_precision_runtimes:
                precision = next(iter(self.supported_precision_runtimes))
            if precision is not None and precision in self.supported_precision_runtimes:
                parsed.target_runtime = _get_default_runtime(
                    self.supported_precision_runtimes[precision]
                )
            else:
                # No precision arg -- fall back to first eligible across all precisions.
                all_runtimes = chain.from_iterable(
                    self.supported_precision_runtimes.values()
                )

                # If all else fails, use TFLITE
                parsed.target_runtime = next(all_runtimes, TargetRuntime.TFLITE)

        quantized_model_id_arg = getattr(parsed, "quantized_model_id", None)
        if quantized_model_id_arg:
            if getattr(parsed, "precision", None) == Precision.float:
                raise ValueError(
                    "--quantized-model-id can only be used with a quantized precision. "
                    "Pass --precision <non-float> or --quantize."
                )
            assert self.model_cls is not None
            if not issubclass(self.model_cls, CollectionModel):
                # BaseModel
                parsed.quantized_model_id = quantized_model_id_arg
            else:
                # CollectionModel
                components = getattr(parsed, "components", None)
                components = (
                    components if components else self.model_cls.component_class_names
                )

                model_ids = [
                    s.strip() for s in quantized_model_id_arg.split(",") if s.strip()
                ]
                if len(model_ids) != len(components):
                    raise ValueError(
                        "For collection models, --quantized-model-id must provide exactly one id per selected component. "
                        f"Expected {len(components)} ids, got {len(model_ids)}."
                    )

                parsed.quantized_model_id = dict(
                    zip(components, model_ids, strict=True)
                )

        if self.supported_precision_runtimes:
            self.validate_precision_runtime(self.supported_precision_runtimes, parsed)

        # FP16 device-precision validation
        precision = getattr(parsed, "precision", None)
        fetch_static_assets = getattr(parsed, "fetch_static_assets", None)
        if (
            parsed.device is not None
            and precision is not None
            and fetch_static_assets is None
        ):
            self._validate_fp16_support(parsed.device, precision)

        if self._dataset_name_to_cls and hasattr(parsed, "dataset_name"):
            parsed.dataset_cls = self._dataset_name_to_cls[parsed.dataset_name]

        return parsed

    @staticmethod
    def _validate_fp16_support(device: hub.Device, precision: Precision) -> None:
        """Check FP16 support using YAML first, falling back to workbench API."""
        raise_if_fp_is_unsupported(device, precision)

    @staticmethod
    def validate_precision_runtime(
        supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
        parsed_args: argparse.Namespace,
    ) -> None:
        """
        Verifies that supported_precision_runtimes contains the precision + runtime pair chosen by the parsed argument namespace.
        If the namespace does not include both precision and runtime, then validation is skipped.
        """
        # If fetch_static_assets is set, validation of whether a specific precision / runtime pair is supported
        # is done downstream. This validation is only necessary when running the export script.
        fetch_static_assets: str | None = getattr(
            parsed_args, "fetch_static_assets", None
        )

        # If precision or target_runtime are None, they aren't args used by this parser. This validation becomes a no-op.
        precision: Precision | None = getattr(parsed_args, "precision", None)
        target_runtime: TargetRuntime | None = getattr(
            parsed_args, "target_runtime", None
        )

        if (
            fetch_static_assets is not None
            or precision is None
            or target_runtime is None
            or DevModeEnvvar.get()
        ):
            return

        if (
            precision not in supported_precision_runtimes
            or target_runtime not in supported_precision_runtimes[precision]
        ):
            str_supported_precision_runtimes = "\n".join(
                f"    {p}: {', '.join([rt.value for rt in rts])}"
                for p, rts in supported_precision_runtimes.items()
            )
            print(
                f"Model does not support runtime {target_runtime.value} with precision {precision}. These combinations are supported:\n"
                + str_supported_precision_runtimes
            )
            sys.exit(1)


def get_parser(
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar] | None = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    allow_dupe_args: bool = False,
) -> QAIHMArgumentParser:
    return QAIHMArgumentParser(
        model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        formatter_class=QAIHMHelpFormatter,
        conflict_handler="resolve" if allow_dupe_args else "error",
    )


def _add_device_args(
    parser: QAIHMArgumentParser,
    default_device: str | None = None,
    default_chipset: str | None = None,
) -> QAIHMArgumentParser:
    # This is an assertion because this is a logic error; it shouldn't be possible to get this at runtime.
    assert not (default_device and default_chipset), (
        "Only one of default_device or default_chipset may be specified."
    )

    parser.default_device = default_device
    parser.default_chipset = default_chipset
    device_group = parser.add_argument_group("Device Selection")
    device_mutex_group = device_group.add_mutually_exclusive_group()
    device_mutex_group.add_argument(
        "--device",
        dest="device_str",
        type=str,
        help="The name of the device used to run this script. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_device}`." if default_device else ""),
    )
    device_mutex_group.add_argument(
        "--chipset",
        type=str,
        help="If set, will choose a random device with this chipset. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_chipset}`." if default_chipset else ""),
    )
    device_group.add_argument(
        "--device-os",
        type=str,
        default="",
        help="Optionally specified together with --device or --chipset",
    )

    return parser


def add_output_dir_arg(parser: ParserT) -> ParserT:
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="If specified, saves demo output (e.g. image) to this directory instead of displaying.",
    )
    return parser


def _get_default_runtime(
    available_runtimes: list[TargetRuntime],
) -> TargetRuntime:
    if len(available_runtimes) == 0:
        raise RuntimeError("available_runtimes empty, expecting at-least one runtime.")
    return available_runtimes[0]


def add_target_runtime_arg(
    parser: ParserT,
    helpmsg: str,
    available_target_runtimes: list[TargetRuntime] | None = None,
    default: TargetRuntime | None = None,
) -> ParserT:
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    parser.add_argument(
        "--target-runtime",
        type=str,
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore[arg-type]
        default=default,
        metavar=f"{{{', '.join(rt.value for rt in available_target_runtimes)}}}",
        help=helpmsg,
    )
    return parser


def add_precision_arg(
    parser: argparse.ArgumentParser,
    supported_precisions: set[Precision],
    default_if_arg_explicitly_passed: Precision,  # the default value if --precision is passed explicitly
    default: Precision,  # the default value if --precision is not passed
) -> argparse.ArgumentParser:
    precision_help = "Desired precision to which the model should be quantized."
    if Precision.float in supported_precisions:
        precision_help += " If set to 'float', the model will not be quantized, and inference will run in fp32 or fp16 (depending on compute unit)."

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--precision",
        action=get_quantize_action_with_default(default),
        default=default,
        metavar=f"{{{', '.join(str(p) for p in supported_precisions)}}}",
        help=precision_help,
    )
    if len(supported_precisions) > 1:
        quantize_options = [
            str(p) for p in supported_precisions if p != Precision.float
        ]
        group.add_argument(
            "--quantize",
            action=get_quantize_action_with_default(default_if_arg_explicitly_passed),
            default=None,
            metavar=f"{{{', '.join(quantize_options)}}}",
            help=f"Quantize the model to this precision. If passed without an explicit argument, precision {default_if_arg_explicitly_passed} will be used. If set, this always supercedes the '--precision' argument.",
            nargs="?",
        )
    return parser


def get_on_device_demo_parser(
    parser: QAIHMArgumentParser | None = None,
    supported_eval_modes: list[EvalMode] | None = None,
    supported_precisions: set[Precision] | None = None,
    available_target_runtimes: list[TargetRuntime] | None = None,
    add_output_dir: bool = False,
    default_device: str | None = None,
) -> QAIHMArgumentParser:
    """
    Get argument parser for on-device demo scripts.

    Parameters
    ----------
    parser
        Existing parser to add arguments to. If None, creates a new parser.
    supported_eval_modes
        Subset of EvalMode.{FP,QUANTSIM,ON_DEVICE,LOCAL_DEVICE}. Default is
        [EvalMode.FP, EvalMode.ON_DEVICE]. The first value of supported_eval_modes
        will be the default.
    supported_precisions
        Subset of {Precision.float, Precision.w8a8, Precision.w8a16}
    available_target_runtimes
        Available target runtimes for this model.
    add_output_dir
        Whether to add an output directory argument.
    default_device
        Default device to use for on-device execution.

    Returns
    -------
    parser : QAIHMArgumentParser
        Argument parser with all required arguments for on-device demos.
    """
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    if not parser:
        parser = get_parser()

    # Add --eval-mode
    supported_eval_modes = supported_eval_modes or [EvalMode.FP, EvalMode.ON_DEVICE]

    # take the first allowed mode as the default
    default_mode = supported_eval_modes[0]

    mode_help_lines = ["Run the model in one of the following modes:"]
    mode_help_lines.extend(
        f"  - {m.value}: {m.description}" for m in supported_eval_modes
    )
    mode_help_msg = "\n".join(mode_help_lines)

    parser.add_argument(
        "--eval-mode",
        type=EvalMode.from_string,
        choices=supported_eval_modes,
        default=default_mode,
        help=mode_help_msg,
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="If mode==on-device, uses this model Hub model ID."
        " Provide comma separated model-ids if multiple models are required for demo."
        " Run export.py to get on-device demo command with models exported for you.",
    )

    _add_device_args(parser, default_device=default_device)

    if add_output_dir:
        add_output_dir_arg(parser)
    parser.add_argument(
        "--inference-options",
        type=str,
        default="",
        help="If running on-device, use these options when submitting the inference job.",
    )
    default_runtime = _get_default_runtime(available_runtimes=available_target_runtimes)
    add_target_runtime_arg(
        parser,
        helpmsg="The runtime to demo (if `--eval-mode on-device` is specified).",
        default=default_runtime,
        available_target_runtimes=available_target_runtimes,
    )

    # TODO: This should only include supported precisions.
    default_precisions = [Precision.float, Precision.w8a8, Precision.w8a16]
    new_supported_precisions = supported_precisions or default_precisions

    add_precision_arg(
        parser,
        set(new_supported_precisions),
        next(iter(new_supported_precisions)),
        next(iter(new_supported_precisions)),
    )

    return parser


def validate_on_device_demo_args(args: argparse.Namespace, model_name: str) -> None:
    """
    Validates the the args for the on device demo are valid.

    Intended for use only in CLI scripts.
    Prints error to console and exits if an error is found.
    """
    is_on_device = args.eval_mode == EvalMode.ON_DEVICE
    if is_on_device and not can_access_qualcomm_ai_hub():
        raise ValueError(
            "On-device demos (--eval-mode on-device) are not available without Qualcomm® AI Hub Workbench access.\n",
            "Please sign up for Qualcomm® AI Hub Workbench at https://myaccount.qualcomm.com/signup .",
        )

    if is_on_device and (args.device is None):
        raise ValueError(
            "--device or --chipset must be specified with --eval-mode on-device."
        )

    if (args.inference_options or args.hub_model_id) and not is_on_device:
        raise ValueError(
            "A Hub model ID and inference options can be provided only with --eval-mode on-device."
        )


def add_function_parser_args(
    signature: dict[str, inspect.Parameter],
    parser: QAIHMArgumentParser,
    help_fn: Callable[[str, Any], str],
) -> None:
    """
    Given a function signature, add the inputs to the function as args to the parser.

    Parameters
    ----------
    signature
        The function signature represented as
        a dict from arg name to parameter metadata.
    parser
        The parser object to which the args are added.
    help_fn
        A function that takes the argument name and the default value
        and returns the help string for the that arg.
    """
    for name, param in signature.items():
        # Determining type from param.annotation is non-trivial (it can be a
        # strings like "bool | None").
        bool_action = None
        arg_name = f"--{name.replace('_', '-')}"
        if param.default is not None:
            type_ = type(param.default)
            if type_ is bool:
                if param.default:
                    bool_action = "store_false"
                    # If the default is true, and the arg name does not start with no_,
                    # then add the no- to the argument (as it should be passed as --no-enable-flag, not --enable-flag)
                    if name.startswith("no_"):
                        arg_name = f"--{name[3:].replace('_', '-')}"
                    elif name.startswith("skip_"):
                        arg_name = f"--do-{name[5:].replace('_', '-')}"
                    else:
                        arg_name = f"--no-{name.replace('_', '-')}"
                else:
                    bool_action = "store_true"
                    # If the default is false, and the arg name starts with no_,
                    # then remove the no- from the argument (as it should be passed as --enable-flag, not --no-enable-flag)
                    arg_name = f"--{name.replace('_', '-')}"
        elif param.annotation == "bool":
            type_ = bool
        else:
            type_ = str

        help_str = help_fn(name, param.default)
        if bool_action:
            parser.add_argument(arg_name, dest=name, action=bool_action, help=help_str)
        elif issubclass(type_, Enum):
            parser.add_argument(
                arg_name,
                type=str,
                action=partial(ParseEnumAction, enum_type=type_),  # type: ignore[arg-type]
                default=param.default,
                choices=[
                    enum.name.lower() for enum in list(type_.__members__.values())
                ],
                help=help_str,
            )
        else:
            parser.add_argument(
                arg_name,
                dest=name,
                type=type_,
                default=param.default,
                help=help_str,
            )


def get_model_cli_parser(
    cls: type[FromPretrainedTypeVar],
    parser: QAIHMArgumentParser | None = None,
    suppress_help_arguments: list | None = None,
    allow_dupe_args: bool = True,
) -> QAIHMArgumentParser:
    """
    Generate the argument parser to create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    if not parser:
        parser = get_parser(allow_dupe_args=allow_dupe_args)

    from_pretrained_sig = inspect.signature(cls.from_pretrained)

    export_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(cls.from_pretrained)["Parameters"]
    }

    def get_help(name: str, default_value: Any) -> str:
        # Suppress help for argument that need not be exposed for model.
        arg_name = f"--{name.replace('_', '-')}"
        if suppress_help_arguments is not None and arg_name in suppress_help_arguments:
            return argparse.SUPPRESS

        helpmsg = export_docs.get(
            name,
            (
                f"For documentation, see {cls.__name__}::from_pretrained::parameter {name}."
            ),
        )
        if default_value is True:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to False."
        elif default_value is False:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to True."
        return helpmsg

    signature = dict(from_pretrained_sig.parameters)
    if "cls" in signature:
        signature.pop("cls")
    if "precision" in signature:
        signature.pop("precision")
    add_function_parser_args(signature, parser, get_help)
    return parser


def add_input_spec_args(
    cls: type[FromPretrainedTypeVar],
    parser: QAIHMArgumentParser,
) -> QAIHMArgumentParser:
    """Adds arguments from get_input_spec."""
    if issubclass(cls, BaseModel):
        parser = get_model_input_spec_parser(cls, parser)
    elif issubclass(cls, CollectionModel):
        parser = get_collection_model_input_spec_parser(cls, parser)
    return parser


def get_model_kwargs(
    model_cls: type[FromPretrainedTypeVar], args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model via `from_pretrained`.
    """
    from_pretrained_sig = inspect.signature(model_cls.from_pretrained)
    model_kwargs = {}
    for name in from_pretrained_sig.parameters:
        if name in ["cls", "kwargs"] or name not in args_dict:
            continue
        model_kwargs[name] = args_dict.get(name)
    return model_kwargs


def get_export_model_name(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    precision: Precision | None,
    model_kwargs: Mapping[str, Any],
) -> str:
    """
    When exporting a model with custom model_kwargs, use a different name
    for the model file saved to disk. Incorporate all customized string args
    into the name.
    """
    sig = inspect.signature(model_cls.from_pretrained, eval_str=True)

    name = model_id
    if precision is not None:
        name += f"_{precision}"
    for key, value in sig.parameters.items():
        # Check for a simple string type.
        anno = value.annotation

        if key not in model_kwargs:
            continue

        from types import UnionType

        if not (
            anno is str
            or (
                isinstance(anno, UnionType) and any(arg is str for arg in anno.__args__)
            )
        ):
            continue

        if model_kwargs[key] != sig.parameters[key].default:
            # If the weights are a url or filepath, .stem will take the final name
            # in the path, it will also trim the suffix (i.e., yolov8n.pt -> yolov8n)
            # Note: if a string arg has a '/' character that is not part of a path,
            # we will erroneously truncate everything before it, which we're ok with.
            name += f"_{Path(str(model_kwargs[key])).stem}"
    return name


def model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar], cli_args: argparse.Namespace
) -> FromPretrainedTypeVar:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    return model_cls.from_pretrained(**get_model_kwargs(model_cls, vars(cli_args)))


def demo_model_components_from_cli_args(
    model_cls: type[PretrainedCollectionModel],
    model_id: str,
    cli_args: argparse.Namespace,
) -> tuple[FromPretrainedProtocol | OnDeviceModel, ...]:
    """
    Similar to demo_model_from_cli_args, but for component models.

    Parameters
    ----------
    model_cls
        Collection model class containing components.
    model_id
        Model ID string.
    cli_args
        Command line arguments namespace.

    Returns
    -------
    components : tuple[FromPretrainedProtocol | OnDeviceModel, ...]
        Model instances for each component.
    """
    res = []
    component_classes = model_cls.component_classes
    if cli_args.hub_model_id and len(cli_args.hub_model_id.split(",")) != len(
        component_classes
    ):
        raise ValueError(
            f"Expected {len(component_classes)} components in hub-model-id, but got {cli_args.hub_model_id}"
        )

    cli_args_comp = copy.deepcopy(cli_args)

    for i, (comp, cls) in enumerate(model_cls.component_classes.items()):
        if cli_args.hub_model_id:
            cli_args_comp.hub_model_id = cli_args.hub_model_id.split(",")[i]
        res.append(demo_model_from_cli_args(cls, model_id, cli_args_comp, comp))

    return tuple(res)


def demo_model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    cli_args: argparse.Namespace,
    component: str | None = None,
) -> FromPretrainedTypeVar | OnDeviceModel:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.

    If the model is a BaseModel and an on-device demo is requested,
        the BaseModel will be wrapped in an OnDeviceModel.
    """
    is_on_device = "eval_mode" in cli_args and cli_args.eval_mode == EvalMode.ON_DEVICE
    inference_model: FromPretrainedTypeVar | OnDeviceModel
    inference_model = model_from_cli_args(model_cls, cli_args)
    if is_on_device and issubclass(model_cls, BaseModel):
        assert isinstance(inference_model, BaseModel)
        device: hub.Device = cli_args.device
        if cli_args.hub_model_id:
            model_from_hub = hub.get_model(cli_args.hub_model_id)
            inference_model = OnDeviceModel(
                model_from_hub,
                list(inference_model.get_input_spec().keys()),
                device,
                cli_args.inference_options,
            )
        else:
            cli_dict = vars(cli_args)
            additional_kwargs = dict(
                get_model_kwargs(model_cls, args_dict=cli_dict),
                **get_input_spec_kwargs(model_cls, cli_dict),
            )
            target_model = compile_model_from_args(
                model_id,
                cli_args,
                additional_kwargs,
                component,
            )
            input_names = list(inference_model.get_input_spec().keys())
            inference_model = OnDeviceModel(
                target_model,
                input_names,
                device,
                inference_options=cli_args.inference_options,
            )
            print(
                f"Exported asset: {model_id}"
                + (f"::{component}" if component else "")
                + "\n"
            )

    return inference_model


def _parse_int_list(value: str) -> list[int]:
    """Parse a CLI value as a comma-separated list of ints."""
    return [int(v.strip()) for v in value.split(",")]


def _resolve_param_type(
    param: inspect.Parameter, model_cls: type[BaseModel | BasePrecompiledModel]
) -> type | Callable:
    """
    Resolve a parameter annotation to a concrete type or callable for argparse.

    locate() converts a string type annotation to a class type.
    Any type can be resolved as long as it's accessible in this scope.

    TODO(#16652): This is brittle since it requires the parameter
    to be imported into that scope exactly, which may not be its
    native location.
    """
    if param.annotation is inspect.Parameter.empty:
        raise TypeError(
            f"Parameter '{param.name}' of {model_cls.__name__}.get_input_spec "
            "has no type annotation."
        )
    if isinstance(param.annotation, type):
        return param.annotation
    anno = param.annotation
    type_ = locate(anno.split(" | ", 1)[0])
    if anno == "list[int]":
        return _parse_int_list
    if type_ is None:
        type_ = locate(f"{model_cls.__module__}.{anno}")
    if not isinstance(type_, type):
        raise TypeError(
            f"Annotation '{anno}' for '{param.name}' did not resolve "
            f"to a type (got {type_!r})."
        )
    return type_


def get_model_input_spec_parser(
    model_cls: type[BaseModel], parser: QAIHMArgumentParser | None = None
) -> QAIHMArgumentParser:
    """
    Generate the argument parser to get this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    """
    if not parser:
        parser = get_parser()

    input_spec_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(func=model_cls.get_input_spec)["Parameters"]
    }

    for name, param in get_params(model_cls.get_input_spec).items():
        resolved_type = _resolve_param_type(param, model_cls)
        help_str = input_spec_docs.get(
            name,
            f"For documentation, see {model_cls.__name__}::get_input_spec.",
        )
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=resolved_type,
            default=param.default,
            help=help_str,
        )
    return parser


def get_collection_model_input_spec_parser(
    model_cls: type[PretrainedCollectionModel],
    parser: QAIHMArgumentParser | None = None,
) -> QAIHMArgumentParser:
    """
    Generate CLI arguments for per-component input spec customization.

    For each component, adds ``--{component_name}-{param_name}`` arguments.
    """
    if not parser:
        parser = get_parser()

    for comp_name, comp_cls in model_cls.component_classes.items():
        params = get_params(comp_cls.get_input_spec)
        cli_prefix = comp_cls.cli_args_prefix  # type: ignore[attr-defined]
        if cli_prefix:
            cli_prefix += "-"

        input_spec_docs = {
            param.name: "\n".join(param.desc)
            for param in FunctionDoc(func=comp_cls.get_input_spec)["Parameters"]
        }

        for param_name, param in params.items():
            cli_name = f"--{cli_prefix.replace('_', '-')}{param_name.replace('_', '-')}"
            resolved_type = _resolve_param_type(param, comp_cls)
            if input_spec_docs.get(param_name):
                help_text = input_spec_docs[param_name]
            elif not comp_cls.cli_args_prefix:  #  type: ignore[attr-defined]
                help_text = f"Set {param_name}"
            else:
                help_text = f"Set {param_name} for {comp_name}"

            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            parser.add_argument(
                cli_name,
                type=resolved_type,
                default=default,
                help=help_text,
            )

    return parser


def get_input_spec_kwargs(
    model: type[WorkbenchModel | MultiGraphWorkbenchModel]
    | WorkbenchModel
    | MultiGraphWorkbenchModel,
    args_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model's input_spec.
    """
    return filter_kwargs(model.get_input_spec, args_dict)


def get_component_input_spec_kwargs(
    model_cls: type[PretrainedCollectionModel | MultiGraphCollectionModel],
    component_name: str,
    args_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract input spec kwargs for a specific component from CLI args.

    For each parameter in the component's ``get_input_spec`` signature:
    - Use the CLI-arg key (``{cli_args_prefix}_{param_name}`` or just
      ``{param_name}`` when ``cli_args_prefix`` is empty) if set
    - Else skip (component uses its default)
    """
    comp_cls = model_cls.component_classes[component_name]
    cli_prefix = getattr(comp_cls, "cli_args_prefix", component_name)
    kwargs: dict[str, Any] = get_input_spec_kwargs(comp_cls, args_dict)
    return {
        f"{cli_prefix}_{param_name}" if cli_prefix else param_name: kwarg
        for param_name, kwarg in kwargs.items()
    }


def input_spec_from_cli_args(
    model: WorkbenchModel | OnDeviceModel, cli_args: argparse.Namespace
) -> InputSpec | hub.InputSpecs:
    """
    Create this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    Also, fetches shapes if demo is run on-device.
    """
    if isinstance(model, OnDeviceModel):
        assert "on_device" in cli_args and cli_args.on_device
        assert isinstance(model.model.producer, hub.CompileJob)
        return model.model.producer.shapes
    return model.get_input_spec(**get_input_spec_kwargs(model, vars(cli_args)))


def _evaluate_export_common_parser(
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    omit_precision: bool = False,
) -> QAIHMArgumentParser:
    """Common arguments between export and evaluate scripts."""
    # Set handler to resolve, to allow from_pretrained and get_input_spec
    # to have the same argument names.
    parser = get_parser(
        model_cls,
        supported_precision_runtimes,
        allow_dupe_args=True,
    )
    # Default runtime for compiled model is fixed for given model
    # Python doesn't have ordered sets, so use a dictionary to preserver order
    available_runtimes: dict[TargetRuntime, None] = {}
    for rts in supported_precision_runtimes.values():
        for rt in rts:
            available_runtimes[rt] = None

    available_runtimes_list = list(available_runtimes.keys())
    # Default is resolved dynamically in parse_args based on the chosen precision.
    add_target_runtime_arg(
        parser,
        available_target_runtimes=available_runtimes_list,
        default=None,
        helpmsg="The runtime for which to export. Default is chosen based on the precision.",
    )
    if issubclass(model_cls, FromPretrainedProtocol):
        # Skip adding CLI from model for compiled model
        # TODO: #9408 Refactor BaseModel, BasePrecompiledModel to fetch
        # parameters from compiled model
        parser = get_model_cli_parser(model_cls, parser)
        parser = add_input_spec_args(model_cls, parser)

        supported_precisions = {
            precision
            for precision, rts in supported_precision_runtimes.items()
            if len(rts) > 0
        }
        non_float_precision = _get_non_float_precision(supported_precisions)
        if not omit_precision:
            add_precision_arg(
                parser,
                supported_precisions,
                default_if_arg_explicitly_passed=non_float_precision or Precision.float,
                default=(
                    Precision.float
                    if (
                        len(supported_precisions) == 0
                        or Precision.float in supported_precisions
                    )
                    else next(iter(supported_precisions))
                ),
            )

    return parser


def add_export_function_args(
    export_fn: Callable,
    parser: QAIHMArgumentParser,
    force_fetch_static_assets: bool = False,
    zip_assets: bool = False,
) -> None:
    """
    Extracts the relevant inputs to the export function and
    adds them to the parser.
    """
    signature = dict(inspect.signature(export_fn).parameters)
    for key in [
        "components",
        "precision",
        "target_runtime",
        "additional_model_kwargs",
        # LLM specific args
        "model_cls",
        "position_processor_cls",
        "model_id",
        "model_name",
        "model_asset_version",
        "sub_components",
        "num_layers_per_split",
        "num_splits",
    ]:
        if key in signature:
            signature.pop(key)

    if "fetch_static_assets" in signature:
        signature.pop("fetch_static_assets")
        parser.add_argument(
            "--fetch-static-assets",
            nargs="?",
            const=QAIHMVersion.CURRENT_TAG_ALIAS,
            default=QAIHMVersion.CURRENT_TAG_ALIAS
            if force_fetch_static_assets
            else None,
            help="If set, known assets are fetched rather than re-computing them. Can be passed as:\n"
            "    `--fetch-static-assets`            (get current release assets)\n"
            "    `--fetch-static-assets latest`     (get latest release assets)\n"
            "    `--fetch-static-assets v<version>` (get assets for a specific version)\n",
        )
    if "zip_assets" in signature:
        signature.pop("zip_assets")
        parser.add_argument(
            "--zip-assets",
            action="store_true",
            help="If set, downloaded assets are zipped.",
        )

    raw_doc = inspect.getdoc(export_fn)
    assert raw_doc is not None, "Export function must have a docstring."

    export_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(export_fn)["Parameters"]
    }

    def _get_export_help(param_name: str, default_value: Any) -> str:
        description = export_docs[param_name]
        assert description is not None, f"Input `{param_name}` must have a description."
        if default_value is True:
            description = description.replace("skips", "does")
        return description

    add_function_parser_args(signature, parser, _get_export_help)

    if "quantized_model_id" in signature:
        signature.pop("quantized_model_id")
        parser.add_argument(
            "--quantized-model-id",
            type=str,
            default=None,
            help="If set, uses this quantized model ID instead of quantizing during export. "
            "For BaseModel: --quantized-model-id <id>. "
            "For CollectionModel: --quantized-model-id <id1,id2,...>. "
            "For CollectionModel, IDs must be provided in the same order as the selected components in --components.",
        )


def export_parser(
    model_cls: type[FromPrecompiledProtocol | FromPretrainedProtocol],
    export_fn: Callable,
    components: list[str] | None = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    default_export_device: str | None = None,
    force_fetch_static_assets: bool = False,
    zip_assets: bool = False,
    omit_precision: bool = False,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in export scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional
        args for model instantiation.
    export_fn
        Export function to extract parameters from.
    components
        Only used for model with component and sub-component, such
        as Llama 2, 3, where two subcomponents (e.g.,
        PromptProcessor_1, TokenGenerator_1)
        are classified under one component (e.g. Llama2_Part1_Quantized).
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    default_export_device
        Default device to set for export.
    force_fetch_static_assets
        If set, fetch_static_assets is always enabled and cannot be turned off.
    zip_assets
        Zips downloaded assets. If set, adds --zip-assets argument to the parser.
    omit_precision
        Do not register --precision.

    Returns
    -------
    parser : QAIHMArgumentParser
        ArgumentParser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {
            Precision.float: [TargetRuntime.TFLITE],
        }
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        omit_precision=omit_precision,
    )
    add_export_function_args(export_fn, parser, force_fetch_static_assets, zip_assets)
    _add_device_args(parser, default_device=default_export_device)
    if components is not None or issubclass(model_cls, CollectionModel):
        choices = components or []
        if not choices and issubclass(model_cls, CollectionModel):
            choices = model_cls.component_class_names
        parser.add_argument(
            "--components",
            nargs="+",
            type=str,
            default=None,
            choices=choices,
            help="Which components of the model to be exported.",
        )

    return parser


def evaluate_parser(
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar],
    supported_dataset_classes: list[type[BaseDataset]],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    uses_quantize_job: bool = True,
    num_calibration_samples: int | None = None,
    default_device: str | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in evaluate scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional args for model instantiation.
    supported_dataset_classes
        List of supported dataset classes (subclasses of BaseDataset).
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    uses_quantize_job
        Whether this model uses quantize job to quantize the model.
    num_calibration_samples
        How many samples to calibrate on when quantizing by default.
        If not set, defers to the dataset to decide the number.
    default_device:
        The default device to use for export + eval.

    Returns
    -------
    parser : QAIHMArgumentParser
        ArgumentParser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {Precision.float: [TargetRuntime.TFLITE]}
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
    )
    parser.set_supported_dataset_classes(supported_dataset_classes)
    parser.add_argument(
        "--compile-options",
        type=str,
        default="",
        help="Additional options to pass when submitting the compile job.",
    )
    parser.add_argument(
        "--profile-options",
        type=str,
        default="",
        help="Additional options to pass when submitting the profile job.",
    )
    if uses_quantize_job:
        parser.add_argument(
            "--quantize-options",
            type=str,
            default="",
            help="Additional options to pass when submitting the quantize job.",
        )

    _add_device_args(parser, default_device)
    if not parser._dataset_name_to_cls:
        return parser
    supported_dataset_names = list(parser._dataset_name_to_cls.keys())
    if uses_quantize_job:
        parser.add_argument(
            "--num-calibration-samples",
            type=int,
            default=num_calibration_samples,
            help="The number of calibration data samples to use for quantization.",
        )
    parser.add_argument(
        "--samples-per-job",
        type=int,
        default=None,
        help="Max size to be submitted in a single inference job.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=supported_dataset_names[0],
        choices=supported_dataset_names,
        help="Name of the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run. If set to -1, will run on full dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use when shuffling the data. If not set, samples data deterministically.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="A compiled hub model id.",
    )
    parser.add_argument(
        "--use-dataset-cache",
        action="store_true",
        help="If set, will store hub dataset ids in a local file and re-use "
        "for subsequent evaluations on the same dataset.",
    )
    if uses_quantize_job:
        parser.add_argument(
            "--compute-quant-cpu-accuracy",
            action="store_true",
            help="If flag is set, computes the accuracy of the quantized onnx model on the CPU.",
        )
    parser.add_argument(
        "--skip-device-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy on device.",
    )
    parser.add_argument(
        "--skip-torch-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy with the torch model.",
    )
    return parser
