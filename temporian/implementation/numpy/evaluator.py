# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from typing import Dict, List

from temporian.core.data import event
from temporian.core.operators import base
from temporian.implementation.numpy.data import event as numpy_event
from temporian.implementation.numpy import implementation_lib
from temporian.core.operators.base import Operator, OperatorExceptionDecorator


def evaluate_schedule(
    inputs: Dict[event.Event, numpy_event.NumpyEvent],
    schedule: List[base.Operator],
    verbose: int,
    check_execution: bool,
) -> Dict[event.Event, numpy_event.NumpyEvent]:
    data = {**inputs}

    for operator in schedule:
        operator_def = operator.definition()

        # Get implementation
        implementation_cls = implementation_lib.get_implementation_class(
            operator_def.key
        )

        # Instantiate implementation
        implementation = implementation_cls(operator)

        if verbose >= 2:
            print(f"Execute {operator}", file=sys.stderr)

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_event]
            for input_key, input_event in operator.inputs().items()
        }

        if check_execution:
            _check_input(
                inputs=operator_inputs,
                operator=operator,
            )

        # Compute output
        operator_outputs = implementation(**operator_inputs)

        if check_execution:
            _check_output(
                inputs=operator_inputs,
                outputs=operator_outputs,
                operator=operator,
            )

        # materialize data in output events
        for output_key, output_event in operator.outputs().items():
            data[output_event] = operator_outputs[output_key]

    # TODO: Only return the required data.
    # TODO: Un-allocate not used anymore object.
    return data


def _check_features(
    values: Dict[str, numpy_event.NumpyEvent],
    definitions: Dict[str, event.Event],
    label: str,
) -> None:
    """Checks if features are matching their definition."""

    # TODO: Check that the index and features have the same number of
    # observations.

    for key, item_def in definitions:
        item_real = values[key]

        # Check sampling
        if item_real.sampling.index != item_def.sampling().index():
            raise RuntimeError(
                f"Non matching {label} sampling. "
                f"effective={item_real.sampling.index} vs "
                f"expected={item_def.sampling().index()}"
            )
        # Check features
        features = item_real._first_index_features

        if len(item_def.features()) != len(features):
            raise RuntimeError(
                f"Non matching number of {label} features. "
                f"expected={len(item_def.features())} vs "
                f"effective={len(features)}"
            )

        for feature_def, feature in zip(item_def.features(), features):
            if feature_def.name() != feature.name:
                raise RuntimeError(
                    f"Non matching {label} feature name. "
                    f"expected={feature_def.name()} vs "
                    f"effective={feature.name}"
                )

            if feature_def.dtype() != feature.dtype:
                raise RuntimeError(
                    f"Non matching {label} feature dtype. "
                    f"expected={feature_def.dtype()} vs "
                    f"effective={feature.dtype}"
                )


def _check_input(
    inputs: Dict[str, numpy_event.NumpyEvent],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check input keys
        effective_input_keys = set(inputs.keys())
        expected_input_keys = set(operator.inputs().keys())
        if effective_input_keys != expected_input_keys:
            raise RuntimeError(
                "Non matching number of inputs. "
                f"{effective_input_keys} vs {expected_input_keys}"
            )

        _check_features(
            inputs, definitions=operator.inputs().items(), label="input"
        )


def _check_output(
    inputs: Dict[str, numpy_event.NumpyEvent],
    outputs: Dict[str, numpy_event.NumpyEvent],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check output keys
        effective_output_keys = set(outputs.keys())
        expected_output_keys = set(operator.outputs().keys())
        if effective_output_keys != expected_output_keys:
            raise RuntimeError(
                "Non matching number of outputs. "
                f"{effective_output_keys} vs {expected_output_keys}"
            )

        for output_key, output_def in operator.outputs().items():
            output_real = outputs[output_key]

            # Check sampling
            if output_real.sampling.index != output_def.sampling().index():
                raise RuntimeError(
                    f"Non matching sampling. {output_real.sampling.index} vs"
                    f" {output_def.sampling().index()}"
                )

            # TODO: Check copy or referencing of feature data.

            # Check copy or referencing of sampling data.
            matching_samplings = set(operator.list_matching_io_samplings())
            for input_key in operator.inputs().keys():
                input_real = inputs[input_key]
                expected_matching_sampling = (
                    input_key,
                    output_key,
                ) in matching_samplings
                effective_matching_sampling = (
                    output_real.sampling is input_real.sampling
                )
                assert effective_matching_sampling == (
                    output_real.sampling.data is input_real.sampling.data
                )
                if (
                    expected_matching_sampling
                    and not effective_matching_sampling
                ):
                    raise RuntimeError(
                        f"The sampling of input '{input_key}' and output "
                        f"'{output_key}' are expected to have THE SAME "
                        "sampling. However, a different sampling was generated "
                        "during the op execution."
                    )
                if (
                    not expected_matching_sampling
                    and effective_matching_sampling
                ):
                    raise RuntimeError(
                        f"The sampling of input '{input_key}' and output "
                        f"'{output_key}' are expected to have A DIFFERENT "
                        "sampling. However, the same sampling was generated "
                        "during the op execution."
                    )

        # Check features
        _check_features(
            outputs, definitions=operator.outputs().items(), label="outputs"
        )
