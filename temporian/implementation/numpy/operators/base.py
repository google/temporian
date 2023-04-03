from abc import ABC, abstractmethod

from typing import Dict
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator, OperatorExceptionDecorator


class OperatorImplementation(ABC):
    def __init__(self, operator: Operator):
        self._operator = operator
        # TODO: Check operator type

    @property
    def operator(self):
        return self._operator

    def call(self, **inputs: Dict[str, NumpyEvent]) -> Dict[str, NumpyEvent]:
        """Like __call__, but with checks."""

        _check_input(inputs=inputs, operator=self.operator)
        outputs = self(**inputs)
        _check_output(inputs=inputs, outputs=outputs, operator=self.operator)
        return outputs

    @abstractmethod
    def __call__(
        self, **inputs: Dict[str, NumpyEvent]
    ) -> Dict[str, NumpyEvent]:
        pass


def _check_features(
    values: Dict[str, NumpyEvent],
    definitions: Dict[str, Event],
    label: str,
) -> None:
    """Checks if features are matching their definition."""

    # TODO: Check that the index and features have the same number of
    # observations.

    for key, item_def in definitions.items():
        item_real = values[key]

        # Check sampling
        if item_real.sampling.index != item_def.sampling.index:
            raise RuntimeError(
                f"Non matching {label} sampling. "
                f"effective={item_real.sampling.index} vs "
                f"expected={item_def.sampling.index}"
            )
        # Check features
        features = item_real._first_index_features

        if len(item_def.features) != len(features):
            raise RuntimeError(
                f"Non matching number of {label} features. "
                f"expected={len(item_def.features)} vs "
                f"effective={len(features)}"
            )

        for feature_def, feature in zip(item_def.features, features):
            if feature_def.name != feature.name:
                raise RuntimeError(
                    f"Non matching {label} feature name. "
                    f"expected={feature_def.name} vs "
                    f"effective={feature.name}"
                )

            if feature_def.dtype != feature.dtype:
                raise RuntimeError(
                    f"Non matching {label} feature dtype. "
                    f"expected={feature_def.dtype} vs "
                    f"effective={feature.dtype}"
                )


def _check_input(
    inputs: Dict[str, NumpyEvent],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check input keys
        effective_input_keys = set(inputs.keys())
        expected_input_keys = set(operator.inputs.keys())
        if effective_input_keys != expected_input_keys:
            raise RuntimeError(
                "Non matching number of inputs. "
                f"{effective_input_keys} vs {expected_input_keys}"
            )

        _check_features(inputs, definitions=operator.inputs, label="input")


def _check_output(
    inputs: Dict[str, NumpyEvent],
    outputs: Dict[str, NumpyEvent],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check output keys
        effective_output_keys = set(outputs.keys())
        expected_output_keys = set(operator.outputs.keys())
        if effective_output_keys != expected_output_keys:
            raise RuntimeError(
                "Non matching number of outputs. "
                f"{effective_output_keys} vs {expected_output_keys}"
            )

        for output_key, output_def in operator.outputs.items():
            output_real = outputs[output_key]

            # Check sampling
            if output_real.sampling.index != output_def.sampling.index:
                raise RuntimeError(
                    f"Non matching sampling. {output_real.sampling.index} vs"
                    f" {output_def.sampling.index}"
                )

            # TODO: Check copy or referencing of feature data.

            # Check copy or referencing of sampling data.
            matching_samplings = set(operator.list_matching_io_samplings())
            for input_key in operator.inputs.keys():
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
                        f"during the op execution ({input_real.sampling} "
                        f"vs {output_real.sampling})."
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
        _check_features(outputs, definitions=operator.outputs, label="outputs")
