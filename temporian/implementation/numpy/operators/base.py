import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from temporian.core.data.node import Node
from temporian.utils import config
from temporian.core.operators.base import Operator
from temporian.core.operators.base import OperatorExceptionDecorator
from temporian.implementation.numpy.data.event_set import DTYPE_MAPPING
from temporian.implementation.numpy.data.event_set import EventSet


class OperatorImplementation(ABC):
    def __init__(self, operator: Operator):
        self._operator = operator
        # TODO: Check operator type

    @property
    def operator(self):
        return self._operator

    def call(self, **inputs: Dict[str, EventSet]) -> Dict[str, EventSet]:
        """Like __call__, but with checks."""

        _check_input(inputs=inputs, operator=self.operator)
        outputs = self(**inputs)
        _check_output(inputs=inputs, outputs=outputs, operator=self.operator)
        return outputs

    @abstractmethod
    def __call__(self, **inputs: Dict[str, EventSet]) -> Dict[str, EventSet]:
        """Applies the operator to its inputs."""


def _check_features(
    values: Dict[str, EventSet],
    definitions: Dict[str, Node],
    label: str,
) -> None:
    """Checks if features are matching their definition."""

    # TODO: Check that the index and features have the same number of
    # observations.
    for key, item_def in definitions.items():
        item_real = values[key]

        # Check sampling
        if item_real.index_names != item_def.sampling.index.names:
            raise RuntimeError(
                f"Non matching {label} sampling. "
                f"effective={item_real.index_names} vs "
                f"expected={item_def.sampling.index.names}"
            )

        # Check features
        if len(item_def.features) != item_real.feature_count:
            raise RuntimeError(
                f"Non matching number of {label} features. "
                f"expected={len(item_def.features)} vs "
                f"effective={item_real.feature_count}"
            )

        for i, feature_def in enumerate(item_def.features):
            if feature_def.name != item_real.feature_names[i]:
                raise RuntimeError(
                    f"Non matching {label} feature name. "
                    f"expected={feature_def.name} vs "
                    f"effective={item_real.feature_names[i]}"
                )
            feat_dtype_real = DTYPE_MAPPING[
                item_real.first_index_data().features[i].dtype.type
            ]
            if feature_def.dtype != feat_dtype_real:
                raise RuntimeError(
                    f"Non matching {label} feature dtype. "
                    f"expected={feature_def.dtype} vs "
                    f"effective={feat_dtype_real}"
                )


def _check_input(
    inputs: Dict[str, EventSet],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check input keys
        effective_input_keys = set(inputs.keys())
        expected_input_keys = set(operator.inputs.keys())
        if effective_input_keys != expected_input_keys:
            raise RuntimeError(
                "Input keys do not match the expected ones. "
                f"Received: {effective_input_keys}. "
                f"Expected: {expected_input_keys}."
            )

        _check_features(inputs, definitions=operator.inputs, label="input")


def _check_output(
    inputs: Dict[str, EventSet],
    outputs: Dict[str, EventSet],
    operator: Operator,
) -> None:
    """Checks if the input/output of an operator matches its definition."""

    with OperatorExceptionDecorator(operator):
        # Check output keys
        effective_output_keys = set(outputs.keys())
        expected_output_keys = set(operator.outputs.keys())
        if effective_output_keys != expected_output_keys:
            raise RuntimeError(
                "Output keys do not match the expected ones. "
                f"Received: {effective_output_keys}. "
                f"Expected: {expected_output_keys}."
            )

        for output_key, output_def in operator.outputs.items():
            output_real = outputs[output_key]

            # Check sampling
            if output_real.index_names != output_def.sampling.index.names:
                # TODO: also check is_unix_timestamp
                raise RuntimeError(
                    f"Non matching sampling. {output_real.index_names} vs"
                    f" {output_def.sampling.index.names}"
                )

            # TODO: Check copy or referencing of feature data.
            matching_samplings = set(operator.list_matching_io_samplings())
            for input_key in operator.inputs.keys():
                input_real = inputs[input_key]
                expected_matching_sampling = (
                    input_key,
                    output_key,
                ) in matching_samplings
                is_same, reason = _check_same_sampling(output_real, input_real)
                if expected_matching_sampling and not is_same:
                    raise RuntimeError(
                        f"The sampling of input '{input_key}' and output "
                        f"'{output_key}' are expected to have THE SAME "
                        "sampling. However, a different sampling was generated "
                        f"during the op execution ({input_real} "
                        f"vs {output_real}). Reason: {reason}"
                    )

        # Check features
        _check_features(outputs, definitions=operator.outputs, label="outputs")


def _check_same_sampling(
    evset_1: EventSet, evset_2: EventSet
) -> Tuple[bool, str]:
    # number of index keys to check in default mode
    num_check = 1000 if config.DEBUG_MODE else len(evset_1.data)
    # compare index names
    if evset_1.index_names != evset_2.index_names:
        return (False, "Different index names")

    # compare timestamps for `num_check` index keys
    for i, (index_key, index_data_1) in enumerate(evset_1.data.items()):
        if i >= num_check:
            # checked all index keys' timestamps
            break

        index_data_2 = evset_2[index_key]
        if index_data_1.timestamps is not index_data_2.timestamps:
            return (
                False,
                (
                    f"Timestamps at index key {index_key} are not the same"
                    " np.ndarray"
                ),
            )

    # compare index keys
    # TODO: is there a way to avoid checking all keys here (keys might come in
    # different orders, can't compare top num_check keys in each evset)
    diff_keys = set(evset_1.data.keys()).difference(evset_2.data.keys())
    if diff_keys:
        return (
            False,
            f"Found {len(diff_keys)} different index keys",
        )

    return (True, "")
