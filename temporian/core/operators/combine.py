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


"""Combine operator class and public API function definitions."""
from enum import Enum
from typing import Any, Union

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck

_INPUT_KEY_PREFIX = "input_"


class How(str, Enum):
    outer = "outer"
    inner = "inner"
    left = "left"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        return isinstance(value, How) or (
            isinstance(value, str) and value in [item.value for item in How]
        )


class Combine(Operator):
    def __init__(self, how: How, **inputs: EventSetNode):
        super().__init__()

        # Note: Support for dictionaries of nodes is required for
        # serialization.

        if len(inputs) < 2:
            raise ValueError("At least two arguments should be provided")

        # Attributes
        self._how = how
        self.add_attribute("how", how)

        # inputs
        first_input = None
        all_unix_timestamp = True
        for key, input in inputs.items():
            # Check that all features are in all nodes
            if first_input is None:
                first_input = input
            first_input.schema.check_compatible_features(
                input.schema, check_order=False
            )
            first_input.schema.check_compatible_index(input.schema)

            # Output is unix if all inputs are
            all_unix_timestamp &= input.schema.is_unix_timestamp
            self.add_input(key, input)

        assert first_input is not None  # for static checker
        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=first_input.schema.features,
                indexes=first_input.indexes,
                is_unix_timestamp=all_unix_timestamp,
                creator=self,
            ),
        )
        self.check()

    @property
    def how(self) -> str:
        return self._how

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="COMBINE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="how",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key_prefix=_INPUT_KEY_PREFIX)],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Combine)


@typecheck
@compile
def combine(
    *inputs: EventSetOrNode,
    how: Union[str, How] = How.outer,
) -> EventSetOrNode:
    """
    Combines events from multiple [`EventSets`][temporian.EventSet] together.

    Input events must have the same features (i.e. same feature names and dtypes)
    and index schemas (i.e. same index names and dtypes).

    Combine is different from `glue` and `join`, since those append together
    different features.

    Args:
        *inputs: EventSets to combine their events.
        how: Whether to use the indexes from "outer" (union of all inputs' index
             values), "inner" (only those present in all inputs) or "left"
             (only use index values from the first input).

    Basic example:

        ```python
        >>> a = tp.event_set(timestamps=[0, 1, 3],
        ...                  features={"A": [0, 10, 30], "B": [0, -10, -30]}
        ...                 )
        >>> b = tp.event_set(timestamps=[1, 4],
        ...                  features={"A": [10, 40], "B": [-10, -40]}
        ...                 )

        >>> # Inputs a and b have some duplicated timestamps
        >>> c = tp.combine(a, b)
        >>> c
        indexes: []
        features: [('A', int64), ('B', int64)]
        events:
            (5 events):
                timestamps: [0. 1. 1. 3. 4.]
                'A': [ 0 10 10 30 40]
                'B': [ 0 -10 -10 -30 -40]
        ...

        >>> # Events with duplicated timestamps can be unified afterwards
        >>> unique_t = c.unique_timestamps()
        >>> d = c.moving_sum(window_length=tp.duration.shortest, sampling=unique_t)
        >>> d
        indexes: []
        features: [('A', int64), ('B', int64)]
        events:
            (4 events):
                timestamps: [0. 1. 3. 4.]
                'A': [ 0 20 30 40]
                'B': [ 0 -20 -30 -40]
        ...

        ```

    Example with different index values

        ```python
        # Index "idx=a" is only in a, "idx=b" in both, "idx=c" only in b
        >>> a = tp.event_set(timestamps=[0, 1, 3],
        ...                  features={"A": [0, 10, 30],
        ...                            "idx": ["a", "a", "b"]},
        ...                  indexes=["idx"],
        ...                 )
        >>> b = tp.event_set(timestamps=[1.5, 4.5, 5.5],
        ...                  features={"A": [15, 45, 55],
        ...                            "idx": ["b", "c", "c"]},
        ...                  indexes=["idx"]
        ...                 )

        >>> # By default, "outer" uses index values from all inputs
        >>> c = tp.combine(a, b)
        >>> c
        indexes: [('idx', str_)]
        features: [('A', int64)]
        events:
            idx=b'a' (2 events):
                timestamps: [0. 1.]
                'A': [ 0 10]
            idx=b'b' (2 events):
                timestamps: [1.5 3. ]
                'A': [15 30]
            idx=b'c' (2 events):
                timestamps: [4.5 5.5]
                'A': [45 55]
        ...

        >>> # Use "left" to use only index values from the first input a
        >>> c = tp.combine(a, b, how="left")
        >>> c
        indexes: [('idx', str_)]
        features: [('A', int64)]
        events:
            idx=b'a' (2 events):
                timestamps: [0. 1.]
                'A': [ 0 10]
            idx=b'b' (2 events):
                timestamps: [1.5 3. ]
                'A': [15 30]
        ...

        >>> # Use "inner" to use only index values that are present in all inputs
        >>> c = tp.combine(a, b, how="inner")
        >>> c
        indexes: [('idx', str_)]
        features: [('A', int64)]
        events:
            idx=b'b' (2 events):
                timestamps: [1.5 3. ]
                'A': [15 30]
        ...

        ```

    Returns:
        An EventSet with events from all inputs combined.
    """
    if not How.is_valid(how):
        raise ValueError(f"Invalid argument: {how=}. Options are {list(How)}")
    how = How[how]

    if len(inputs) == 1 and isinstance(inputs[0], EventSetNode):
        return inputs[0]

    # NOTE: input name must match op. definition name
    inputs_dict = {
        f"{_INPUT_KEY_PREFIX}{idx}": input for idx, input in enumerate(inputs)
    }
    return Combine(how=how, **inputs_dict).outputs["output"]  # type: ignore
