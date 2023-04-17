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

"""Type cast operator."""
from typing import List, Union, Mapping
from temporian.core.data.feature import Feature

from temporian.core.data.dtype import DType, ALL_TYPES
from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class CastOperator(Operator):
    """Type cast operator."""

    def __init__(
        self,
        event: Event,
        target: Union[DType, Mapping[Union[str, DType], DType]],
        check_overflow: bool = True,
    ):
        super().__init__()

        # Check that all origin keys are DType or feature_names
        if target not in ALL_TYPES:  # Will be 'not isinstance(to, DType)'
            if not isinstance(target, Mapping):
                raise ValueError(
                    "Cast parameter 'to' must be a temporian dtype or a Mapping"
                )
            for origin_key in target:
                if (
                    origin_key not in ALL_TYPES  # Also here
                    and origin_key not in event.feature_names
                ):
                    raise KeyError(
                        f"Invalid key to cast: {origin_key}. "
                        "Expected dtype or feature name."
                    )

        # Convert any input format to feature_name->target_dtype
        target_dtypes = {}
        for feature in event.features:
            if target in ALL_TYPES:
                # "to" is a dtype, not map: cast all features this dtype
                target_dtypes[feature.name] = target
            elif feature.name in target:
                # Cast by feature name
                target_dtypes[feature.name] = target[feature.name]
            elif feature.dtype in target:
                # Cast by dtype
                target_dtypes[feature.name] = target[feature.dtype]
            else:
                # Keep same dtype
                target_dtypes[feature.name] = feature.dtype
        self.add_attribute("target_dtypes", target_dtypes)
        self.add_attribute("check_overflow", check_overflow)

        # inputs
        self.add_input("event", event)

        # outputs
        output_features = []
        reuse_event = True
        for feature in event.features:
            if target_dtypes[feature.name] is feature.dtype:
                # Reuse feature
                output_features.append(feature)
            else:
                # Create new feature
                reuse_event = False
                output_features.append(
                    # Note: we're not renaming output features here
                    Feature(
                        feature.name,
                        target_dtypes[feature.name],
                        feature.sampling,
                        creator=self,
                    )
                )

        # Output event: don't create new if all features are reused
        self.add_output(
            "event",
            event
            if reuse_event
            else Event(
                features=output_features,
                sampling=event.sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="CAST",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="target_dtypes",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                    is_optional=False,
                ),
                pb.OperatorDef.Attribute(
                    key="check_overflow",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(CastOperator)


def cast(
    event: Event,
    target: Union[DType, Mapping[Union[str, DType], DType]],
    check_overflow: bool = True,
) -> Event:
    """
    Changes the dtype of event features to the type specified in `target`.
    Feature names are preserved, and reused (not copied) if not changed.

    Args:
        event:
            The input `Event` object to cast the columns from.
        target:
            A single dtype or a map. Providing a single dtype will cast all
            columns to it. The mapping keys can be either feature names or
            the original dtypes, and the values are the target dtypes for them.
            All dtypes must be temporian types (see `dtype.py`)
        check_overflow:
            A flag to check overflow when casting to a dtype with a shorter
            range (e.g: `INT64`->`INT32`).
            Note that this check adds some computation overhead.
            Defaults to `True`.

    Returns:
        A new `Event` (or the same if no features actually changed dtype), with
        the same feature names as the input one, but with the new dtypes as
        specified in `to`.

    Raises:
        ValueError:
            If `check_overflow=True` and some value is out of the range of the
            target dtype.
        ValueError:
            If trying to cast a non-numeric string to numeric dtype.
        ValueError:
            If the `target` parameter is not a dtype nor a mapping.
        KeyError:
            If `target` is a mapping, but some of the keys are not a dtype nor
            a feature in `event.feature_names`.

    Examples:
        Given an input `Event` with features 'A' (`INT64`), 'B'
        (`INT64`), 'C' (`FLOAT64`) and 'D' (`STRING`):

        1. `cast(event, target=dtype.INT32)`
           Try to convert all features to `INT32`, or raise `ValueError` if some
           string value in 'D' is invalid, or any column value is out of range
           for `INT32`.

        2. `cast(event, target={dtype.INT64: dtype.INT32, dtype.STRING: dtype.FLOAT32})`
            Convert features 'A' and 'B' to `INT32`, 'D' to `FLOAT32`, leave 'C'
            unchanged.

        3. `cast(event, target={'A': dtype.FLOAT32, dtype.INT64: dtype.INT32})`
            Convert 'A' to `FLOAT32` (feature name key takes precedence) and any
            other feature of type `INT64` to `INT32` (in this case, only 'B').
    """
    return CastOperator(event, target, check_overflow).outputs["event"]
