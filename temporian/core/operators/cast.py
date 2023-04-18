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
from typing import Union, Mapping, Optional
from temporian.core.data.feature import Feature

from temporian.core.data.dtype import DType
from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class CastOperator(Operator):
    """Type cast operator."""

    def __init__(
        self,
        event: Event,
        to_dtype: Optional[DType] = None,
        from_dtypes: Optional[Mapping[DType, DType]] = None,
        from_features: Optional[Mapping[str, DType]] = None,
        check_overflow: bool = True,
    ):
        super().__init__()

        # Check that provided arguments are coherent
        self._check_args(event, to_dtype, from_dtypes, from_features)

        # Convert to {feature_name: target_dtype}
        target_dtypes = self._get_feature_dtype_map(
            event, to_dtype, from_dtypes, from_features
        )

        # Attributes
        self.add_attribute("target_dtypes", target_dtypes)
        self.add_attribute("check_overflow", check_overflow)

        # Inputs
        self.add_input("event", event)

        # Output event features
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

    def _check_args(
        self,
        event: Event,
        to_dtype: Optional[DType] = None,
        from_dtypes: Optional[Mapping[DType, DType]] = None,
        from_features: Optional[Mapping[str, DType]] = None,
    ) -> None:
        # Check that only one of these args was provided
        oneof_args = [to_dtype, from_dtypes, from_features]
        if sum(arg is not None for arg in oneof_args) != 1:
            raise ValueError(
                "One and only one of to_dtype, from_dtypes or from_features"
                " should be provided to CastOperator."
            )

        # Check: to_dtype is actually a dtype
        if to_dtype is not None and not isinstance(to_dtype, DType):
            raise ValueError(f"Cast: expected DType but got {type(to_dtype)=}")

        # Check: from_dtypes is a dict {dtype: dtype}
        if from_dtypes is not None and (
            not isinstance(from_dtypes, Mapping)
            or any(not isinstance(key, DType) for key in from_dtypes)
        ):
            raise ValueError(
                "Cast: target can be a dict with only DType or feature keys"
            )

        # Check: from_features is {feature_name: dtype}
        if from_features is not None and (
            not isinstance(from_features, Mapping)
            or any(key not in event.feature_names for key in from_features)
        ):
            raise ValueError(
                "Cast: target can be a dict with only DType or feature keys"
            )

    def _get_feature_dtype_map(
        self,
        event: Event,
        to_dtype: Optional[DType] = None,
        from_dtypes: Optional[Mapping[DType, DType]] = None,
        from_features: Optional[Mapping[str, DType]] = None,
    ) -> dict:
        target_dtypes = {}
        for feature in event.features:
            if to_dtype is not None:
                # cast all features this dtype
                target_dtypes[feature.name] = to_dtype
            elif from_features is not None:
                # Cast by feature name (use same feature.dtype if not found)
                target_dtypes[feature.name] = from_features.get(
                    feature.name, feature.dtype
                )
            elif from_dtypes is not None:
                # Cast by dtype (use same feature.dtype if not found)
                target_dtypes[feature.name] = from_dtypes.get(
                    feature.dtype, feature.dtype
                )
        return target_dtypes

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
    target: Union[DType, Union[Mapping[str, DType], Mapping[DType, DType]]],
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
            the original dtypes (and not both types mixed),
            and the values are the target dtypes for them.
            All dtypes must be temporian types (see `dtype.py`)
        check_overflow:
            A flag to check overflow when casting to a dtype with a shorter
            range (e.g: `INT64`->`INT32`).
            Note that this check adds some computation overhead.
            Defaults to `True`.

    Returns:
        A new `Event` (or the same if no features actually changed dtype), with
        the same feature names as the input one, but with the new dtypes as
        specified in `target`.

    Raises:
        ValueError:
            If `check_overflow=True` and some value is out of the range of the
            target dtype.
        ValueError:
            If trying to cast a non-numeric string to numeric dtype.
        ValueError:
            If the `target` parameter is not a dtype nor a mapping.
        ValueError:
            If `target` is a mapping, but some of the keys are not a dtype nor
            a feature in `event.feature_names`, or if those types are mixed.

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

        3. `cast(event, target={'A': dtype.FLOAT32, 'B': dtype.INT32})`
            Convert 'A' to `FLOAT32` and 'B' to `INT32`.

        4. `cast(event, target={'A': dtype.FLOAT32, dtype.FLOAT64: dtype.INT32})`
            Raises ValueError: don't mix dtype and feature name keys
    """
    # Convert 'target' to one of these:
    to_dtype = None
    from_features = None
    from_dtypes = None

    # Further type verifications are done in the operator
    if isinstance(target, Mapping):
        if all(key in event.feature_names for key in target):
            from_features = target
        elif all(isinstance(key, DType) for key in target):
            from_dtypes = target
        else:
            raise ValueError(
                "Cast: mapping keys should be all DType or feature names"
            )
    else:
        to_dtype = target

    return CastOperator(
        event,
        to_dtype=to_dtype,
        from_features=from_features,
        from_dtypes=from_dtypes,
        check_overflow=check_overflow,
    ).outputs["event"]
