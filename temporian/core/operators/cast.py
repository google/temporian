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

"""Cast operator class and public API function definition."""

from typing import Union, Dict, Optional, List, Any
from temporian.core.data.schema import Schema, FeatureSchema


from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import (
    EventSetNode,
    Feature,
    create_node_with_new_reference,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode, TypeOrDType
from temporian.proto import core_pb2 as pb


def _normalize_dtype(x: Any) -> DType:
    if isinstance(x, DType):
        return x
    if x == int:
        return DType.INT64
    if x == float:
        return DType.FLOAT64
    if x == str:
        return DType.STRING
    if x == bool:
        return DType.BOOLEAN
    raise ValueError(f"Cannot normalize {x!r} as a DType.")


class CastOperator(Operator):
    def __init__(
        self,
        input: EventSetNode,
        check_overflow: bool,
        dtype: Optional[DType] = None,
        dtype_to_dtype: Optional[Dict[DType, DType]] = None,
        feature_name_to_dtype: Optional[Dict[str, DType]] = None,
        dtypes: Optional[List[DType]] = None,
    ):
        """Constructor.

        There can only be one of dtype, dtype_to_dtype, feature_name_to_dtype,
        or dtypes.

        Args:
            input: Input node.
            check_overflow: Check for casting overflow.
            dtype: All the input features are casted to dtype.
            dtype_to_dtype: Mapping between feature name and new dtype.
                feature_name_to_dtype: Mapping between current dtype and new
                dtype.
            dtypes: Dtype for each of the input feature (indexed by feature
                idx).
        """

        super().__init__()

        # Exactly one argument is set.
        assert (
            sum(
                x is not None
                for x in [dtype, dtype_to_dtype, feature_name_to_dtype, dtypes]
            )
            == 1
        )

        # "_new_dtypes" is an array of target dtype for all the input features
        # (in the same order as the input features).
        if dtypes is None:
            dtypes = self._build_dtypes(
                input, dtype, dtype_to_dtype, feature_name_to_dtype
            )

        self._dtypes = dtypes
        self._check_overflow = check_overflow
        assert len(self._dtypes) == len(input.schema.features)

        # Attributes
        self.add_attribute("dtypes", self._dtypes)
        self.add_attribute("check_overflow", check_overflow)

        # Inputs
        self.add_input("input", input)

        # Output node features
        output_features = []
        output_schema = Schema(
            features=[],
            indexes=input.schema.indexes,
            is_unix_timestamp=input.schema.is_unix_timestamp,
        )
        is_noop = True
        for new_dtype, feature_node, feature_schema in zip(
            self._dtypes, input.feature_nodes, input.schema.features
        ):
            output_schema.features.append(
                FeatureSchema(feature_schema.name, new_dtype)
            )
            if new_dtype is feature_schema.dtype:
                # Reuse feature
                output_features.append(feature_node)
            else:
                # Create new feature
                is_noop = False
                output_features.append(Feature(creator=self))

        if is_noop:
            # Nothing to cast
            self.add_output("output", input)
        else:
            # Some of the features are new, some of the features are re-used.
            self.add_output(
                "output",
                create_node_with_new_reference(
                    schema=output_schema,
                    features=output_features,
                    sampling=input.sampling_node,
                    creator=self,
                ),
            )

        # Used in implementation
        self._is_noop = is_noop

        self.check()

    @property
    def check_overflow(self) -> bool:
        return self._check_overflow

    @property
    def dtypes(self) -> List[DType]:
        return self._dtypes

    @property
    def is_noop(self) -> bool:
        return self._is_noop

    def _build_dtypes(
        self,
        input: EventSetNode,
        dtype: Optional[DType] = None,
        dtype_to_dtype: Optional[Dict[DType, DType]] = None,
        feature_name_to_dtype: Optional[Dict[str, DType]] = None,
    ) -> List[DType]:
        if dtype is not None:
            return [dtype] * len(input.schema.features)

        if feature_name_to_dtype is not None:
            return [
                feature_name_to_dtype.get(f.name, f.dtype)
                for f in input.schema.features
            ]

        if dtype_to_dtype is not None:
            return [
                dtype_to_dtype.get(f.dtype, f.dtype)
                for f in input.schema.features
            ]

        assert False

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="CAST",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="dtypes",
                    type=pb.OperatorDef.Attribute.Type.LIST_DTYPE,
                ),
                pb.OperatorDef.Attribute(
                    key="check_overflow",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(CastOperator)


@compile
def cast(
    input: EventSetOrNode,
    target: Union[
        TypeOrDType,
        Dict[str, TypeOrDType],
        Dict[TypeOrDType, TypeOrDType],
    ],
    check_overflow: bool = True,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    # Convert 'target' to one of these:
    dtype: Optional[DType] = None
    feature_name_to_dtype: Optional[Dict[str, DType]] = None
    dtype_to_dtype: Optional[Dict[DType, DType]] = None

    # Further type verifications are done in the operator
    if isinstance(target, dict):
        keys_are_strs = all(isinstance(v, str) for v in target.keys())
        keys_are_dtypes = all(
            isinstance(v, (DType, type)) for v in target.keys()
        )
        values_are_dtypes = all(
            isinstance(v, (DType, type)) for v in target.values()
        )

        if keys_are_strs and values_are_dtypes:
            feature_name_to_dtype = {
                key: _normalize_dtype(value) for key, value in target.items()
            }

            input_feature_names = input.schema.feature_names()
            for feature_name in feature_name_to_dtype.keys():
                if feature_name not in input_feature_names:
                    raise ValueError(f"Unknown feature {feature_name!r}")

        elif keys_are_dtypes and values_are_dtypes:
            dtype_to_dtype = {
                _normalize_dtype(key): _normalize_dtype(value)
                for key, value in target.items()
            }
    elif isinstance(target, DType) or target in [float, int, str, bool]:
        dtype = _normalize_dtype(target)

    if (
        dtype is None
        and feature_name_to_dtype is None
        and dtype_to_dtype is None
    ):
        raise ValueError(
            "`target` should be one of the following: (1) a Temporian dtype"
            " e.g. tp.float64, (2) a dictionary of feature name (str) to"
            " temporian dtype, or (3) a dictionary of temporian dtype to"
            " temporian dtype. Alternatively, Temporian dtypes can be replaced"
            " with python type. For example cast(..., target=float) is"
            " equivalent to cast(..., target=tp.float64).\nInstead got,"
            f" `target` = {target!r}."
        )

    assert isinstance(input, EventSetNode)

    return CastOperator(
        input,
        dtype=dtype,
        feature_name_to_dtype=feature_name_to_dtype,
        dtype_to_dtype=dtype_to_dtype,
        check_overflow=check_overflow,
    ).outputs["output"]
