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

from typing import Dict, Optional, List
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
from temporian.core.typing import EventSetOrNode, TargetDtypes
from temporian.implementation.numpy.data.dtype_normalization import (
    build_dtypes_list_from_target_dtypes,
    normalize_target_dtypes,
)
from temporian.proto import core_pb2 as pb


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
            dtype_to_dtype: Mapping between current dtype and new dtype.
            feature_name_to_dtype: Mapping between feature name and new dtype.
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

        # "self._dtypes" is an array of target dtype for all the input features
        # (in the same order as the input features).
        if dtypes is None:
            dtypes = build_dtypes_list_from_target_dtypes(
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
    target: TargetDtypes,
    check_overflow: bool = True,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    assert isinstance(input, EventSetNode)

    dtype, feature_name_to_dtype, dtype_to_dtype = normalize_target_dtypes(
        input, target
    )

    return CastOperator(
        input,
        dtype=dtype,
        feature_name_to_dtype=feature_name_to_dtype,
        dtype_to_dtype=dtype_to_dtype,
        check_overflow=check_overflow,
    ).outputs["output"]
