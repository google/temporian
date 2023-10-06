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

"""Map operator class and public API function definitions."""

from inspect import signature
from typing import Any, Callable, Dict, Optional, Union
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import Operator
from temporian.core.types import MapExtras
from temporian.core.typing import EventSetOrNode, TargetDtypes, Scalar
from temporian.implementation.numpy.data.dtype_normalization import (
    build_dtypes_list_from_target_dtypes,
    normalize_target_dtypes,
)
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


MapFunction = Union[Callable[[Any], Scalar], Callable[[Any, MapExtras], Scalar]]
"""A function that maps an [`EventSet`][temporian.EventSet]'s value to another
value.

The function must receive the original value and optionally a
[`MapExtras`][temporian.types.MapExtras] object, which includes additional
information about the value's position in the EventSet, and return the new
value.
"""


class Map(Operator):
    def __init__(
        self,
        input: EventSetNode,
        func: MapFunction,
        dtype: Optional[DType] = None,
        dtype_to_dtype: Optional[Dict[DType, DType]] = None,
        feature_name_to_dtype: Optional[Dict[str, DType]] = None,
    ):
        """Constructor.

        There can only be one of dtype, dtype_to_dtype or feature_name_to_dtype.

        Args:
            input: Input node.
            func: Function to apply to each elemnent.
            dtype: All the output features are expected to be of this type.
            dtype_to_dtype: Mapping between current dtype and new dtype.
            feature_name_to_dtype: Mapping between feature name and new dtype.
        """
        super().__init__()

        # at most one argument is set.
        assert (
            sum(
                x is not None
                for x in [dtype, dtype_to_dtype, feature_name_to_dtype]
            )
            <= 1
        )

        # output_dtypes is an array of output dtype for all the input features,
        # in the same order
        output_dtypes = build_dtypes_list_from_target_dtypes(
            input, dtype, dtype_to_dtype, feature_name_to_dtype
        )
        assert len(output_dtypes) == len(input.schema.features)

        if len(signature(func).parameters) > 2:
            raise ValueError("`func` must receive at most 2 arguments.")

        self.add_attribute("func", func)
        self._func = func

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[
                    FeatureSchema(f.name, dtype)
                    for f, dtype in zip(input.schema.features, output_dtypes)
                ],
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    @property
    def func(self) -> MapFunction:
        return self._func

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="MAP",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="func",
                    type=pb.OperatorDef.Attribute.Type.CALLABLE,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Map)


@typecheck
@compile
def map(
    input: EventSetOrNode,
    func: MapFunction,
    output_dtypes: Optional[TargetDtypes],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    dtype = feature_name_to_dtype = dtype_to_dtype = None
    if output_dtypes is not None:
        dtype, feature_name_to_dtype, dtype_to_dtype = normalize_target_dtypes(
            input, output_dtypes
        )

    return Map(
        input=input,
        func=func,
        dtype=dtype,
        feature_name_to_dtype=feature_name_to_dtype,
        dtype_to_dtype=dtype_to_dtype,
    ).outputs["output"]
