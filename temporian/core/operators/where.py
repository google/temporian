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


"""Where operator class and public API function definitions."""
from typing import Tuple, Union, Any, Optional

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck
from temporian.core.data.dtype import DType


class Where(Operator):
    def __init__(
        self,
        input: EventSetNode,
        on_true: Union[EventSetNode, Any],
        on_false: Union[EventSetNode, Any],
    ):
        super().__init__()

        # Check input
        dtypes = input.schema.feature_dtypes()
        if len(dtypes) != 1 or dtypes[0] != DType.BOOLEAN:
            raise ValueError(
                "Input should have only 1 boolean feature but got"
                f" {len(dtypes)} features ({dtypes=})"
            )
        self.add_input("input", input)

        # on_true: can be input or attribute
        true_dtype, is_node_t = self._add_argument(on_true, "on_true", input)
        self._on_true_attribute = None if is_node_t else on_true

        # on_false: can be input or attribute
        false_dtype, is_node_f = self._add_argument(on_false, "on_false", input)
        self._on_false_attribute = None if is_node_f else on_false
        output_dtype = self._output_dtypes(
            true_dtype, is_node_t, false_dtype, is_node_f
        )

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[(self.output_feature_name, output_dtype)],
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    def _output_dtypes(
        self,
        true_dtype: DType,
        is_node_t: bool,
        false_dtype: DType,
        is_node_f: bool,
    ) -> DType:
        """Checks if operand dtypes are compatible."""

        if true_dtype == false_dtype:
            return true_dtype

        # The dtypes of the inputs do no match.
        #
        # This situation is only permitted when one of the operands is a
        # scalar integer and the other operand (scalar or event) is a float.

        if not is_node_t and not is_node_f:
            # Two scalars
            raise ValueError(
                "At leasat one of the operants of 'where' should be an"
                " EventSet. Instead, got two scalars."
            )

        if not is_node_t or not is_node_f:
            # An eventset and a scalar
            if is_node_t:
                node_dtype = true_dtype
                attribute_dtype = false_dtype
            else:
                node_dtype = false_dtype
                attribute_dtype = true_dtype

            if node_dtype.is_float and attribute_dtype.is_numerical:
                # Casting of the attribute integer or float dtype to the node
                # float dtype
                return node_dtype

            elif node_dtype.is_integer and attribute_dtype.is_integer:
                # Casting of the attribute integer dtype to the node integer
                # dtype
                return node_dtype

        raise ValueError(
            f"Arguments 'on_true' (dtype={true_dtype}) and 'on_false'"
            f" (dtype={false_dtype}) should have the same dtype or compatible"
            " dtypes. Cast one of the values, check the cast() method if it's"
            " an EventSet."
        )

    def _add_argument(
        self,
        arg_value: Union[EventSetNode, Any],
        arg_name: str,
        input: EventSetNode,
    ) -> Tuple[DType, bool]:
        # Check if argument is a node
        is_node = isinstance(arg_value, EventSetNode)
        if is_node:
            input.check_same_sampling(arg_value)
            dtypes = arg_value.schema.feature_dtypes()
            if len(dtypes) != 1:
                raise ValueError(
                    f"Argument '{arg_name}' should have only 1 feature but got "
                    f" an EventSet with {len(dtypes)} features."
                )
            dtype = dtypes[0]
            self.add_input(arg_name, arg_value)
        else:
            # Argument is single-value attribute
            dtype = DType.from_python_value(arg_value)
            self.add_attribute(arg_name, arg_value)

        return dtype, is_node

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="WHERE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="on_true",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                    is_optional=True,
                ),
                pb.OperatorDef.Attribute(
                    key="on_false",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                    is_optional=True,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="on_true", is_optional=True),
                pb.OperatorDef.Input(key="on_false", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def on_true(self) -> Optional[Any]:
        """Returns None if 'on_true' is EventSet instead of a single value"""
        return self._on_true_attribute

    @property
    def on_false(self) -> Optional[Any]:
        """Returns None if 'on_false' is EventSet instead of a single value"""
        return self._on_false_attribute

    @property
    def output_feature_name(self) -> str:
        return self.inputs["input"].schema.feature_names()[0]


operator_lib.register_operator(Where)


@typecheck
@compile
def where(
    input: EventSetOrNode,
    on_true: Union[EventSetOrNode, Any],
    on_false: Union[EventSetOrNode, Any],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return Where(input=input, on_true=on_true, on_false=on_false).outputs[
        "output"
    ]
