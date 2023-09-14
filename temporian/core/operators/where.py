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
from typing import Union, Any

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
        dtypes = on_true.schema.feature_dtypes()
        if len(dtypes) != 1 or dtypes[0] != DType.BOOLEAN:
            raise ValueError(
                "Input should have only one boolean feature but got"
                f" {len(dtypes)} features ({dtypes=})"
            )
        self.add_input("input", input)

        # Check on_true
        true_dtype = self._add_argument(on_true, "on_true")
        false_dtype = self._add_argument(on_false, "on_false")
        if true_dtype != false_dtype:
            raise ValueError(
                f"Arguments 'on_true' (dtype={true_dtype}) and 'on_false'"
                f" (dtype={false_dtype}) should have the same dtype. Cast one"
                " of the values, check the cast() method if it's an EventSet."
            )

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[(input.schema.feature_names()[0], true_dtype)],
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    def _add_argument(
        self, arg_value: Union[EventSetNode, Any], arg_name: str
    ) -> DType:
        # Argument is another node (input)
        if isinstance(arg_value, EventSetNode):
            dtypes = arg_value.schema.feature_dtypes()
            if len(dtypes) != 1:
                raise ValueError(
                    f"Argument '{arg_name}' should have only 1 feature but got "
                    f" an EventSet with {len(dtypes)} features."
                )
            self.add_input(arg_name, arg_value)
            return dtypes[0]

        # Argument is a single value (attribute)
        dtype = DType.infer_from_value(arg_value)  # check dtype before adding
        self.add_attribute(arg_name, arg_value)
        return dtype

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="WHERE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="param",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


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
