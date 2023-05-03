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

"""Base arithmetic scalar operator class definition."""

from typing import Union, List
from abc import abstractmethod

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseArithmeticScalarOperator(Operator):
    """Interface definition and common logic for arithmetic scalar operators."""

    def __init__(
        self,
        input: Node,
        value: Union[float, int, str, bool],
        is_value_first: bool = False,  # useful for non-commutative operators
    ):
        super().__init__()

        self.value = value
        self.is_value_first = is_value_first

        # inputs
        self.add_input("input", input)

        self.add_attribute("value", value)
        self.add_attribute("is_value_first", is_value_first)

        if not isinstance(input, Node):
            raise TypeError(f"Node must be of type Node but got {type(input)}")

        # check that every dtype of input feature is equal to value dtype
        value_dtype = DType.from_python_type(type(value))

        # check that value dtype is in self.dtypes_to_check
        if value_dtype not in self.supported_value_dtypes:
            raise ValueError(
                "Expected value DType to be one of"
                f" {self.supported_value_dtypes}, but got {value_dtype}"
            )

        # TODO: Check if we want to compare kind of dtype or just dtype
        # it makes sense to allow subtypes of the same kind. Value will
        # always be 64 bits. We need a way to allow 32 bits.
        if not self.ignore_value_dtype_checking:
            for feature in input.features:
                if feature.dtype != value_dtype:
                    raise ValueError(
                        f"Feature {feature.name} has dtype {feature.dtype} "
                        f"but value has dtype {value_dtype}. Both must be "
                        "equal."
                    )

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=self.output_feature_name(feature.name),
                dtype=self.output_feature_dtype(feature),
                sampling=input.sampling,
                creator=self,
            )
            for feature in input.features
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key,
            attributes=[
                pb.OperatorDef.Attribute(
                    key="value",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                    is_optional=False,
                ),
                pb.OperatorDef.Attribute(
                    key="is_value_first",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Gets the prefix to use for the output features."""

    @property
    @abstractmethod
    def supported_value_dtypes(self) -> List[DType]:
        """Supported DTypes for value."""

    def output_feature_name(self, feature_name: str) -> str:
        return f"{self.prefix}_{feature_name}_{self.value}"

    def output_feature_dtype(self, feature: Feature) -> DType:
        return feature.dtype

    @property
    def ignore_value_dtype_checking(self) -> bool:
        """Returns True if we want to ignore the value dtype checking."""
        return False
