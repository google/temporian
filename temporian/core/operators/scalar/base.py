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

"""Base scalar operator class definition."""

from typing import Union, List

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseScalarOperator(Operator):
    """Interface definition and common code for scalar operators."""

    DEF_KEY = ""

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
            raise TypeError(f"Input must be of type Node but got {type(input)}")

        # check that every dtype of input feature is equal to value dtype
        value_dtype = DType.from_python_type(type(value))

        # check that value dtype is in self.dtypes_to_check
        if value_dtype not in self.supported_value_dtypes:
            raise ValueError(
                "Expected value DType to be one of"
                f" {self.supported_value_dtypes}, but got {value_dtype}"
            )

        # Check that the feature dtype doesn't need an upcast to operate with
        # this value type
        self.map_vtype_dtype = {
            float: [DType.FLOAT32, DType.FLOAT64],
            int: [DType.INT32, DType.INT64, DType.FLOAT32, DType.FLOAT64],
            str: [DType.STRING],
            bool: [
                DType.BOOLEAN,
                DType.INT32,
                DType.INT64,
                DType.FLOAT32,
                DType.FLOAT64,
            ],
        }
        if not self.ignore_value_dtype_checking:
            for feature in input.schema.features:
                if feature.dtype not in self.map_vtype_dtype[type(value)]:
                    raise ValueError(
                        f"Scalar has {type(value)=}, which can only operate"
                        f" with dtypes: {self.map_vtype_dtype[type(value)]}. "
                        f"But {feature.name} has dtype {feature.dtype}."
                    )

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            FeatureSchema(
                name=feature.name,
                dtype=self.output_feature_dtype(feature),
            )
            for feature in input.schema.features
        ]

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=output_features,
                sampling_node=input,
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
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""
        return cls.DEF_KEY

    @property
    def supported_value_dtypes(self) -> List[DType]:
        """Supported DTypes for value."""
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]

    def output_feature_dtype(self, feature: FeatureSchema) -> DType:
        return feature.dtype

    @property
    def ignore_value_dtype_checking(self) -> bool:
        """Returns True if we want to ignore the value dtype checking."""
        return False
