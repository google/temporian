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

"""Base boolean scalar operator."""

from abc import ABC, abstractmethod

from temporian.core.data import dtype as dtype_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseBooleanScalarOperator(Operator, ABC):
    """
    Base boolean scalar operator to implement common logic.
    """

    def __init__(self, event: Event, value: any):
        super().__init__()

        self.add_attribute("value", value)

        # inputs
        self.add_input("event", event)

        # convert value type to temporian dtype
        value_dtype = dtype_lib.python_type_to_temporian_dtype(type(value))

        # ensure all features have same dtype as value
        if not all(
            dtype_lib.same_kind(dtype, value_dtype)
            for dtype in event.dtypes.values()
        ):
            raise ValueError(
                f"All features must have the same dtype ({value_dtype}) as"
                f" value {value}. Current feature dtypes: {event.dtypes}"
            )

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=self.feature_name(f, value),
                dtype=dtype_lib.BOOLEAN,
                sampling=f.sampling,
                creator=self,
            )
            for f in event.features
        ]

        output_sampling = event.sampling
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
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
                    # dtype depends on value dtype, this is not always float64
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Get the key of the operator definition."""

    def feature_name(self, feature: Feature, value: any) -> str:
        """Returns the name of the feature to be created."""
        return f"{feature.name}_{self.operation_name}_{value}"

    @property
    @abstractmethod
    def operation_name(self) -> str:
        """Returns name of the operation to be performed."""
