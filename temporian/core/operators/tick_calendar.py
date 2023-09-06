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


"""TickCalendar operator class and public API function definitions."""
from typing import Union

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class TickCalendar(Operator):
    def __init__(
        self,
        input: EventSetNode,
        min_second: int,
        min_minute: int,
        min_hour: int,
        min_day_of_month: int,
        min_month: int,
        min_day_of_week: int,
        max_second: int,
        max_minute: int,
        max_hour: int,
        max_day_of_month: int,
        max_month: int,
        max_day_of_week: int,
    ):
        super().__init__()

        # Attributes
        self._min_second = min_second
        self._max_second = max_second
        self._min_minute = min_minute
        self._max_minute = max_minute
        self._min_hour = min_hour
        self._max_hour = max_hour
        self._min_day_of_month = min_day_of_month
        self._max_day_of_month = max_day_of_month
        self._min_month = min_month
        self._max_month = max_month
        self._min_day_of_week = min_day_of_week
        self._max_day_of_week = max_day_of_week
        self.add_attribute("min_second", min_second)
        self.add_attribute("max_second", max_second)
        self.add_attribute("min_minute", min_minute)
        self.add_attribute("max_minute", max_minute)
        self.add_attribute("min_hour", min_hour)
        self.add_attribute("max_hour", max_hour)
        self.add_attribute("min_day_of_month", min_day_of_month)
        self.add_attribute("max_day_of_month", max_day_of_month)
        self.add_attribute("min_month", min_month)
        self.add_attribute("max_month", max_month)
        self.add_attribute("min_day_of_week", min_day_of_week)
        self.add_attribute("max_day_of_week", max_day_of_week)

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @property
    def min_second(self) -> int:
        return self._min_second

    @property
    def max_second(self) -> int:
        return self._max_second

    @property
    def min_minute(self) -> int:
        return self._min_minute

    @property
    def max_minute(self) -> int:
        return self._max_minute

    @property
    def min_hour(self) -> int:
        return self._min_hour

    @property
    def max_hour(self) -> int:
        return self._max_hour

    @property
    def min_day_of_month(self) -> int:
        return self._min_day_of_month

    @property
    def max_day_of_month(self) -> int:
        return self._max_day_of_month

    @property
    def min_month(self) -> int:
        return self._min_month

    @property
    def max_month(self) -> int:
        return self._max_month

    @property
    def min_day_of_week(self) -> int:
        return self._min_day_of_week

    @property
    def max_day_of_week(self) -> int:
        return self._max_day_of_week

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TICK_CALENDAR",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="min_second",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_second",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="min_minute",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_minute",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="min_hour",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_hour",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="min_day_of_month",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_day_of_month",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="min_month",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_month",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="min_day_of_week",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_day_of_week",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(TickCalendar)


@typecheck
@compile
def tick_calendar(
    input: EventSetOrNode,
    second: Union[int, str, None],
    minute: Union[int, str, None],
    hour: Union[int, str, None],
    day_of_month: Union[int, str, None],
    month: Union[int, str, None],
    day_of_week: Union[int, str, None],
) -> EventSetOrNode:
    args = [second, minute, hour, day_of_month, month, day_of_week]

    # Default for empty args
    if all(arg is None for arg in args):
        day_of_month = "*"
        month = "*"

    if second == "*":
        min_second = 0
        max_second = 59
    else:
        min_second = max_second = 0 if second is None else int(second)

    if minute == "*":
        min_minute = 0
        max_minute = 59
    elif minute is not None:
        min_minute = max_minute = int(minute)
    else:  # None (auto set): only if adjacent values are specified
        raise ValueError()  # TODO

    # TODO
    min_hour = 0
    max_hour = 23
    min_day_of_month = 1
    max_day_of_month = 31
    min_month = 1
    max_month = 12
    min_day_of_week = 0
    max_day_of_week = 6

    return TickCalendar(
        input=input,
        min_second=min_second,
        max_second=max_second,
        min_minute=min_minute,
        max_minute=max_minute,
        min_hour=min_hour,
        max_hour=max_hour,
        min_day_of_month=min_day_of_month,
        max_day_of_month=max_day_of_month,
        min_month=min_month,
        max_month=max_month,
        min_day_of_week=min_day_of_week,
        max_day_of_week=max_day_of_week,
    ).outputs[
        "output"
    ]  # type: ignore
