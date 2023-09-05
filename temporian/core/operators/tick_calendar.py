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
        second: Union[int, str],
        minute: Union[int, str],
        hour: Union[int, str],
        day_of_month: Union[int, str],
        month: Union[int, str],
        day_of_week: Union[int, str],
    ):
        super().__init__()

        # Attributes
        self._second = second
        self._minute = minute
        self._hour = hour
        self._day_of_month = day_of_month
        self._month = month
        self._day_of_week = day_of_week
        self.add_attribute("second", second)
        self.add_attribute("minute", minute)
        self.add_attribute("hour", hour)
        self.add_attribute("day_of_month", day_of_month)
        self.add_attribute("month", month)
        self.add_attribute("day_of_week", day_of_week)

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
    def second(self) -> Union[int, str]:
        return self._second

    @property
    def minute(self) -> Union[int, str]:
        return self._minute

    @property
    def hour(self) -> Union[int, str]:
        return self._hour

    @property
    def day_of_month(self) -> Union[int, str]:
        return self._day_of_month

    @property
    def month(self) -> Union[int, str]:
        return self._month

    @property
    def day_of_week(self) -> Union[int, str]:
        return self._day_of_week

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TICK_CALENDAR",
            # TODO: add attributes
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
    # TODO: Logic for auto arguments (None)
    assert second is not None
    assert minute is not None
    assert hour is not None
    assert day_of_month is not None
    assert month is not None
    assert day_of_week is not None
    return TickCalendar(input=input, second=second, minute=minute, hour=hour, day_of_month=day_of_month, month=month, day_of_week=day_of_week).outputs["output"]  # type: ignore
