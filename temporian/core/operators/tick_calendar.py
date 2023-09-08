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
        min_mday: int,
        min_month: int,
        min_wday: int,
        max_second: int,
        max_minute: int,
        max_hour: int,
        max_mday: int,
        max_month: int,
        max_wday: int,
    ):
        super().__init__()
        if not input.schema.is_unix_timestamp:
            raise ValueError(
                "Can only use tick_calendar on unix timestamp samplings"
            )

        # Attributes
        self._min_second = min_second
        self._max_second = max_second
        self._min_minute = min_minute
        self._max_minute = max_minute
        self._min_hour = min_hour
        self._max_hour = max_hour
        self._min_mday = min_mday
        self._max_mday = max_mday
        self._min_month = min_month
        self._max_month = max_month
        self._min_wday = min_wday
        self._max_wday = max_wday
        self.add_attribute("min_second", min_second)
        self.add_attribute("max_second", max_second)
        self.add_attribute("min_minute", min_minute)
        self.add_attribute("max_minute", max_minute)
        self.add_attribute("min_hour", min_hour)
        self.add_attribute("max_hour", max_hour)
        self.add_attribute("min_mday", min_mday)
        self.add_attribute("max_mday", max_mday)
        self.add_attribute("min_month", min_month)
        self.add_attribute("max_month", max_month)
        self.add_attribute("min_wday", min_wday)
        self.add_attribute("max_wday", max_wday)

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
    def min_mday(self) -> int:
        return self._min_mday

    @property
    def max_mday(self) -> int:
        return self._max_mday

    @property
    def min_month(self) -> int:
        return self._min_month

    @property
    def max_month(self) -> int:
        return self._max_month

    @property
    def min_wday(self) -> int:
        return self._min_wday

    @property
    def max_wday(self) -> int:
        return self._max_wday

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
                    key="min_mday",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_mday",
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
                    key="min_wday",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="max_wday",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(TickCalendar)


def set_arg_range(arg_value, val_range, prefer_free):
    if arg_value == "*":
        range_ini, range_end = val_range
    elif arg_value is not None:
        range_ini = range_end = int(arg_value)
    else:  # None (auto setup)
        if prefer_free:  # Don't restrict the range
            range_ini, range_end = val_range
        else:  # Fix to first value
            range_ini = range_end = val_range[0]

    return range_ini, range_end


@typecheck
@compile
def tick_calendar(
    input: EventSetOrNode,
    second: Union[int, str, None],
    minute: Union[int, str, None],
    hour: Union[int, str, None],
    mday: Union[int, str, None],
    month: Union[int, str, None],
    wday: Union[int, str, None],
) -> EventSetOrNode:
    # Default for empty args
    if all(arg is None for arg in (second, minute, hour, mday, month, wday)):
        mday = "*"
        month = "*"

    # All defined values must be consecutive (no gaps with None)
    if wday is not None:
        sorted_args = [second, minute, hour, wday]
    else:
        sorted_args = [second, minute, hour, mday, month]
    for idx, arg in enumerate(sorted_args):
        if (
            arg is None
            and any(a is not None for a in sorted_args[:idx])
            and any(a is not None for a in sorted_args[idx + 1 :])
        ):
            raise ValueError(
                "Can't set argument to None because previous and"
                " following arguments were specified. Set to '*' or an"
                " integer value instead"
            )

    prefer_free = False
    min_second, max_second = set_arg_range(second, (0, 59), prefer_free)

    # prefer_free becomes True when next None args should be set to '*'
    # e.g: only hour=1 -> second=0,minute=0, mday='*', month='*'
    prefer_free = second is not None
    min_minute, max_minute = set_arg_range(minute, (0, 59), prefer_free)

    prefer_free = prefer_free or minute is not None
    min_hour, max_hour = set_arg_range(hour, (0, 23), prefer_free)

    prefer_free = prefer_free or hour is not None
    min_mday, max_mday = set_arg_range(mday, (1, 31), prefer_free)

    prefer_free = prefer_free or mday is not None
    min_month, max_month = set_arg_range(month, (1, 12), prefer_free)

    prefer_free = True  # Always free wday by default
    min_wday, max_wday = set_arg_range(wday, (0, 6), True)

    return TickCalendar(
        input=input,
        min_second=min_second,
        max_second=max_second,
        min_minute=min_minute,
        max_minute=max_minute,
        min_hour=min_hour,
        max_hour=max_hour,
        min_mday=min_mday,
        max_mday=max_mday,
        min_month=min_month,
        max_month=max_month,
        min_wday=min_wday,
        max_wday=max_wday,
    ).outputs[
        "output"
    ]  # type: ignore
