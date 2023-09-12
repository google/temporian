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
from typing import Union, Literal, Tuple

import numpy as np

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

TypeWildCard = Literal["*"]


class TickCalendar(Operator):
    def __init__(
        self,
        input: EventSetNode,
        second: Union[int, TypeWildCard],
        minute: Union[int, TypeWildCard],
        hour: Union[int, TypeWildCard],
        mday: Union[int, TypeWildCard],
        month: Union[int, TypeWildCard],
        wday: Union[int, TypeWildCard],
    ):
        super().__init__()
        if not input.schema.is_unix_timestamp:
            raise ValueError(
                "Can only use tick_calendar on unix timestamp samplings"
            )

        # Attributes
        self._second = self._check_arg(second, self.seconds_max_range())
        self._minute = self._check_arg(minute, self.minutes_max_range())
        self._hour = self._check_arg(hour, self.hours_max_range())
        self._mday = self._check_arg(mday, self.mday_max_range())
        self._month = self._check_arg(month, self.month_max_range())
        self._wday = self._check_arg(wday, self.wday_max_range())
        self.add_attribute("second", second)
        self.add_attribute("minute", minute)
        self.add_attribute("hour", hour)
        self.add_attribute("mday", mday)
        self.add_attribute("month", month)
        self.add_attribute("wday", wday)

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[],
                indexes=input.schema.indexes,
                is_unix_timestamp=True,
                creator=self,
            ),
        )

        self.check()

    def _check_arg(self, arg_value, val_range):
        if arg_value == "*" or (
            isinstance(arg_value, (int, np.integer))
            and arg_value >= val_range[0]
            and arg_value <= val_range[1]
        ):
            return arg_value
        raise ValueError(
            f"Value should be '*' or integer in range {val_range}, got:"
            f" {arg_value} (type {type(arg_value)})"
        )

    @property
    def second(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._second == "*" or not isinstance(self._second, str)
        return self._second

    @property
    def minute(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._minute == "*" or not isinstance(self._minute, str)
        return self._minute

    @property
    def hour(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._hour == "*" or not isinstance(self._hour, str)
        return self._hour

    @property
    def mday(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._mday == "*" or not isinstance(self._mday, str)
        return self._mday

    @property
    def month(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._month == "*" or not isinstance(self._month, str)
        return self._month

    @property
    def wday(self) -> Union[int, TypeWildCard]:
        # assert for typecheck
        assert self._wday == "*" or not isinstance(self._wday, str)
        return self._wday

    @classmethod
    def seconds_max_range(cls) -> Tuple[int, int]:
        return (0, 59)

    @classmethod
    def minutes_max_range(cls) -> Tuple[int, int]:
        return (0, 59)

    @classmethod
    def hours_max_range(cls) -> Tuple[int, int]:
        return (0, 23)

    @classmethod
    def mday_max_range(cls) -> Tuple[int, int]:
        return (1, 31)

    @classmethod
    def month_max_range(cls) -> Tuple[int, int]:
        return (1, 12)

    @classmethod
    def wday_max_range(cls) -> Tuple[int, int]:
        return (0, 6)

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TICK_CALENDAR",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="second",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                ),
                pb.OperatorDef.Attribute(
                    key="minute",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                ),
                pb.OperatorDef.Attribute(
                    key="hour",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                ),
                pb.OperatorDef.Attribute(
                    key="mday",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                ),
                pb.OperatorDef.Attribute(
                    key="month",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                ),
                pb.OperatorDef.Attribute(
                    key="wday",
                    type=pb.OperatorDef.Attribute.Type.ANY,
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
    second: Union[int, TypeWildCard, None] = None,
    minute: Union[int, TypeWildCard, None] = None,
    hour: Union[int, TypeWildCard, None] = None,
    mday: Union[int, TypeWildCard, None] = None,
    month: Union[int, TypeWildCard, None] = None,
    wday: Union[int, TypeWildCard, None] = None,
) -> EventSetOrNode:
    # Don't allow empty args
    if all(arg is None for arg in (second, minute, hour, mday, month, wday)):
        raise ValueError("At least one argument must be provided (not None).")

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

    # prefer_free becomes True when next args should be set to '*' by default
    # e.g: user sets only hour=1 -> second=0,minute=0, mday='*', month='*'
    release_ranges = False

    # Always set second=0 by default
    if second is None:
        second = 0
    else:
        release_ranges = True  # fixed seconds, free minute, hour

    if minute is None:
        minute = "*" if release_ranges else 0
    else:
        release_ranges = True  # fixed minutes, free hour, day, month

    if hour is None:
        hour = "*" if release_ranges else 0
    else:
        release_ranges = True

    if mday is None:
        # If wday is specified, always leave mday free by default
        free_mday = release_ranges or wday is not None
        mday = "*" if free_mday else 1

    # Always free range by default
    month = "*" if month is None else month
    wday = "*" if wday is None else wday

    return TickCalendar(
        input=input,  # type: ignore
        second=second,
        minute=minute,
        hour=hour,
        mday=mday,
        month=month,
        wday=wday,
    ).outputs["output"]
