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

"""Calendar day of week operator."""

from temporian.core import operator_lib
from temporian.core.data import dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class CalendarDayOfWeekOperator(Operator):
    """
    Calendar operator to obtain the day of the week each timestamp belongs to.
    """

    def __init__(self, sampling: Event):
        super().__init__()

        # input
        self.add_input("sampling", sampling)

        output_feature = Feature(
            name="calendar_day_of_week",
            dtype=dtype.INT32,
            sampling=sampling.sampling(),
            creator=self,
        )

        # output
        self.add_output(
            "event",
            Event(
                features=[output_feature],
                sampling=sampling.sampling(),
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="CALENDAR_DAY_OF_WEEK",
            inputs=[pb.OperatorDef.Input(key="sampling")],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(CalendarDayOfWeekOperator)


def calendar_day_of_week(sampling: Event) -> Event:
    """Obtain the day of the week each of the timestamps in an event belongs to.
    Features in input event are ignored. Output feature contains numbers from
    0 (Monday) to 6 (Sunday).

    Args:
        sampling: the event to get the days of week from.

    Returns:
        event with a single feature corresponding to the day of the week
            each timestamp in `event`'s sampling belongs to, with the same
            sampling as `event`.
    """
    return CalendarDayOfWeekOperator(sampling).outputs()["event"]
