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

from typing import Dict

from temporian.core.operators.calendar.iso_week import CalendarISOWeekOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc


class CalendarISOWeekNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the calendar_iso_week operator."""

    def __init__(self, operator: CalendarISOWeekOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, CalendarISOWeekOperator)

    def __call__(self, sampling: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, CalendarISOWeekOperator)
        output_schema = self.output_schema("output")

        # create destination EventSet
        dst_evset = EventSet(data={}, schema=output_schema)
        for index_key, index_data in sampling.data.items():
            value = operators_cc.calendar_isoweek(
                index_data.timestamps, self.operator.tz
            )

            dst_evset.set_index_value(
                index_key,
                IndexData([value], index_data.timestamps, schema=output_schema),
                normalize=False,
            )

        return {"output": dst_evset}


implementation_lib.register_operator_implementation(
    CalendarISOWeekOperator, CalendarISOWeekNumpyImplementation
)
