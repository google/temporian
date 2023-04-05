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
import copy

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core.operators.glue import GlueOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class GlueNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: GlueOperator):
        super().__init__(operator)
        assert isinstance(operator, GlueOperator)

    def __call__(
        self,
        **dict_events: Dict[str, NumpyEvent],
    ) -> Dict[str, NumpyEvent]:
        """Glues a dictionary of NumpyEvents.

        The output features are ordered by the argument name of the event.

        Example:
            If dict_events is {"event_0": X, "event_3": Z, "event_2": Y}
            output is guarenteed to be [X, Y, Z].
        """
        events = list(list(zip(*sorted(list(dict_events.items()))))[1])
        assert len(events) >= 2

        dst_event = NumpyEvent(data={}, sampling=events[0].sampling)

        for index in events[0].data.keys():
            dst_features = []
            for event in events:
                dst_features.extend(copy.copy(event.data[index]))
            dst_event.data[index] = dst_features

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    GlueOperator, GlueNumpyImplementation
)
