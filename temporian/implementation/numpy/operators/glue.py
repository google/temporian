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

"""Implementation for the Glue operator."""

from typing import Dict

from temporian.core.operators.glue import GlueOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class GlueNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the glue operator."""

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
        # convert input event dict to list of events
        events = list(list(zip(*sorted(list(dict_events.items()))))[1])
        if len(events) < 2:
            raise ValueError(
                f"Glue operator cannot be called on a {len(events)} events"
            )

        # create output event
        dst_event = NumpyEvent(
            data={},
            feature_names=[
                feature_name
                for event in events
                for feature_name in event.feature_names
            ],
            index_names=events[0].index_names,
            is_unix_timestamp=events[0].is_unix_timestamp,
        )
        # fill output event data
        for index_key, index_data in events[0].iterindex():
            dst_event[index_key] = index_data
            for event in events[1:]:
                dst_event[index_key].features.extend(event[index_key].features)

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    GlueOperator, GlueNumpyImplementation
)
