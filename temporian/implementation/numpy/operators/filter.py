from typing import Dict

from temporian.core.operators.filter import FilterOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class FilterNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the filter operator."""

    def __init__(self, operator: FilterOperator) -> None:
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, condition: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        output_event = NumpyEvent(
            {}, event.feature_names, event.index_names, event.is_unix_timestamp
        )
        for index_key, index_data in condition.iterindex():
            # get boolean mask from condition
            index_mask = index_data.features[0]

            # filter timestamps
            filtered_timestamps = event[index_key].timestamps[index_mask]

            # if filtered timestamps is empty, skip
            if len(filtered_timestamps) == 0:
                continue

            # filter features
            filtered_features = [
                event_feature[index_mask]
                for event_feature in event[index_key].features
            ]
            # set filtered data
            output_event[index_key] = IndexData(
                filtered_features, filtered_timestamps
            )

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    FilterOperator, FilterNumpyImplementation
)
