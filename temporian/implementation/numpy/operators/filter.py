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
        for condition_index, condition_data in condition.iterindex():
            # get boolean mask from condition
            mask = condition_data.features[0]

            src_event = event[condition_index]

            filtered_timestamps = src_event.timestamps[mask]

            filtered_features = [
                feature_data[mask] for feature_data in src_event.features
            ]

            output_event[condition_index] = IndexData(
                filtered_features, filtered_timestamps
            )

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    FilterOperator, FilterNumpyImplementation
)
