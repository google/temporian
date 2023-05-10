from typing import Dict

from temporian.core.operators.filter import FilterOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class FilterNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the filter operator."""

    def __init__(self, operator: FilterOperator) -> None:
        super().__init__(operator)

    def __call__(
        self, input: EventSet, condition: EventSet
    ) -> Dict[str, EventSet]:
        output_evset = EventSet(
            {}, input.feature_names, input.index_names, input.is_unix_timestamp
        )
        for condition_index, condition_data in condition.iterindex():
            # get boolean mask from condition
            mask = condition_data.features[0]

            src_event = input[condition_index]

            filtered_timestamps = src_event.timestamps[mask]

            filtered_features = [
                feature_data[mask] for feature_data in src_event.features
            ]

            output_evset[condition_index] = IndexData(
                filtered_features, filtered_timestamps
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    FilterOperator, FilterNumpyImplementation
)
