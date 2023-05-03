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
        for index_key, index_data in condition.iterindex():
            # get boolean mask from condition
            index_mask = index_data.features[0]

            # filter timestamps
            filtered_timestamps = input[index_key].timestamps[index_mask]

            # if filtered timestamps is empty, skip
            if len(filtered_timestamps) == 0:
                continue

            # filter features
            filtered_features = [
                evset_feature[index_mask]
                for evset_feature in input[index_key].features
            ]
            # set filtered data
            output_evset[index_key] = IndexData(
                filtered_features, filtered_timestamps
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    FilterOperator, FilterNumpyImplementation
)
