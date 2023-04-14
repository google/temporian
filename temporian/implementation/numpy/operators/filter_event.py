from typing import Dict

from temporian.core.operators.filter import FilterOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class FilterNumpyImplementation(OperatorImplementation):
    """Filter timestamps from an event based on a condition."""

    def __init__(self, operator: FilterOperator) -> None:
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, condition: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        new_sampling = {}
        event_filtered_data = {}

        for index, features in condition.data.items():
            # get boolean mask from condition
            condition_feature = features[0].data

            # filtered sampling
            new_sampling[index] = event.sampling.data[index][condition_feature]

            # filter features
            event_filtered_data[index] = [
                NumpyFeature(
                    name=self.operator.feature_name(event_feature),
                    data=event_feature.data[condition_feature],
                )
                for event_feature in event.data[index]
            ]

        output_event = NumpyEvent(
            data=event_filtered_data,
            sampling=NumpySampling(
                index=event.sampling.index, data=new_sampling
            ),
        )

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    FilterOperator, FilterNumpyImplementation
)
