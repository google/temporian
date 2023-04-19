from typing import Dict

from temporian.core.data.duration import duration_abbreviation
from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class LagNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: LagOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        duration = self.operator.attributes["duration"]
        prefix = "lag" if duration > 0 else "leak"
        duration_str = duration_abbreviation(duration)
        output_event = NumpyEvent(
            {},
            feature_names=[
                f"{prefix}[{duration_str}]_{feature_name}"
                for feature_name in event.feature_names
            ],
            index_names=event.index_names,
            is_unix_timestamp=event.is_unix_timestamp,
        )
        for index_key, index_data in event.iterindex():
            output_event.data[index_key] = IndexData(
                index_data.features, index_data.timestamps.copy() + duration
            )

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    LagOperator, LagNumpyImplementation
)
