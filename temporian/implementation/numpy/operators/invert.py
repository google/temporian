from typing import Dict

from temporian.core.operators.invert import InvertOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class InvertNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: InvertOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        """Applies the invert operation

        Args:
            input: An EventSet with BOOLEAN features to invert.

        Returns:
            not input (~input)
        """
        dst_evset = EventSet(
            data={},
            feature_names=input.feature_names,
            index_names=input.index_names,
            is_unix_timestamp=input.is_unix_timestamp,
        )
        for index_key, index_data in input.iterindex():
            dst_evset[index_key] = IndexData(
                [~feature for feature in index_data.features],
                index_data.timestamps,
            )

        return {"output": dst_evset}


implementation_lib.register_operator_implementation(
    InvertOperator, InvertNumpyImplementation
)
