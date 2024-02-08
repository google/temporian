from abc import abstractmethod
from typing import Callable, Dict

import numpy as np

from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseCalendarNumpyImplementation(OperatorImplementation):
    """Interface definition and common logic for numpy implementation of
    calendar operators."""

    def __init__(self, operator: BaseCalendarOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseCalendarOperator)

    def __call__(self, sampling: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, BaseCalendarOperator)
        output_schema = self.output_schema("output")
        implementation = self._implementation()

        # create destination EventSet
        dst_evset = EventSet(data={}, schema=output_schema)
        for index_key, index_data in sampling.data.items():
            output = np.zeros(shape=index_data.timestamps.shape, dtype=np.int32)
            error = implementation(
                index_data.timestamps, self.operator.tz, output
            )
            if len(error) > 0:
                raise ValueError(error)
            dst_evset.set_index_value(
                index_key,
                IndexData(
                    [output], index_data.timestamps, schema=output_schema
                ),
                normalize=False,
            )

        return {"output": dst_evset}

    @abstractmethod
    def _implementation(self) -> Callable:
        raise NotImplementedError()
