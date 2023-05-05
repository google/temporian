from abc import abstractmethod
from typing import Dict

import numpy as np

from temporian.core.operators.unary import (
    BaseUnaryOperator,
    AbsOperator,
    InvertOperator,
    NotNanOperator,
    IsNanOperator,
    LogOperator,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseUnaryNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: BaseUnaryOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        dst_evset = EventSet(
            data={},
            feature_names=input.feature_names,
            index_names=input.index_names,
            is_unix_timestamp=input.is_unix_timestamp,
        )
        for index_key, index_data in input.iterindex():
            dst_evset[index_key] = IndexData(
                [
                    self._do_operation(feature)
                    for feature in index_data.features
                ],
                index_data.timestamps,
            )

        return {"output": dst_evset}

    @abstractmethod
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        """
        Actually perform the operation on each feature array
        """


class InvertNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return ~feature


class IsNanNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.isnan(feature)


class NotNanNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return ~np.isnan(feature)


class AbsNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return abs(feature)


class LogNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.log(feature)


implementation_lib.register_operator_implementation(
    AbsOperator, AbsNumpyImplementation
)
implementation_lib.register_operator_implementation(
    InvertOperator, InvertNumpyImplementation
)
implementation_lib.register_operator_implementation(
    IsNanOperator, IsNanNumpyImplementation
)
implementation_lib.register_operator_implementation(
    NotNanOperator, NotNanNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LogOperator, LogNumpyImplementation
)
