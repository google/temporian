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
    RoundOperator,
    SinOperator,
    CosOperator,
    TanOperator,
    ArcSinOperator,
    ArcCosOperator,
    ArcTanOperator,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseUnaryNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: BaseUnaryOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, BaseUnaryOperator)

        output_schema = self.output_schema("output")
        dst_evset = EventSet(
            data={},
            schema=output_schema,
        )
        for index_key, index_data in input.data.items():
            dst_evset.set_index_value(
                index_key,
                IndexData(
                    [
                        self._do_operation(feature)
                        for feature in index_data.features
                    ],
                    index_data.timestamps,
                    schema=output_schema,
                ),
                normalize=False,
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



class RoundNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.round(feature)

class SinNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.sin(feature)


class CosNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.cos(feature)


class TanNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.tan(feature)


class ArcSinNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.arcsin(feature)


class ArcCosNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.arccos(feature)


class ArcTanNumpyImplementation(BaseUnaryNumpyImplementation):
    def _do_operation(self, feature: np.ndarray) -> np.ndarray:
        return np.arctan(feature)


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
implementation_lib.register_operator_implementation(

    RoundOperator, RoundNumpyImplementation

    SinOperator, SinNumpyImplementation
)
implementation_lib.register_operator_implementation(
    CosOperator, CosNumpyImplementation
)
implementation_lib.register_operator_implementation(
    TanOperator, TanNumpyImplementation
)
implementation_lib.register_operator_implementation(
    ArcSinOperator, ArcSinNumpyImplementation
)
implementation_lib.register_operator_implementation(
    ArcCosOperator, ArcCosNumpyImplementation
)
implementation_lib.register_operator_implementation(
    ArcTanOperator, ArcTanNumpyImplementation
)
