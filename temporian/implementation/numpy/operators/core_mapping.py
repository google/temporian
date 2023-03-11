from temporian.implementation.numpy.operators.assign import NumpyAssignOperator
from temporian.implementation.numpy.operators.select import NumpySelectOperator
from temporian.implementation.numpy.operators.sum import NumpySumOperator
from temporian.implementation.numpy.operators.simple_moving_average import (
    SimpleMovingAverageNumpyImplementation as SmaImp,
)
from temporian.core.operators.simple_moving_average import (
    SimpleMovingAverage as SmaDef,
)

OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "SUM": NumpySumOperator,
    "ASSIGN": NumpyAssignOperator,
    SmaDef.implementation_key(): SmaImp,
}
