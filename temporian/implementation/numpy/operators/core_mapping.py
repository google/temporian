from temporian.implementation.numpy.operators.assign import (
    AssignNumpyImplementation,
)
from temporian.implementation.numpy.operators.arithmetic import (
    ArithmeticNumpyImplementation,
)
from temporian.implementation.numpy.operators.lag import LagNumpyImplementation
from temporian.implementation.numpy.operators.select import NumpySelectOperator
from temporian.implementation.numpy.operators.simple_moving_average import (
    SimpleMovingAverageNumpyImplementation as SmaImp,
)
from temporian.core.operators.simple_moving_average import (
    SimpleMovingAverage as SmaDef,
)

# TODO: Use a registration mechanism instead.
# TODO: Make sure the implementation key is defined only one (in the operator
# definition).
OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "ASSIGN": AssignNumpyImplementation,
    "ARITHMETIC": ArithmeticNumpyImplementation,
    "LAG": LagNumpyImplementation,
    SmaDef.operator_key(): SmaImp,  # Simple moving average
}
