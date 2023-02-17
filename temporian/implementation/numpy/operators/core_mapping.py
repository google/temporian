from temporian.implementation.numpy.operators.assign import (
    AssignNumpyImplementation,
)
from temporian.implementation.numpy.operators.arithmetic import (
    ArithmeticNumpyImplementation,
)
from temporian.implementation.numpy.operators.lag import LagNumpyImplementation
from temporian.implementation.numpy.operators.select import NumpySelectOperator


OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "ASSIGN": AssignNumpyImplementation,
    "ARITHMETIC": ArithmeticNumpyImplementation,
    "LAG": LagNumpyImplementation,
    "LEAK": LagNumpyImplementation,
}
