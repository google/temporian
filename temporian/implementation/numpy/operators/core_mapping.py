from temporian.implementation.numpy.operators.assign import (
    AssignNumpyImplementation,
)
from temporian.implementation.numpy.operators.select import NumpySelectOperator
from temporian.implementation.numpy.operators.arithmetic import (
    ArithmeticNumpyImplementation,
)

OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "ASSIGN": AssignNumpyImplementation,
    "ARITHMETIC": ArithmeticNumpyImplementation,
}
