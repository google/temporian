from temporian.implementation.numpy.operators.assign import (
    AssignNumpyImplementation,
)
from temporian.implementation.numpy.operators.select import NumpySelectOperator
from temporian.implementation.numpy.operators.sum import NumpySumOperator
from temporian.implementation.numpy.operators.arithmetic import (
    ArithmeticNumpyImplementation,
)

OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "SUM": NumpySumOperator,
    "ASSIGN": AssignNumpyImplementation,
    "ARITHMETIC": ArithmeticNumpyImplementation,
}
