from temporian.implementation.numpy.operators.assign import NumpyAssignOperator
from temporian.implementation.numpy.operators.select import NumpySelectOperator
from temporian.implementation.numpy.operators.sum import NumpySumOperator

OPERATOR_IMPLEMENTATIONS = {
    "SELECT": NumpySelectOperator,
    "SUM": NumpySumOperator,
    "ASSIGN": NumpyAssignOperator,
}
