from typing import Dict

import numpy as np

from temporian.core.operators.equal import EqualOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy.data.feature import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class EqualNumpyImplementation(OperatorImplementation):
    """New event with booleans where features are equal to a value."""

    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        def are_same_general_type(dtype1, dtype2) -> bool:
            """Check if two dtypes are of the same general type."""

            # convert temporian dtypes to numpy dtypes
            if dtype1 in DTYPE_REVERSE_MAPPING:
                dtype1 = np.dtype(DTYPE_REVERSE_MAPPING[dtype1])
            if dtype2 in DTYPE_REVERSE_MAPPING:
                dtype2 = np.dtype(DTYPE_REVERSE_MAPPING[dtype2])

            if dtype1.kind == "i" and dtype2.kind == "i":  # Both are integers
                return True

            if dtype1.kind == "f" and dtype2.kind == "f":  # Both are floats
                return True

            if (
                dtype1.kind == "U" and dtype2.kind == "U"
            ):  # Both are Unicode strings
                return True

            if dtype1.kind == "b" and dtype2.kind == "b":  # Both are booleans
                return True

            return False

        value = self.operator.attributes["value"]

        value_dtype = np.dtype(type(value))

        output_event = NumpyEvent(data={}, sampling=event.sampling)

        for index_value, features in event.data.items():
            equal_features = []
            for feature in features:
                # compare general dtypes. all ints, all floats, all strings.
                if are_same_general_type(feature.dtype, value_dtype):
                    equal_feature = np.equal(feature.data, value)
                else:
                    equal_feature = np.full(
                        feature.data.shape, False, dtype=np.bool_
                    )

                equal_np_feature = NumpyFeature(
                    f"{feature.name}_equal_{value}", equal_feature
                )
                equal_features.append(equal_np_feature)

            output_event.data[index_value] = equal_features

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    EqualOperator, EqualNumpyImplementation
)
