from typing import Dict

from temporian.core.operators.rename import RenameOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class RenameNumpyImplementation(OperatorImplementation):
    """Numpy implementation for the rename operator."""

    def __init__(self, operator: RenameOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, RenameOperator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, RenameOperator)

        features = self.operator.features
        index = self.operator.index

        # rename features
        new_feature_names = [
            features.get(feature_name, feature_name)
            for feature_name in input.schema.feature_names()
        ]

        # rename index
        new_index_names = [
            index.get(index_name, index_name)
            for index_name in input.schema.index_names()
        ]

        # check that after renaming everything there are no common values
        # between index names and feature names
        if set(new_feature_names).intersection(set(new_index_names)):
            raise ValueError(
                "Index names and feature names must be unique. Got"
                f" {new_feature_names} and {new_index_names}."
            )

        # create output evset
        output_evset = EventSet(
            data=input.data,
            schema=self.output_schema("output"),
        )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    RenameOperator, RenameNumpyImplementation
)
