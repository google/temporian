from collections import defaultdict
from typing import Dict

from temporian.core.operators.add_index import AddIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc


class AddIndexNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the set index operator."""

    def __init__(self, operator: AddIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, AddIndexOperator)
        output_node = self.operator.outputs["output"]

        # Idx of input features added to index.
        src_feature_names = input.schema.feature_names()
        new_index_idxs = [
            src_feature_names.index(f_name) for f_name in self.operator.indexes
        ]

        # Idx of input features not added to index.
        kept_feature_idxs = [
            idx
            for idx, f_name in enumerate(src_feature_names)
            if f_name not in self.operator.indexes
        ]

        dst_data = {}
        for src_index, src_data in input.data.items():
            index_features = [src_data.features[i] for i in new_index_idxs]
            (
                group_keys,
                row_idxs,
                group_begin_idx,
            ) = operators_cc.add_index_compute_index(index_features)

            for group_idx, group_key in enumerate(group_keys):
                dst_index = src_index + group_key
                assert isinstance(dst_index, tuple)

                example_idxs = row_idxs[
                    group_begin_idx[group_idx] : group_begin_idx[group_idx + 1]
                ]
                dst_data[dst_index] = IndexData(
                    features=[
                        src_data.features[f_idx][example_idxs]
                        for f_idx in kept_feature_idxs
                    ],
                    timestamps=src_data.timestamps[example_idxs],
                    schema=output_node.schema,
                )

        return {
            "output": EventSet(
                data=dst_data,
                schema=output_node.schema,
            )
        }


implementation_lib.register_operator_implementation(
    AddIndexOperator, AddIndexNumpyImplementation
)
