from collections import defaultdict
from typing import Dict

from temporian.core.operators.add_index import AddIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


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
            src_feature_names.index(f_name)
            for f_name in self.operator.index_to_add
        ]

        # Idx of input features not added to index.
        kept_feature_idxs = [
            idx
            for idx, f_name in enumerate(src_feature_names)
            if f_name not in self.operator.index_to_add
        ]

        dst_data = {}
        for src_index, src_data in input.data.items():
            # Maps, for each new index value, the indices of the events in
            # src_data.
            #
            # TODO: Do more efficiently. E.g. with numpy masks.
            new_index_to_value_idxs = defaultdict(list)
            for event_idx, new_index in enumerate(
                zip(*[src_data.features[f_idx] for f_idx in new_index_idxs])
            ):
                new_index = tuple(new_index)
                new_index_to_value_idxs[new_index].append(event_idx)

            for new_index, example_idxs in new_index_to_value_idxs.items():
                # Note: The new index is added after the existing index items.
                dst_index = src_index + new_index
                assert isinstance(dst_index, tuple)

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
