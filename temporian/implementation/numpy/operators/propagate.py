"""Implementation for the Propagate operator."""


from typing import Dict

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PropagateNumpyImplementation(OperatorImplementation):
    """Numpy implementation for the propagate operator."""

    def __init__(self, operator: Propagate) -> None:
        assert isinstance(operator, Propagate)
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, to: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        # All the features of "to" are added as part of the new index.
        added_index = [
            index_lvl_name for index_lvl_name, _ in self._operator.added_index()
        ]
        num_new_index = len(added_index)

        dst_sampling = NumpySampling(
            index=event.sampling.index + added_index, data={}
        )
        dst_event = NumpyEvent(data={}, sampling=dst_sampling)

        # For each index
        for src_index_item, src_event_item in event.data.items():
            src_to_item = to.data[src_index_item]
            src_timestamps_item = event.sampling.data[src_index_item]

            # TODO: Change the NumpyEvent structure so that feature names are
            # not repeated for each index value, and such that features can be
            # refered by an integer accross all index values. When done, remove
            # the following assets.
            assert len(src_to_item) == len(added_index)
            for idx, key in enumerate(added_index):
                assert key == src_to_item[idx].name

            num_values = len(src_to_item[0].data)
            unique_index_values = set(
                [
                    tuple(
                        [src_to_item[j].data[i] for j in range(num_new_index)]
                    )
                    for i in range(num_values)
                ]
            )

            for extra_index in unique_index_values:
                dst_index = src_index_item + extra_index

                # Copy the sampling data
                #
                # Note: The data is effectively not duplicated. This is possible
                # since ops are not allowed to modify op data during processing.
                #
                # TODO: This means that output event data might contain
                # cross-references. This should be documented.
                dst_sampling.data[dst_index] = src_timestamps_item

                # Copy the feature data
                dst_mts = []
                for src_feature in src_event_item:
                    dst_mts.append(
                        NumpyFeature(src_feature.name, src_feature.data)
                    )
                dst_event.data[dst_index] = dst_mts

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Propagate, PropagateNumpyImplementation
)
