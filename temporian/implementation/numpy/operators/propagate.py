"""Implementation for the Propagate operator."""


from typing import Dict

import numpy as np
import itertools

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy import implementation_lib


class PropagateNumpyImplementation:
    """Numpy implementation for the propagat operator."""

    def __init__(self, op: Propagate) -> None:
        assert isinstance(op, Propagate)
        self._op = op

    def __call__(
        self,
        event: NumpyEvent,
        add_event: NumpyEvent = None,
    ) -> Dict[str, NumpyEvent]:
        # All the features of "add_event" are added as part of the new index.
        added_index = self._op.added_index()
        dst_sampling = NumpySampling(
            index=event.sampling.index + added_index, data={}
        )
        dst_event = NumpyEvent(data={}, sampling=dst_sampling)

        # For each index
        for src_index_item, src_event_item in event.data.items():
            src_add_event_item = add_event.data[src_index_item]
            src_timestamps_item = event.sampling.data[src_index_item]

            # TODO: Change the NumpyEvent structure so that feature names are
            # not repeated for each index value, and such that features can be
            # refered by an integer accross all index values. When done, remove
            # the following assets.
            assert len(src_add_event_item) == len(added_index)
            index_uniques = []
            for idx, key in enumerate(added_index):
                assert key == src_add_event_item[idx].name

            index_uniques = [
                np.unique(src_add_event_item[idx].data)
                for idx in range(len(added_index))
            ]

            for extra_index in itertools.product(*index_uniques):
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
