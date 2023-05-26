from absl.testing import absltest

import numpy as np
import math

from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet

# Load all the implementations
from temporian.implementation.numpy.operators import all_operators as _impls


class IOTest(absltest.TestCase):
    def test_event_set(self):
        evtset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
            },
            index_features=["feature_2"],
        )

        expected_evset = EventSet(
            data={
                ("red",): IndexData(
                    features=[np.array([0.5, math.nan]), np.array([10, 5])],
                    timestamps=np.array([1.0, 3.0], dtype=np.float64),
                ),
                ("blue",): IndexData(
                    features=[np.array([0.6, 0.9]), np.array([-1, 5])],
                    timestamps=np.array([2.0, 4.0], dtype=np.float64),
                ),
            },
            feature_names=["feature_1", "feature_3"],
            index_names=["feature_2"],
            is_unix_timestamp=False,
        )

        print("evtset:\n", evtset)
        print("expected_evset:\n", expected_evset)
        self.assertEquals(repr(evtset), repr(expected_evset))


if __name__ == "__main__":
    absltest.main()
