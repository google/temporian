from absl.testing import absltest

import numpy as np
import math

from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.data.schema import Schema
from temporian.core.data.dtype import DType


class IOTest(absltest.TestCase):
    def test_event_set(self):
        evtset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
                "feature_4": ["a", "b", math.nan, "c"],
            },
            index_features=["feature_2"],
        )

        expected_schema = Schema(
            features=[
                ("feature_1", DType.FLOAT64),
                ("feature_3", DType.INT64),
                ("feature_4", DType.STRING),
            ],
            indexes=[("feature_2", DType.STRING)],
            is_unix_timestamp=False,
        )
        expected_evset = EventSet(
            data={
                ("red",): IndexData(
                    features=[
                        np.array([0.5, math.nan]),
                        np.array([10, 5]),
                        np.array(["a", ""]),
                    ],
                    timestamps=np.array([1.0, 3.0], dtype=np.float64),
                    schema=expected_schema,
                ),
                ("blue",): IndexData(
                    features=[
                        np.array([0.6, 0.9]),
                        np.array([-1, 5]),
                        np.array(["b", "c"]),
                    ],
                    timestamps=np.array([2.0, 4.0], dtype=np.float64),
                    schema=expected_schema,
                ),
            },
            schema=expected_schema,
        )

        print("evtset:\n", evtset)
        print("expected_evset:\n", expected_evset)
        self.assertEqual(repr(evtset), repr(expected_evset))


if __name__ == "__main__":
    absltest.main()
