from absl.testing import absltest

from absl import logging
import numpy as np
import math
import pandas as pd
from numpy.testing import assert_array_equal
from datetime import datetime

from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.data.schema import Schema
from temporian.core.data.dtype import DType


class IOTest(absltest.TestCase):
    def test_event_set(self):
        # a
        evtset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
                "feature_4": ["a", "b", math.nan, "c"],
                "feature_5": pd.Series(["d", "e", math.nan, "f"]),
                "feature_6": pd.Series([1, 2, 3, 4]),
            },
            index_features=["feature_2"],
        )

        expected_schema = Schema(
            features=[
                ("feature_1", DType.FLOAT64),
                ("feature_3", DType.INT64),
                ("feature_4", DType.STRING),
                ("feature_5", DType.STRING),
                ("feature_6", DType.INT64),
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
                        np.array(["d", ""]),
                        np.array([1, 3]),
                    ],
                    timestamps=np.array([1.0, 3.0], dtype=np.float64),
                    schema=expected_schema,
                ),
                ("blue",): IndexData(
                    features=[
                        np.array([0.6, 0.9]),
                        np.array([-1, 5]),
                        np.array(["b", "c"]),
                        np.array(["e", "f"]),
                        np.array([2, 4]),
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

    def test_timestamps_non_unix_time(self):
        for timestamps in [
            [2],
            [2.0],
            np.array([2], dtype=np.int32),
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.float32),
            np.array([2], dtype=np.float64),
            pd.Series([2]),
        ]:
            logging.info("Testing: %s (%s)", timestamps, type(timestamps))
            evtset = event_set(timestamps)
            assert_array_equal(
                evtset.get_arbitrary_index_data().timestamps,
                np.array([2], dtype=np.float64),
            )
            self.assertFalse(evtset.schema.is_unix_timestamp)

    def test_timestamps_unix_time(self):
        for timestamps in [
            # Python
            [datetime(1970, 1, 2)],
            ["1970-01-02"],
            [b"1970-01-02"],
            # Numpy
            [np.datetime64("1970-01-02")],
            np.array(["1970-01-02"], dtype=np.str_),
            np.array(["1970-01-02"], dtype=np.string_),
            np.array(["1970-01-02"], dtype=np.object_),
            # Pandas
            pd.Series([pd.Timestamp("1970-01-02")]),
            pd.Series(["1970-01-02"]),
        ]:
            logging.info("Testing: %s (%s)", timestamps, type(timestamps))
            evtset = event_set(timestamps)
            assert_array_equal(
                evtset.get_arbitrary_index_data().timestamps,
                np.array([86400], dtype=np.float64),
            )
            self.assertTrue(evtset.schema.is_unix_timestamp)


if __name__ == "__main__":
    absltest.main()
