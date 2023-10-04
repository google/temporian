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
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
                "feature_4": ["a", "b", math.nan, "c"],
                "feature_5": pd.Series(["d", "e", math.nan, "f"]),
                "feature_6": pd.Series([1, 2, 3, 4]),
            },
            indexes=["feature_2"],
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
                (b"red",): IndexData(
                    features=[
                        np.array([0.5, math.nan]),
                        np.array([10, 5]),
                        np.array([b"a", b"nan"]),
                        np.array([b"d", b""]),
                        np.array([1, 3]),
                    ],
                    timestamps=np.array([1.0, 3.0], dtype=np.float64),
                    schema=expected_schema,
                ),
                (b"blue",): IndexData(
                    features=[
                        np.array([0.6, 0.9]),
                        np.array([-1, 5]),
                        np.array([b"b", b"c"]),
                        np.array([b"e", b"f"]),
                        np.array([2, 4]),
                    ],
                    timestamps=np.array([2.0, 4.0], dtype=np.float64),
                    schema=expected_schema,
                ),
            },
            schema=expected_schema,
        )

        print("evset:\n", evset)
        print("expected_evset:\n", expected_evset)
        self.assertEqual(evset, expected_evset)

    def test_timestamps_non_unix_time(self):
        for timestamps in [
            [2],
            [2.0],
            np.array([2], dtype=np.int32),
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.float32),
            np.array([2], dtype=np.float64),
            np.array([2], dtype=np.byte),
            np.array([2], dtype=np.short),
            np.array([2], dtype=np.longlong),
            np.array([2], dtype=np.ubyte),
            np.array([2], dtype=np.ushort),
            np.array([2], dtype=np.ulonglong),
            pd.Series([2]),
        ]:
            logging.info("Testing: %s (%s)", timestamps, type(timestamps))
            evset = event_set(timestamps)
            assert_array_equal(
                evset.get_arbitrary_index_data().timestamps,
                np.array([2], dtype=np.float64),
            )
            self.assertFalse(evset.schema.is_unix_timestamp)

    def test_timestamps_unix_time(self):
        for timestamps in [
            # Python
            [datetime(1970, 1, 2)],
            ["1970-01-02"],
            [b"1970-01-02"],
            # Numpy
            [np.datetime64("1970-01-02")],
            np.array(["1970-01-02"], dtype=np.str_),
            np.array(["1970-01-02"], dtype=np.bytes_),
            np.array(["1970-01-02"], dtype=np.object_),
            # Pandas
            pd.Series([pd.Timestamp("1970-01-02")]),
            np.array([pd.Timestamp("1970-01-02")]),  # dtype object
            pd.Series(["1970-01-02"]),
        ]:
            logging.info("Testing: %s (%s)", timestamps, type(timestamps))
            evset = event_set(timestamps)
            assert_array_equal(
                evset.get_arbitrary_index_data().timestamps,
                np.array([86400], dtype=np.float64),
            )
            self.assertTrue(evset.schema.is_unix_timestamp)

    def test_arrays_not_same_length(self):
        with self.assertRaisesRegex(
            ValueError, "Timestamps and all features must have the same length."
        ):
            event_set(
                timestamps=[1, 2],
                features={
                    "feature_1": [0.59],
                },
            )

        with self.assertRaisesRegex(
            ValueError, "Timestamps and all features must have the same length."
        ):
            event_set(
                timestamps=[1],
                features={
                    "feature_1": [0.59, 0.5],
                },
            )

        with self.assertRaisesRegex(
            ValueError, "Timestamps and all features must have the same length."
        ):
            event_set(
                timestamps=[1, 2],
                features={
                    "feature_1": [0.59, 0.5],
                    "feature_2": [0.59, 0.5, 0.5],
                },
            )

        # Shouldn't raise
        event_set(timestamps=[1])

    def test_feature_wrong_type(self):
        with self.assertRaisesRegex(
            ValueError, "Feature values should be provided in a tuple, list"
        ):
            event_set(
                timestamps=[1, 2],
                features={
                    # np.array({1, 2}) would produce a single-item value
                    "y": {1, 2},
                },
            )


if __name__ == "__main__":
    absltest.main()
