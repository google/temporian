from absl.testing import absltest

from absl import logging
import numpy as np
import math
import pandas as pd
from numpy.testing import assert_array_equal
from datetime import datetime

from temporian.implementation.numpy.data.io import event_set, from_indexed_dicts
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

    def test_unix_timestamps_null(self):
        for timestamps in [
            ["2020-01-01", ""],
            ["2020-01-01", None],
            ["2020-01-01", np.datetime64("NaT")],
            pd.to_datetime(pd.Series(["2020-01-01", pd.NaT])),
            np.array(["2020-01-01", None]),  # dtype object
            np.array(["2020-01-01", ""]),  # dtype str
            np.array(["2020-01-01", None]).astype("datetime64[ns]"),
        ]:
            logging.info(f"Testing: {timestamps}")
            with self.assertRaisesRegex(
                ValueError, "Timestamps contain null/NaT values"
            ):
                _ = event_set(timestamps)

    def test_timestamps_nan(self):
        for timestamps in [np.array([1, None], dtype=float), [1.0, np.nan]]:
            logging.info(f"Testing: {timestamps}")
            with self.assertRaisesRegex(
                ValueError, "Timestamps contain NaN values"
            ):
                _ = event_set(timestamps)

    def test_timestamps_invalid_str(self):
        for timestamps in [
            ["2020-01-01", "nan"],
            ["2020-01-01", "-"],
        ]:
            logging.info(f"Testing: {timestamps}")
            with self.assertRaisesRegex(
                ValueError, "Error parsing datetime string"
            ):
                _ = event_set(timestamps)

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

    def test_from_indexed_dicts(self):
        evset = from_indexed_dicts(
            [
                (
                    {"i1": 1, "i2": "A"},
                    {"timestamp": [1, 2], "f1": [10, 11], "f2": ["X", "Y"]},
                ),
                (
                    {"i1": 1, "i2": "B"},
                    {"timestamp": [3, 4], "f1": [12, 13], "f2": ["X", "X"]},
                ),
                (
                    {"i1": 2, "i2": "A"},
                    {"timestamp": [5, 6], "f1": [14, 15], "f2": ["Y", "Y"]},
                ),
                (
                    {"i1": 2, "i2": "B"},
                    {"timestamp": [7, 8], "f1": [16, 17], "f2": ["Y", "Z"]},
                ),
            ]
        )
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8],
            features={
                "f1": [10, 11, 12, 13, 14, 15, 16, 17],
                "f2": ["X", "Y", "X", "X", "Y", "Y", "Y", "Z"],
                "i1": [1, 1, 1, 1, 2, 2, 2, 2],
                "i2": ["A", "A", "B", "B", "A", "A", "B", "B"],
            },
            indexes=["i1", "i2"],
        )
        self.assertEqual(evset, expected)


if __name__ == "__main__":
    absltest.main()
