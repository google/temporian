import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.numpy import to_numpy


class NumpyTest(absltest.TestCase):

    def test_correct(self):
        evset = event_set(
            timestamps=["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
            indexes=["my_index"],
        )

        result = to_numpy(evset)

        expected = {
            "timestamp": np.array(
                ["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
                dtype="datetime64[s]",
            ),
            "feature_1": np.array([0.5, 0.6]),
            "my_index": np.array([b"red", b"blue"]),
        }

        for k in expected:
            np.testing.assert_array_equal(
                np.sort(result[k]), np.sort(expected[k])
            )

    def test_no_index(self):
        evset = event_set(
            timestamps=["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
        )

        result = to_numpy(evset)

        expected = {
            "timestamp": np.array(
                ["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
                dtype="datetime64[s]",
            ),
            "feature_1": np.array([0.5, 0.6]),
            "my_index": np.array([b"red", b"blue"]),
        }

        for k in expected:
            np.testing.assert_array_equal(
                np.sort(result[k]), np.sort(expected[k])
            )

    def test_no_timestamps(self):
        evset = event_set(
            timestamps=["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
            indexes=["my_index"],
        )

        result = to_numpy(evset, timestamps=False)
        assert "timestamp" not in result

    def test_timestamp_to_datetime_param(self):
        evset = event_set(
            timestamps=[
                np.datetime64("2022-01-01"),
                np.datetime64("2022-01-02"),
            ],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
            indexes=["my_index"],
        )

        result = to_numpy(evset, timestamp_to_datetime=False)

        assert "timestamp" in result
        assert np.issubdtype(result["timestamp"].dtype, np.float64)

    def test_empty_event_set(self):
        evset = event_set(
            timestamps=["2023-11-08T17:14:38", "2023-11-29T21:44:46"]
        )
        result = to_numpy(evset)

        expected = {
            "timestamp": np.array(
                ["2023-11-08T17:14:38", "2023-11-29T21:44:46"],
                dtype="datetime64[s]",
            )
        }

        np.testing.assert_array_equal(
            np.sort(result["timestamp"]), np.sort(expected["timestamp"])
        )


if __name__ == "__main__":
    absltest.main()
