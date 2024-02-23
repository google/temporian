import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.numpy import to_numpy


class NumpyTest(absltest.TestCase):

    def test_correct(self):
        evset = event_set(
            timestamps=['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
            indexes=["my_index"],
        )

        result = to_numpy(evset)

        expected_timestamps = np.array(['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
                                       dtype='datetime64[s]')
        expected_feature_1 = np.array([0.5, 0.6])
        expected_my_index = np.array(["red", "blue"])

        # Convert byte strings in x to Unicode strings for a fair comparison
        result['my_index'] = np.array([item.decode('utf-8') for item in result['my_index']])

        np.testing.assert_array_equal(np.sort(result["feature_1"]), np.sort(expected_feature_1))
        np.testing.assert_array_equal(np.sort(result["timestamp"]), np.sort(expected_timestamps))
        np.testing.assert_array_equal(np.sort(result["my_index"]), np.sort(expected_my_index))

    def test_no_index(self):
        evset = event_set(
            timestamps=['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            }
        )

        result = to_numpy(evset)

        expected_timestamps = np.array(['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
                                       dtype='datetime64[s]')
        expected_feature_1 = np.array([0.5, 0.6])
        expected_my_index = np.array(["red", "blue"])

        # Convert byte strings in x to Unicode strings for a fair comparison
        result['my_index'] = np.array([item.decode('utf-8') for item in result['my_index']])

        np.testing.assert_array_equal(np.sort(result["feature_1"]), np.sort(expected_feature_1))
        np.testing.assert_array_equal(np.sort(result["timestamp"]), np.sort(expected_timestamps))
        np.testing.assert_array_equal(np.sort(result["my_index"]), np.sort(expected_my_index))

    def test_no_timestamps(self):
        evset = event_set(
            timestamps=['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "blue"],
            },
            indexes=["my_index"],
        )

        result = to_numpy(evset, timestamps=False)

        expected_timestamps = np.array(['2023-11-08T17:14:38', '2023-11-29T21:44:46'], dtype='datetime64[s]')
        expected_feature_1 = np.array([0.5, 0.6])
        expected_my_index = np.array(["red", "blue"])

        # Convert byte strings in x to Unicode strings for a fair comparison
        result['my_index'] = np.array([item.decode('utf-8') for item in result['my_index']])

        np.testing.assert_array_equal(np.sort(result["feature_1"]), np.sort(expected_feature_1))
        np.testing.assert_array_equal(np.sort(result["my_index"]), np.sort(expected_my_index))


if __name__ == "__main__":
    absltest.main()
