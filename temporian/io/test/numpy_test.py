import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.numpy import to_numpy


class ToNumpyTest(absltest.TestCase):
    def test_conversion_includes_timestamps(self):
        evset = event_set(
            timestamps=['2023-11-08T17:14:38', '2023-11-29T21:44:46'],
            features={
                "feature_1": [0.5, 0.6],
                "my_index": ["red", "red"],
            },
            indexes=["my_index"],
        )

        # Action: Convert the EventSet to numpy-flattened dictionary
        result = to_numpy(evset)

        expected_timestamps = np.array(['2023-11-08T17:14:38', '2023-11-29T21:44:46'], dtype='datetime64[s]')
        expected_feature_1 = np.array([0.5, 0.6])
        expected_my_index = np.array(["red", "red"])

        self.assertTrue(np.array_equal(result["timestamps"], expected_timestamps))
        self.assertTrue(np.array_equal(result["feature_1"], expected_feature_1))
        self.assertTrue(np.array_equal(result["my_index"], expected_my_index))


# Additional test cases can be added here to cover other scenarios

if __name__ == "__main__":
    absltest.main()
