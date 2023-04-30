from absl.testing import absltest

import numpy as np

from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent


class EventTest(absltest.TestCase):
    def setUp(self):
        self._event = NumpyEvent(
            data={
                (1, "hello"): IndexData(
                    features=[
                        np.array([1, 2, 3]),
                        np.array([4, 5, 6]),
                    ],
                    timestamps=np.array([0.1, 0.2, 0.3]),
                ),
                (2, "world"): IndexData(
                    features=[
                        np.array([7, 8]),
                        np.array([9, 10]),
                    ],
                    timestamps=np.array([0.4, 0.5]),
                ),
            },
            feature_names=["a", "b"],
            index_names=["x", "y"],
            is_unix_timestamp=False,
        )

    def test_data_access(self):
        self.assertEqual(
            repr(self._event.features()), "[('a', int64), ('b', int64)]"
        )
        self.assertEqual(
            repr(self._event.indexes()), "[('x', int64), ('y', str_)]"
        )

    def test_repr(self):
        print(self._event)
        self.assertEqual(
            repr(self._event),
            """indexes: [('x', int64), ('y', str_)]
features: [('a', int64), ('b', int64)]
events:
    x=1 y=hello (3 events):
        timestamps: [0.1 0.2 0.3]
        'a': [1 2 3]
        'b': [4 5 6]
    x=2 y=world (2 events):
        timestamps: [0.4 0.5]
        'a': [7 8]
        'b': [ 9 10]
""",
        )


if __name__ == "__main__":
    absltest.main()
