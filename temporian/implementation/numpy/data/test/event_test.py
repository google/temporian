from absl.testing import absltest

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class EventTest(absltest.TestCase):
    def test_repr(self):
        event = NumpyEvent(
            data={
                (1,): [
                    NumpyFeature("a", np.array([1, 2, 3])),
                    NumpyFeature("b", np.array([4, 5, 6])),
                ],
                (2,): [
                    NumpyFeature("a", np.array([7, 8])),
                    NumpyFeature("b", np.array([9, 10])),
                ],
            },
            sampling=NumpySampling(
                index={"x": np.int32},
                data={
                    (1,): np.array([0.1, 0.2, 0.3]),
                    (2,): np.array([0.4, 0.5]),
                },
            ),
        )

        self.assertEqual(
            repr(event),
            """data (2):
    (1,):
        a <INT64>: [1 2 3]
        b <INT64>: [4 5 6]
    (2,):
        a <INT64>: [7 8]
        b <INT64>: [ 9 10]
sampling:
    index: {'x': <class 'numpy.int32'>}
    data (2):
        (1,): [0.1 0.2 0.3]
        (2,): [0.4 0.5]
""",
        )


if __name__ == "__main__":
    absltest.main()
