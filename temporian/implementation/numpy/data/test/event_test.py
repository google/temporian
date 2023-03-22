from absl.testing import absltest

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class EventTest(absltest.TestCase):
    def setUp(self):
        pass

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
                index=["x"],
                data={
                    (1,): np.array([0.1, 0.2, 0.3]),
                    (2,): np.array([0.4, 0.5]),
                },
            ),
        )

        self.assertEqual(
            repr(event),
            """Event
  data:
    (1,):
      a <int64>: [1 2 3]
      b <int64>: [4 5 6]
    (2,):
      a <int64>: [7 8]
      b <int64>: [ 9 10]
  sampling:
    index: ['x']
    data:
      (1,): [0.1 0.2 0.3]
      (2,): [0.4 0.5]
""",
        )


if __name__ == "__main__":
    absltest.main()
