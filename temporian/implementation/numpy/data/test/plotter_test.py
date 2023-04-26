from absl.testing import absltest

import numpy as np

from temporian.implementation.numpy.data import plotter
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent


class PlotterTest(absltest.TestCase):
    def test_plot(self):
        event = NumpyEvent(
            data={
                (1,): IndexData(
                    features=[
                        np.array([1, 2, 3]),
                        np.array([4, 5, 6]),
                    ],
                    timestamps=np.array([0.1, 0.2, 0.3]),
                ),
                (2,): IndexData(
                    features=[
                        np.array([7, 8]),
                        np.array([9, 10]),
                    ],
                    timestamps=np.array([0.4, 0.5]),
                ),
            },
            feature_names=["a", "b"],
            index_names="x",
            is_unix_timestamp=False,
        )
        _ = plotter.plot(event, indexes=(1,))


if __name__ == "__main__":
    absltest.main()
