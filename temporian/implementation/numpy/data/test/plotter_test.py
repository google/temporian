from absl.testing import parameterized, absltest

import numpy as np

from temporian.implementation.numpy.data import plotter
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent


class PlotterTest(parameterized.TestCase):
    @parameterized.parameters((True,), (False,))
    def test_plot(self, interactive):
        event = NumpyEvent(
            data={
                (1,): IndexData(
                    features=[
                        np.array([1, 2, 3]),
                        np.array([4, 5, 6]),
                        np.array(["X", "Y", "X"]),
                    ],
                    timestamps=np.array([0.1, 0.2, 0.3]),
                ),
                (2,): IndexData(
                    features=[
                        np.array([7, 8]),
                        np.array([9, 10]),
                        np.array(["X", "Z"]),
                    ],
                    timestamps=np.array([0.4, 0.5]),
                ),
            },
            feature_names=["a", "b", "c"],
            index_names=["x"],
            is_unix_timestamp=False,
        )

        _ = plotter.plot(
            event, indexes=None, interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            event, indexes=1, interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            event, indexes=[1, 2], interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            event,
            indexes=[(1,), (2,)],
            interactive=interactive,
            return_fig=True,
        )

    def test_is_uniform(self):
        self.assertTrue(plotter.is_uniform([]))
        self.assertTrue(plotter.is_uniform([1]))
        self.assertTrue(plotter.is_uniform([1, 2, 3]))
        self.assertFalse(plotter.is_uniform([1, 2, 2.5]))


if __name__ == "__main__":
    absltest.main()
