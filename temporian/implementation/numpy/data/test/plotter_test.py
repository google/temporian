from absl.testing import absltest

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.data import plotter


class PlotterTest(absltest.TestCase):
    def test_plot(self):
        event = NumpyEvent(
            data={
                (1,): [
                    NumpyFeature("a", np.array([1, 2, 2.1])),
                    NumpyFeature("b", np.array([4, 5, 4.5])),
                ],
                (2,): [
                    NumpyFeature("a", np.array([7, 6])),
                    NumpyFeature("b", np.array([9, 5])),
                ],
            },
            sampling=NumpySampling(
                index=["x"],
                data={
                    (1,): np.array([0.1, 0.2, 0.25]),
                    (2,): np.array([0.4, 0.5]),
                },
            ),
        )

        _ = plotter.plot(event, indexes=None)
        _ = plotter.plot(event, indexes=1)
        _ = plotter.plot(event, indexes=[1, 2])
        _ = plotter.plot(event, indexes=[(1,), (2,)])

    def test_is_uniform(self):
        self.assertTrue(plotter.is_uniform([]))
        self.assertTrue(plotter.is_uniform([1]))
        self.assertTrue(plotter.is_uniform([1, 2, 3]))
        self.assertFalse(plotter.is_uniform([1, 2, 2.5]))


if __name__ == "__main__":
    absltest.main()
