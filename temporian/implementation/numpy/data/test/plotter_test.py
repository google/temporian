from absl.testing import parameterized, absltest

import matplotlib

from temporian.implementation.numpy.data import plotter
from temporian.implementation.numpy.data.io import event_set


class PlotterTest(parameterized.TestCase):
    def setUp(self):
        # Make sure that the plot functions don't fail on command line
        print("Setting matplotlib backend to: agg")
        matplotlib.use("agg")

    @parameterized.parameters((True,), (False,))
    def test_plot(self, interactive):
        try:
            import IPython.display
        except ImportError:
            # IPython is not installed / supported
            return

        evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            features={
                "a": [1, 2, 3, 7, 8],
                "b": [4, 5, 6, 9, 10],
                "c": ["X", "Y", "X", "X", "Z"],
                "x": [1, 1, 1, 2, 2],
            },
            indexes=["x"],
        )

        _ = plotter.plot(
            evset, indexes=None, interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            evset, indexes=1, interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            evset, indexes=[1, 2], interactive=interactive, return_fig=True
        )
        _ = plotter.plot(
            evset,
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
