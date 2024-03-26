from absl.testing import absltest

import argparse
from temporian.cli import view

import temporian as tp


class ViewTest(absltest.TestCase):
    def test_camp01(self):
        self.assertEqual(view.clamp01(2), 1)
        self.assertEqual(view.clamp01(-2), 0)

    def test_render_frame(self):
        evtsets = [
            tp.event_set(
                [1, 2, 3, 4],
                features={
                    "f1": [1, 2, 3, 4],
                    "f2": ["A", "B", "C", "A"],
                    "f3": [True, False, False, True],
                    "i1": [1, 1, 2, 2],
                },
                indexes=["i1"],
            )
        ]

        class TestArgs:
            pass

        args = TestArgs()
        setattr(args, "font_size", 14)
        v = view.Viewer(evtsets, args, create_ui=False)
        frame = v.render_frame()
        frame.save("/tmp/frame.png")


if __name__ == "__main__":
    absltest.main()
