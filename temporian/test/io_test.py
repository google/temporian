import os
from pathlib import Path

from absl import flags
from absl.testing import absltest
import pandas as pd
import tempfile

import temporian as tp


def test_data() -> str:
    return os.path.join(flags.FLAGS.test_srcdir, "temporian")


class IOTest(absltest.TestCase):
    def test_read_event_set(self) -> None:
        path = os.path.join(
            test_data(), "temporian/test/test_data/io/input.csv"
        )
        evset = tp.read_event_set(
            path=path,
            timestamp_column="timestamp",
            index_names=["product_id"],
        )

        expected_evset = tp.pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [666964, 1.0, 740.0],
                    [666964, 2.0, 508.0],
                    [574016, 3.0, 573.0],
                ],
                columns=["product_id", "timestamp", "costs"],
            ),
            index_names=["product_id"],
            timestamp_column="timestamp",
        )

        self.assertEqual(evset, expected_evset)

    def test_save_event_set(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        evset = tp.pd_dataframe_to_event_set(df=df, index_names=["product_id"])

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "events.csv")
            tp.save_event_set(evset=evset, path=path)

            # check if file exists
            self.assertTrue(Path(path).exists())

            saved_evset = tp.read_event_set(
                path=path,
                timestamp_column="timestamp",
                index_names=["product_id"],
            )

            self.assertEqual(evset, saved_evset)


if __name__ == "__main__":
    absltest.main()
