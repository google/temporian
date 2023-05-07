import os
from pathlib import Path

from absl.testing import absltest
import pandas as pd

import temporian as tp
from temporian.implementation.numpy.data.event_set import EventSet


class IOTest(absltest.TestCase):
    def setUp(self) -> None:
        self.read_path = "temporian/test/test_data/io/input.csv"
        self.save_path = "temporian/test/test_data/io/save_event.csv"

        if Path(self.save_path).exists():
            os.remove(self.save_path)

    def tearDown(self) -> None:
        if Path(self.save_path).exists():
            os.remove(self.save_path)

    def test_read_event_set(self) -> None:
        evset = tp.read_event_set(
            path=self.read_path,
            timestamp_column="timestamp",
            index_names=["product_id"],
        )

        expected_evset = EventSet.from_dataframe(
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

        evset = EventSet.from_dataframe(df=df, index_names=["product_id"])

        tp.save_event_set(evset=evset, path=self.save_path)

        # check if file exists
        self.assertTrue(Path(self.save_path).exists())

        saved_evset = tp.read_event_set(
            path=self.save_path,
            timestamp_column="timestamp",
            index_names=["product_id"],
        )

        self.assertEqual(evset, saved_evset)


if __name__ == "__main__":
    absltest.main()
