import os
import pandas as pd

from absl.testing import absltest
from pathlib import Path

import temporian as tp
from temporian.implementation.numpy.data.event import NumpyEvent


class IOTest(absltest.TestCase):
    def setUp(self) -> None:
        self.read_path = "temporian/test/test_data/io/input.csv"
        self.save_path = "temporian/test/test_data/io/save_event.csv"

        if Path(self.save_path).exists():
            os.remove(self.save_path)

    def tearDown(self) -> None:
        if Path(self.save_path).exists():
            os.remove(self.save_path)

    def test_read_event(self) -> None:
        event_data = tp.read_event(
            path=self.read_path,
            timestamp_column="timestamp",
            index_names=["product_id"],
        )

        expected_event = NumpyEvent.from_dataframe(
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

        self.assertEqual(event_data, expected_event)

    def test_save_event(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        event = NumpyEvent.from_dataframe(df=df, index_names=["product_id"])

        tp.save_event(event=event, path=self.save_path)

        # check if file exists
        self.assertTrue(Path(self.save_path).exists())

        saved_event = tp.read_event(
            path=self.save_path,
            timestamp_column="timestamp",
            index_names=["product_id"],
        )

        self.assertEqual(event, saved_event)


if __name__ == "__main__":
    absltest.main()
