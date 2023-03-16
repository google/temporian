import os
import pandas as pd

from absl.testing import absltest
from pathlib import Path

from temporian.io.read_event import read_event
from temporian.implementation.numpy.data.event import NumpyEvent


class IOTest(absltest.TestCase):
    def setUp(self) -> None:
        self.read_path = "temporian/test/test_data/io/read_event.csv"
        self.save_path = "temporian/test/test_data/io/save_event.csv"

        if Path(self.save_path).exists():
            os.remove(self.save_path)

    def test_read_event(self) -> None:
        event_data = read_event(
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
        pass


if __name__ == "__main__":
    absltest.main()
