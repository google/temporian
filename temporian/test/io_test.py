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
    def test_from_csv(self) -> None:
        path = os.path.join(
            test_data(), "temporian/test/test_data/io/input.csv"
        )
        evset = tp.from_csv(
            path=path,
            timestamps="timestamp",
            indexes=["product_id"],
        )

        expected_evset = tp.from_pandas(
            pd.DataFrame(
                [
                    [666964, 1.0, 740.0],
                    [666964, 2.0, 508.0],
                    [574016, 3.0, 573.0],
                ],
                columns=["product_id", "timestamp", "costs"],
            ),
            indexes=["product_id"],
            timestamps="timestamp",
        )

        self.assertEqual(evset, expected_evset)

    def test_to_csv(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        evset = tp.from_pandas(df=df, indexes=["product_id"])

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "events.csv")
            tp.to_csv(evset=evset, path=path)

            # check if file exists
            self.assertTrue(Path(path).exists())

            saved_evset = tp.from_csv(
                path=path,
                timestamps="timestamp",
                indexes=["product_id"],
            )

            self.assertEqual(evset, saved_evset)


if __name__ == "__main__":
    absltest.main()
