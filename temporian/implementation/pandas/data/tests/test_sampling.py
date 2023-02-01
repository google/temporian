# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import pandas as pd

from temporian.implementation.pandas.data.sampling import PandasSampling


class PandasSamplingTest(absltest.TestCase):
    def test_create_correct(self) -> None:
        """Test creation is successful if last level is a DatetimeIndex."""
        PandasSampling.from_arrays(
            [
                ["a", "a", "b"],
                [
                    pd.Timestamp("2013-01-01"),
                    pd.Timestamp("2013-01-02"),
                    pd.Timestamp("2013-01-03"),
                ],
            ],
            names=["str", "datetime"],
        )

    def test_create_wrong_timestamps_dtype(self) -> None:
        """Test creation fails if last index level isn't a DatetimeIndex."""
        with self.assertRaises(ValueError):
            PandasSampling.from_arrays(
                [["a", "a", "b"], [1, 2, 3]], names=["str", "int"]
            )


if __name__ == "__main__":
    absltest.main()
