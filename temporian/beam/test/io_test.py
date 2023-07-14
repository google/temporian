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

import os
import tempfile

from absl.testing import absltest
from absl import flags
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from temporian.beam.io import read_csv_raw, read_csv, write_csv
from temporian.implementation.numpy.data.io import event_set
from temporian.io.csv import to_csv


def test_data() -> str:
    return os.path.join(flags.FLAGS.test_srcdir, "temporian")


class IOTest(absltest.TestCase):
    def test_read_csv_raw(self):
        input_csv_path = os.path.join(
            test_data(), "temporian/test/test_data/io/input.csv"
        )
        with TestPipeline() as p:
            output = p | read_csv_raw(input_csv_path)
            assert_that(
                output,
                equal_to(
                    [
                        {
                            "product_id": "666964",
                            "timestamp": "1.0",
                            "costs": "740.0",
                        },
                        {
                            "product_id": "666964",
                            "timestamp": "2.0",
                            "costs": "508.0",
                        },
                        {
                            "product_id": "574016",
                            "timestamp": "3.0",
                            "costs": "573.0",
                        },
                    ]
                ),
            )

    def test_read_and_write_csv(self):
        # Create csv dataset
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, "input.csv")
        output_path = os.path.join(tmp_dir, "output.csv")
        input_data = event_set(
            timestamps=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            features={
                "a": [2, 3, 4, 3, 2, 22, 23, 24, 23, 22],
                "b": ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"],
                "c": ["X", "Y", "Y", "X", "Z", "Z", "Z", "X", "Y", "X"],
                "d": [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
                "e": [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
            },
            indexes=["b", "e"],
        )
        to_csv(input_data, path=input_path)

        # Note: It is not clear how to check values of PCollection that contains
        # numpy arrays. assert_that + equal_to does not work.
        with TestPipeline() as p:
            output = (
                p
                | read_csv(input_path, input_data.schema)
                | write_csv(
                    output_path, input_data.schema, shard_name_template=""
                )
            )
            assert_that(
                output,
                equal_to([output_path]),
            )

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(
                content,
                """timestamp,b,e,a,c,d
1.0,y,1,22,Z,-6
2.0,y,1,23,Z,-7
3.0,y,1,24,X,-8
4.0,y,1,23,Y,-9
5.0,y,1,22,X,-10
4.0,x,2,3,X,-4
5.0,x,2,2,Z,-5
1.0,x,1,2,X,-1
2.0,x,1,3,Y,-2
3.0,x,1,4,Y,-3
""",
            )


if __name__ == "__main__":
    absltest.main()
