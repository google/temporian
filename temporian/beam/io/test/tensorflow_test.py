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
from apache_beam.testing.test_pipeline import TestPipeline

from temporian.beam.io.tensorflow import (
    to_tensorflow_record,
    from_tensorflow_record,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.io.tensorflow import (
    to_tensorflow_record as in_process_to_tensorflow_record,
)
from temporian.io.tensorflow import (
    from_tensorflow_record as in_process_from_tensorflow_record,
)
from temporian.implementation.numpy.operators.test.utils import (
    assertEqualEventSet,
)


class IOTest(absltest.TestCase):
    def test_to_and_from_tensorflow_record(self):
        tmp_dir_handle = tempfile.TemporaryDirectory()
        input_file = os.path.join(tmp_dir_handle.name, "input")
        output_file = os.path.join(tmp_dir_handle.name, "output")

        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "f1": [10, 11, 12, 13],
                "f2": [0.1, 0.2, 0.3, 0.4],
                "f3": [b"a", b"b", b"c", b"d"],
                "i1": [1, 1, 2, 2],
                "i2": [b"x", b"x", b"x", b"y"],
            },
            indexes=["i1", "i2"],
        )

        in_process_to_tensorflow_record(evset, input_file)
        with TestPipeline() as p:
            (
                p
                | from_tensorflow_record(input_file, evset.schema)
                | to_tensorflow_record(
                    output_file, evset.schema, shard_name_template=""
                )
            )
            p.run()

        loaded_evtset = in_process_from_tensorflow_record(
            output_file, evset.schema
        )
        assertEqualEventSet(self, evset, loaded_evtset)


if __name__ == "__main__":
    absltest.main()
