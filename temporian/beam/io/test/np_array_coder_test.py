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

import numpy as np
from absl.testing import absltest, parameterized
from apache_beam.typehints import trivial_inference
from apache_beam.coders import typecoders
from temporian.beam.io import np_array_coder
from numpy.testing import assert_array_equal


class NDArrayCoderTest(parameterized.TestCase):
    def test_registration(self):
        value = np.array([1, 2, 3])
        value_type = trivial_inference.instance_to_type(value)
        value_coder = typecoders.registry.get_coder(value_type)
        self.assertIsInstance(value_coder, np_array_coder.NDArrayCoder)

    @parameterized.parameters(
        (np.array([], np.int32),),
        (np.array([], np.float32),),
        (np.array([[]], np.int64),),
        (np.array([[], []], np.float32),),
        (np.array([1, 2, 3]),),
        (np.array([[1], [2], [3]]),),
        (np.array(["1", "2", "3"]),),
        (np.array([["a", "b"], ["c", "d"]]),),
        (np.array([b"a", b"bbb", b"cccc", b"ddddd"]),),
        (np.array([1.0, 2.0, 3.0]),),
        (np.array([[1.0, 2.0], [3.0, 4.0]]),),
        (np.array([[[True], [False]], [[False], [True]]]),),
    )
    def test_encoding_and_decoding(self, value):
        coder = np_array_coder.NDArrayCoder()
        encoded = coder.encode(value)
        decoded = coder.decode(encoded)
        assert_array_equal(value, decoded)


if __name__ == "__main__":
    absltest.main()
