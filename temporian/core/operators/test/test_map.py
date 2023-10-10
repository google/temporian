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
from absl.testing.parameterized import TestCase
import numpy as np

from temporian.core.compilation import compile
from temporian.core.serialization import save
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class MapTest(TestCase):
    def test_basic(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [10, 20, 30]})

        result = evset.map(lambda x: x * 2)

        expected = event_set(
            timestamps=[1, 2, 3],
            features={"x": [20, 40, 60]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_builtin_func(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [1.1, 2.3, 2.8]})

        result = evset.map(round)

        expected = event_set(
            timestamps=[1, 2, 3],
            features={"x": [1.0, 2.0, 3.0]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_with_extras(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [10, 20, 30]})

        result = evset.map(lambda v, e: v + e.timestamp, receive_extras=True)

        expected = event_set(
            timestamps=[1, 2, 3],
            features={"x": [11, 22, 33]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_output_dtype(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [10, 20, 30]})

        result = evset.map(
            lambda v: "v" + str(v),
            output_dtypes=str,
        )

        expected = event_set(
            timestamps=[1, 2, 3],
            features={"x": ["v10", "v20", "v30"]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_output_dtype_mapping(self):
        evset = event_set(
            timestamps=[1, 2], features={"x": [10, 20], "y": ["100", "200"]}
        )

        def f(v):
            if v.dtype == np.int64:
                return v + 1.0
            return int(v) + 2

        result = evset.map(f, output_dtypes={str: int, int: float})

        expected = event_set(
            timestamps=[1, 2],
            features={"x": [11.0, 21.0], "y": [102, 202]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_output_dtype_feature_mapping(self):
        evset = event_set(
            timestamps=[1, 2], features={"x": [10, 20], "y": ["100", "200"]}
        )

        def f(v):
            if v.dtype == np.int64:
                return v + 1.0
            return int(v) + 2

        result = evset.map(f, output_dtypes={"x": float, "y": int})

        expected = event_set(
            timestamps=[1, 2],
            features={"x": [11.0, 21.0], "y": [102, 202]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_wrong_output_dtype(self):
        evset = event_set(timestamps=[1, 2], features={"x": [10, 20]})

        with self.assertRaisesRegex(
            ValueError,
            (
                "Failed to build array of type int64 with the results of"
                " `func`. Make sure you are specifying the correct"
                " `output_dypes` and returning those types in `func`."
            ),
        ):
            evset.map(lambda x: "v" + str(x))

    def test_too_many_args(self):
        evset = event_set(timestamps=[1, 2], features={"x": [10, 20]})

        with self.assertRaisesRegex(
            TypeError, "missing 1 required positional argument"
        ):
            evset.map(lambda x, e: "v" + str(x))

    def test_too_little_args(self):
        evset = event_set(timestamps=[1, 2], features={"x": [10, 20]})

        with self.assertRaisesRegex(
            TypeError, "takes 1 positional argument but 2 were given"
        ):
            evset.map(lambda x: "v" + str(x), receive_extras=True)

    def test_serialize_fails(self):
        @compile
        def f(e):
            return {"output": e.map(lambda x: x * 2)}

        evset = event_set([])

        with self.assertRaisesRegex(
            ValueError,
            "MAP operator is not serializable.",
        ):
            save(f, "path", evset)


if __name__ == "__main__":
    absltest.main()
