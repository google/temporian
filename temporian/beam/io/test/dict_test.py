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

from typing import Any, Dict

import numpy as np
from apache_beam.testing import util
import apache_beam as beam
from absl.testing import absltest
from absl import flags
from apache_beam.testing.test_pipeline import TestPipeline
from temporian.beam.io.dict import (
    to_event_set,
    to_dict,
)
from temporian.implementation.numpy.data.io import Schema
from temporian.core.data.dtype import DType


def structure_np_to_list(data):
    """
    Apply a function to a recursive structure of dict and list.

    Args:
      data: The data to apply the function to.

    Returns:
      The data with the function applied.
    """

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, dict):
        return {key: structure_np_to_list(value) for key, value in data.items()}
    if isinstance(data, list):
        return [structure_np_to_list(item) for item in data]

    if isinstance(data, (int, float, str, bytes)):
        return data

    raise ValueError(f"Non supported type {type(data)}")


class IOTest(absltest.TestCase):
    def test_to_event_set_and_to_dict_singleEvents(self):
        schema = Schema(
            [("f1", DType.INT32), ("f2", DType.STRING)],
            [("i1", DType.INT32), ("i2", DType.STRING)],
        )

        raw_data = [
            {"timestamp": 100.0, "f1": 1, "f2": b"a", "i1": 10, "i2": b"x"},
            {"timestamp": 101.0, "f1": 2, "f2": b"b", "i1": 10, "i2": b"x"},
            {"timestamp": 102.0, "f1": 3, "f2": b"c", "i1": 10, "i2": b"y"},
            {"timestamp": 103.0, "f1": 4, "f2": b"d", "i1": 11, "i2": b"y"},
            {"timestamp": 104.0, "f1": 5, "f2": b"e", "i1": 11, "i2": b"y"},
        ]

        with TestPipeline() as p:
            output = (
                p
                | beam.Create(raw_data)
                | to_event_set(schema, grouped_by_index=False)
                | to_dict(schema, grouped_by_index=False)
            )
            util.assert_that(output, util.equal_to(raw_data))

    def test_to_event_set_and_to_dict_singleEvents_no_features(self):
        schema = Schema(
            [],
            [("i1", DType.INT32), ("i2", DType.STRING)],
        )

        raw_data = [
            {"timestamp": 100.0, "i1": 10, "i2": b"x"},
            {"timestamp": 101.0, "i1": 10, "i2": b"x"},
            {"timestamp": 102.0, "i1": 10, "i2": b"y"},
            {"timestamp": 103.0, "i1": 11, "i2": b"y"},
            {"timestamp": 104.0, "i1": 11, "i2": b"y"},
        ]

        with TestPipeline() as p:
            output = (
                p
                | beam.Create(raw_data)
                | to_event_set(schema, grouped_by_index=False)
                | to_dict(schema, grouped_by_index=False)
            )
            util.assert_that(output, util.equal_to(raw_data))

    def test_to_event_set_and_to_dict_singleEvents_errors(self):
        def test(
            schema: Schema,
            data: Dict[str, Any],
            regexp_error: str,
            exception: Exception = ValueError,
        ):
            with self.assertRaisesRegex(exception, regexp_error):
                with TestPipeline() as p:
                    _ = (
                        p
                        | beam.Create([data])
                        | to_event_set(schema, grouped_by_index=False)
                    )

        test(
            Schema([("f1", DType.INT32)], [("i1", DType.INT64)]),
            {"timestamp": 1.0, "f1": "CCC", "i1": 10},
            "invalid literal",
        )

        test(
            Schema([("f1", DType.INT32)], [("i1", DType.INT64)]),
            {"timestamp": 1.0, "f1": 1, "i1": "AAA"},
            "invalid literal",
        )

        test(
            Schema([("f1", DType.INT32)], [("i1", DType.INT64)]),
            {"timestamp": "BBB", "f1": 1, "i1": 10},
            "could not convert string to float",
        )

    def test_to_event_set_and_to_dict_indexedEvents(self):
        schema = Schema(
            features=[("f1", DType.INT64), ("f2", DType.STRING)],
            indexes=[("i1", DType.INT64), ("i2", DType.STRING)],
        )

        raw_data = [
            {
                "timestamp": np.array([100.0, 101.0, 102.0]),
                "f1": np.array([1, 2, 3]),
                "f2": np.array([b"a", b"b", b"c"]),
                "i1": 10,
                "i2": b"x",
            },
            {
                "timestamp": np.array([103.0]),
                "f1": np.array([4]),
                "f2": np.array([b"d"]),
                "i1": 10,
                "i2": b"y",
            },
            {
                "timestamp": np.array([104.0]),
                "f1": np.array([5]),
                "f2": np.array([b"e"]),
                "i1": 11,
                "i2": b"x",
            },
        ]

        with TestPipeline() as p:
            output = (
                p
                | beam.Create(raw_data)
                | to_event_set(schema, grouped_by_index=True)
                | to_dict(schema, grouped_by_index=True)
                | beam.Map(structure_np_to_list)
            )
            util.assert_that(
                output, util.equal_to(structure_np_to_list(raw_data))
            )

    def test_to_event_set_and_to_dict_indexedEvents_no_features(self):
        schema = Schema(
            features=[],
            indexes=[("i1", DType.INT64), ("i2", DType.STRING)],
        )

        raw_data = [
            {
                "timestamp": np.array([100.0, 101.0, 102.0]),
                "i1": 10,
                "i2": b"x",
            },
            {
                "timestamp": np.array([103.0]),
                "i1": 10,
                "i2": b"y",
            },
            {
                "timestamp": np.array([104.0]),
                "i1": 11,
                "i2": b"x",
            },
        ]

        with TestPipeline() as p:
            output = (
                p
                | beam.Create(raw_data)
                | to_event_set(schema, grouped_by_index=True)
                | to_dict(schema, grouped_by_index=True)
                | beam.Map(structure_np_to_list)
            )
            util.assert_that(
                output, util.equal_to(structure_np_to_list(raw_data))
            )

    def test_to_event_set_and_to_dict_indexedEvents_errors(self):
        def test(
            schema: Schema,
            data: Dict[str, Any],
            regexp_error: str,
            exception: Exception = ValueError,
        ):
            with self.assertRaisesRegex(exception, regexp_error):
                with TestPipeline() as p:
                    _ = (
                        p
                        | beam.Create([data])
                        | to_event_set(schema, grouped_by_index=True)
                    )

        test(
            Schema([("f1", DType.INT64)], [("i1", DType.INT64)]),
            {
                "timestamp": np.array([100.0, 101.0]),
                "f1": np.array([1, 2], np.int32),  # Wrong
                "i1": 10,
            },
            "expected to by a numpy array with dtype",
        )

        test(
            Schema([("f1", DType.INT64)], [("i1", DType.INT64)]),
            {
                "timestamp": np.array([100.0, 101.0]),
                "f1": np.array([1, 2]),
                "i1": np.array([10]),  # Wrong
            },
            "is expected to be of type <class 'int'>",
        )

        test(
            Schema([("f1", DType.INT64)], [("i1", DType.INT64)]),
            {
                "timestamp": [100.0, 101.0],  # Wrong
                "f1": np.array([1, 2]),
                "i1": 10,
            },
            "is expected to be np.float64 numpy array",
        )

        test(
            Schema([("f1", DType.INT64)], [("i1", DType.INT64)]),
            {
                "timestamp": np.array([100.0, 101.0]),
                "i1": 10,
            },
            "f1",
            KeyError,
        )

        test(
            Schema([("f1", DType.INT64)], [("i1", DType.INT64)]),
            {
                "timestamp": np.array([100.0, 101.0]),
                "f1": np.array([1, 2]),
                "i1_wrong_spelling": 10,
            },
            "i1",
            KeyError,
        )


if __name__ == "__main__":
    absltest.main()
