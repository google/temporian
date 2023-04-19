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

import numpy as np
import pandas as pd

from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.glue import GlueOperator
from temporian.core.data import event as event_lib
from temporian.core.data.dtype import DType

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.glue import (
    GlueNumpyImplementation,
)


class GlueNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of glue operator."""

    def setUp(self) -> None:
        pass

    def test_base(self):
        common_data = {
            "timestamp": np.array([1, 1, 2, 3, 4]),
            "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
        }
        numpy_event_1 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_1": [10, 11, 12, 13, 14],
                }
            ),
            index_names=["user_id"],
        )
        numpy_event_2 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_2": [20, 21, 22, 23, 24],
                    "feature_3": [30, 31, 32, 33, 34],
                }
            ),
            index_names=["user_id"],
        )
        numpy_event_3 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_4": [40, 41, 42, 43, 44],
                }
            ),
            index_names=["user_id"],
        )
        # set same sampling
        for index_key, index_data in numpy_event_1.data.items():
            numpy_event_2[index_key].timestamps = index_data.timestamps
            numpy_event_3[index_key].timestamps = index_data.timestamps

        expected_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_1": [10, 11, 12, 13, 14],
                    "feature_2": [20, 21, 22, 23, 24],
                    "feature_3": [30, 31, 32, 33, 34],
                    "feature_4": [40, 41, 42, 43, 44],
                }
            ),
            index_names=["user_id"],
        )

        # TODO: Update when "from_dataframe" support the creation of events
        # with shared sampling.
        event_1 = numpy_event_1.schema()
        event_2 = numpy_event_2.schema()
        event_3 = numpy_event_3.schema()

        event_2._sampling = event_1._sampling
        event_3._sampling = event_1._sampling

        operator = GlueOperator(
            event_0=event_1,
            event_1=event_2,
            event_2=event_3,
        )
        implementation = GlueNumpyImplementation(operator=operator)
        output = implementation.call(
            event_0=numpy_event_1, event_1=numpy_event_2, event_2=numpy_event_3
        )
        self.assertEqual(
            output["event"],
            expected_numpy_output_event,
        )

    def test_non_matching_sampling(self):
        with self.assertRaisesRegex(
            ValueError,
            "All glue arguments should have the same sampling.",
        ):
            _ = GlueOperator(
                event_0=event_lib.input_event(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=Sampling(index_levels=[("x", DType.INT64)]),
                ),
                event_1=event_lib.input_event(
                    [Feature(name="b", dtype=DType.FLOAT64)],
                    sampling=Sampling(index_levels=[("x", DType.INT64)]),
                ),
            )

    def test_duplicate_feature(self):
        with self.assertRaisesRegex(
            ValueError,
            'Feature "a" is defined in multiple input events',
        ):
            sampling = Sampling(index_levels=[("x", DType.INT64)])
            _ = GlueOperator(
                event_0=event_lib.input_event(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=sampling,
                ),
                event_1=event_lib.input_event(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=sampling,
                ),
            )


if __name__ == "__main__":
    absltest.main()
