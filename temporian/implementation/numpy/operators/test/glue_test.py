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

from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.glue import GlueOperator
from temporian.core.data import event as event_lib
from temporian.core.data import dtype as dtype_lib

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators.glue import (
    GlueNumpyImplementation,
)
from temporian.implementation.numpy.evaluator import run_with_check


class GlueNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of glue operator."""

    def setUp(self) -> None:
        pass

    def test_base(self):
        common_data = {
            "timestamp": np.array([1, 1, 2, 3, 4]),
            "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
        }

        event_1_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_1": [10, 11, 12, 13, 14],
                }
            ),
            index_names=["user_id"],
        )

        event_2_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_2": [20, 21, 22, 23, 24],
                    "feature_3": [30, 31, 32, 33, 34],
                }
            ),
            index_names=["user_id"],
        )

        event_3_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_4": [40, 41, 42, 43, 44],
                }
            ),
            index_names=["user_id"],
        )

        expected_output_data = NumpyEvent.from_dataframe(
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
        event_1 = event_1_data.schema()
        event_2 = event_2_data.schema()
        event_3 = event_3_data.schema()

        event_2._sampling = event_1._sampling
        event_3._sampling = event_1._sampling
        event_2_data.sampling = event_1_data.sampling
        event_3_data.sampling = event_1_data.sampling

        operator = GlueOperator(
            event_0=event_1,
            event_1=event_2,
            event_2=event_3,
        )
        implementation = GlueNumpyImplementation(operator=operator)
        output = run_with_check(
            operator,
            implementation,
            {
                "event_1": event_1_data,
                "event_2": event_2_data,
                "event_3": event_3_data,
            },
        )

        self.assertEqual(
            output["event"],
            expected_output_data,
        )

    def test_non_matching_sampling(self):
        with self.assertRaisesRegex(
            ValueError,
            "All the events do not have the same sampling.",
        ):
            _ = GlueOperator(
                event_0=event_lib.input_event(
                    [Feature(name="a", dtype=dtype_lib.FLOAT64)],
                    sampling=Sampling(index=["x"]),
                ),
                event_1=event_lib.input_event(
                    [Feature(name="b", dtype=dtype_lib.FLOAT64)],
                    sampling=Sampling(index=["x"]),
                ),
            )

    def test_duplicate_feature(self):
        with self.assertRaisesRegex(
            ValueError,
            "Feature a is defined in multiple input events",
        ):
            sampling = Sampling(index=["x"])
            _ = GlueOperator(
                event_0=event_lib.input_event(
                    [Feature(name="a", dtype=dtype_lib.FLOAT64)],
                    sampling=sampling,
                ),
                event_1=event_lib.input_event(
                    [Feature(name="a", dtype=dtype_lib.FLOAT64)],
                    sampling=sampling,
                ),
            )


if __name__ == "__main__":
    absltest.main()
