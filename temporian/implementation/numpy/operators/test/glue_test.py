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

from temporian.core.data.node import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.glue import GlueOperator
from temporian.core.data import node as node_lib
from temporian.core.data.dtype import DType

from temporian.implementation.numpy.data.node_0 import EventSet
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
        evset_1 = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_1": [10, 11, 12, 13, 14],
                }
            ),
            index_names=["user_id"],
        )
        evset_2 = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_2": [20, 21, 22, 23, 24],
                    "feature_3": [30, 31, 32, 33, 34],
                }
            ),
            index_names=["user_id"],
        )
        evset_3 = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    **common_data,
                    "feature_4": [40, 41, 42, 43, 44],
                }
            ),
            index_names=["user_id"],
        )
        # set same sampling
        for index_key, index_data in evset_1.data.items():
            evset_2[index_key].timestamps = index_data.timestamps
            evset_3[index_key].timestamps = index_data.timestamps

        expected_evset = EventSet.from_dataframe(
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

        # TODO: Update when "from_dataframe" support the creation of event sets
        # with shared sampling.
        node_1 = evset_1.node()
        node_2 = evset_2.node()
        node_3 = evset_3.node()

        node_2._sampling = node_1._sampling
        node_3._sampling = node_1._sampling

        operator = GlueOperator(
            node_0=node_1,
            node_1=node_2,
            node_2=node_3,
        )
        implementation = GlueNumpyImplementation(operator=operator)
        output = implementation.call(
            node_0=evset_1, node_1=evset_2, node_2=evset_3
        )
        self.assertEqual(
            output["node"],
            expected_evset,
        )

    def test_non_matching_sampling(self):
        with self.assertRaisesRegex(
            ValueError,
            "All glue arguments should have the same sampling.",
        ):
            _ = GlueOperator(
                node_0=node_lib.input_node(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=Sampling(index_levels=[("x", DType.INT64)]),
                ),
                node_1=node_lib.input_node(
                    [Feature(name="b", dtype=DType.FLOAT64)],
                    sampling=Sampling(index_levels=[("x", DType.INT64)]),
                ),
            )

    def test_duplicate_feature(self):
        with self.assertRaisesRegex(
            ValueError,
            'Feature "a" is defined in multiple input nodes',
        ):
            sampling = Sampling(index_levels=[("x", DType.INT64)])
            _ = GlueOperator(
                node_0=node_lib.input_node(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=sampling,
                ),
                node_1=node_lib.input_node(
                    [Feature(name="a", dtype=DType.FLOAT64)],
                    sampling=sampling,
                ),
            )


if __name__ == "__main__":
    absltest.main()
