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

from temporian.core.operators.glue import GlueOperator, glue
from temporian.core.data.node import input_node
from temporian.core.data.dtype import DType
from temporian.implementation.numpy.operators.glue import (
    GlueNumpyImplementation,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class GlueNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of glue operator."""

    def setUp(self) -> None:
        pass

    def test_manual_nodes(self):
        evset_1 = event_set(
            timestamps=[1, 1, 2, 3, 4],
            features={
                "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
                "feature_1": [10, 11, 12, 13, 14],
            },
            indexes=["user_id"],
        )

        evset_2 = event_set(
            timestamps=[1, 1, 2, 3, 4],
            features={
                "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
                "feature_2": [20, 21, 22, 23, 24],
                "feature_3": [30, 31, 32, 33, 34],
            },
            indexes=["user_id"],
            same_sampling_as=evset_1,
        )

        evset_3 = event_set(
            timestamps=[1, 1, 2, 3, 4],
            features={
                "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
                "feature_4": [40, 41, 42, 43, 44],
            },
            indexes=["user_id"],
            same_sampling_as=evset_1,
        )

        expected_evset = event_set(
            timestamps=[1, 1, 2, 3, 4],
            features={
                "user_id": ["user_1", "user_1", "user_1", "user_1", "user_2"],
                "feature_1": [10, 11, 12, 13, 14],
                "feature_2": [20, 21, 22, 23, 24],
                "feature_3": [30, 31, 32, 33, 34],
                "feature_4": [40, 41, 42, 43, 44],
            },
            indexes=["user_id"],
        )

        operator = GlueOperator(
            input_0=evset_1.node(),
            input_1=evset_2.node(),
            input_2=evset_3.node(),
        )
        operator.outputs["output"].check_same_sampling(evset_1.node())
        operator.outputs["output"].check_same_sampling(evset_2.node())
        operator.outputs["output"].check_same_sampling(evset_3.node())

        implementation = GlueNumpyImplementation(operator=operator)
        testOperatorAndImp(self, operator, implementation)
        output = implementation.call(
            input_0=evset_1, input_1=evset_2, input_2=evset_3
        )["output"]
        assertEqualEventSet(self, output, expected_evset)

    def test_non_matching_sampling(self):
        with self.assertRaisesRegex(
            ValueError,
            "Arguments should have the same sampling.",
        ):
            n1 = input_node(
                features=[("a", DType.FLOAT64)], indexes=[("x", DType.STRING)]
            )
            n2 = input_node(
                features=[("b", DType.FLOAT64)], indexes=[("x", DType.STRING)]
            )
            _ = GlueOperator(input_0=n1, input_1=n2)

    def test_duplicate_feature(self):
        with self.assertRaisesRegex(
            ValueError,
            'Feature "a" is defined in multiple input EventSetNodes',
        ):
            n1 = input_node(
                features=[("a", DType.FLOAT64)], indexes=[("x", DType.STRING)]
            )
            n2 = input_node(
                features=[("a", DType.FLOAT64)], same_sampling_as=n1
            )
            _ = GlueOperator(input_0=n1, input_1=n2)

    def test_order_unchanged(self):
        """Tests that input evsets' order is kept.

        Regression test for failing case where glue misordered its inputs when
        more than 10, because of sorted() being called over a list where
        "input_10"  was interpreted as less than "input_2".
        """
        evset_0 = event_set(
            timestamps=[1],
            features={
                "f0": np.asarray([1], dtype=np.int64),
            },
        )
        evset_1 = evset_0.rename("f1")
        evset_2 = evset_0.rename("f2")
        evset_3 = evset_0.rename("f3")
        evset_4 = evset_0.rename("f4")
        evset_5 = evset_0.rename("f5")
        evset_6 = evset_0.rename("f6")
        evset_7 = evset_0.rename("f7")
        evset_8 = evset_0.rename("f8")
        evset_9 = evset_0.rename("f9")

        evset_10 = event_set(
            timestamps=[1],
            features={
                "f10": np.asarray([1], dtype=np.int32),
            },
            same_sampling_as=evset_0,
        )

        glue(
            evset_0,
            evset_1,
            evset_2,
            evset_3,
            evset_4,
            evset_5,
            evset_6,
            evset_7,
            evset_8,
            evset_9,
            evset_10,
        )


if __name__ == "__main__":
    absltest.main()
