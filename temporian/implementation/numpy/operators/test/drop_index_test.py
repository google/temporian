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

from temporian.core.operators.drop_index import DropIndexOperator
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.drop_index import (
    DropIndexNumpyImplementation,
)
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class DropIndexNumpyImplementationTest(absltest.TestCase):
    def setUp(self) -> None:
        self.input_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1, 0.4, 10.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [1, 2, 0.3, 15.0],
                    [1, 2, 0.4, 16.0],
                    [1, 3, 0.3, 17.0],
                ],
                columns=["store_id", "item_id", "timestamp", "sales"],
            ),
            index_names=["store_id", "item_id"],
        )
        self.input_node = self.input_evset.node()

    def test_drop_all(self) -> None:
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0.1, 13.0, 0, 2],
                    [0.2, 14.0, 0, 2],
                    [0.3, 15.0, 1, 2],
                    [0.3, 17.0, 1, 3],
                    [0.4, 16.0, 1, 2],
                    [0.4, 10.0, 0, 1],
                    [0.5, 11.0, 0, 1],
                    [0.6, 12.0, 0, 1],
                ],
                columns=["timestamp", "sales", "store_id", "item_id"],
            ),
            index_names=[],
        )

        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id", "item_id"], keep=True
        )
        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=self.input_evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_drop_item_id(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 0.1, 13.0, 2],
                    [0, 0.2, 14.0, 2],
                    [1, 0.3, 15.0, 2],
                    [1, 0.3, 17.0, 3],
                    [0, 0.4, 10.0, 1],
                    [1, 0.4, 16.0, 2],
                    [0, 0.5, 11.0, 1],
                    [0, 0.6, 12.0, 1],
                ],
                columns=["store_id", "timestamp", "sales", "item_id"],
            ),
            index_names=["store_id"],
        )

        operator = DropIndexOperator(
            self.input_node, index_to_drop=["item_id"], keep=True
        )
        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=self.input_evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_drop_store_id(self) -> None:
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [2, 0.1, 13.0, 0],
                    [2, 0.2, 14.0, 0],
                    [2, 0.3, 15.0, 1],
                    [3, 0.3, 17.0, 1],
                    [1, 0.4, 10.0, 0],
                    [2, 0.4, 16.0, 1],
                    [1, 0.5, 11.0, 0],
                    [1, 0.6, 12.0, 0],
                ],
                columns=["item_id", "timestamp", "sales", "store_id"],
            ),
            index_names=["item_id"],
        )

        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id"], keep=True
        )

        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=self.input_evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_drop_item_id_keep_false(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 0.1, 13.0],
                    [0, 0.2, 14.0],
                    [1, 0.3, 15.0],
                    [1, 0.3, 17.0],
                    [0, 0.4, 10.0],
                    [1, 0.4, 16.0],
                    [0, 0.5, 11.0],
                    [0, 0.6, 12.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = DropIndexOperator(
            self.input_node, index_to_drop=["item_id"], keep=False
        )

        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=self.input_evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_drop_store_id_keep_false(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [2, 0.1, 13.0],
                    [2, 0.2, 14.0],
                    [2, 0.3, 15.0],
                    [3, 0.3, 17.0],
                    [1, 0.4, 10.0],
                    [2, 0.4, 16.0],
                    [1, 0.5, 11.0],
                    [1, 0.6, 12.0],
                ],
                columns=["item_id", "timestamp", "sales"],
            ),
            index_names=["item_id"],
        )

        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id"], keep=False
        )
        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=self.input_evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_str_index(self):
        evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 2, 3],
                    "a": [1, 2, 3, 4],
                    "d": ["D1", "D2", "D3", "D4"],
                    "b": ["B1", "B1", "B2", "B2"],
                    "c": ["C1", "C2", "C1", "C2"],
                }
            ),
            index_names=["b", "c"],
        )
        node = evset.node()

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 2, 3],
                    "a": [1, 2, 3, 4],
                    "d": ["D1", "D2", "D3", "D4"],
                    "b": ["B1", "B1", "B2", "B2"],
                    "c": ["C1", "C2", "C1", "C2"],
                }
            ),
            index_names=["c"],
        )

        # Run op
        operator = DropIndexOperator(input=node, index_to_drop=["b"], keep=True)
        operator_impl = DropIndexNumpyImplementation(operator)
        testOperatorAndImp(self, operator, operator_impl)
        output = operator_impl.__call__(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
