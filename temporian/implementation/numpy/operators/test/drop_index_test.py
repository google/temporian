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


class DropIndexNumpyImplementationTest(absltest.TestCase):
    def setUp(self) -> None:
        self.input_evset = EventSet.from_dataframe(
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
        expected_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [1, 2, 0.3, 15.0],
                    [1, 3, 0.3, 17.0],
                    [0, 1, 0.4, 10.0],
                    [1, 2, 0.4, 16.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                ],
                columns=["store_id", "item_id", "timestamp", "sales"],
            ),
            index_names=[],
        )
        # instance core operator
        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id", "item_id"], keep=True
        )
        # instance operator implementation
        operator_impl = DropIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.__call__(evset=self.input_evset)[
            "node"
        ]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_evset)

    def test_drop_item_id(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [1, 2, 0.3, 15.0],
                    [1, 3, 0.3, 17.0],
                    [0, 1, 0.4, 10.0],
                    [1, 2, 0.4, 16.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                ],
                columns=["store_id", "item_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )
        # instance core operator
        operator = DropIndexOperator(
            self.input_node, index_to_drop=["item_id"], keep=True
        )
        # instance operator implementation
        operator_impl = DropIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(node=self.input_evset)["node"]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_evset)

    def test_drop_store_id(self) -> None:
        expected_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [1, 2, 0.3, 15.0],
                    [1, 3, 0.3, 17.0],
                    [0, 1, 0.4, 10.0],
                    [1, 2, 0.4, 16.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                ],
                columns=["store_id", "item_id", "timestamp", "sales"],
            ),
            index_names=["item_id"],
        )
        # instance core operator
        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id"], keep=True
        )
        # instance operator implementation
        operator_impl = DropIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(node=self.input_evset)["node"]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_evset)

    def test_drop_item_id_keep_false(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_evset = EventSet.from_dataframe(
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
        # instance core operator
        operator = DropIndexOperator(
            self.input_node, index_to_drop=["item_id"], keep=False
        )
        # instance operator implementation
        operator_impl = DropIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(node=self.input_evset)["node"]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_evset)

    def test_drop_store_id_keep_false(self) -> None:
        # Need to do some re-ordering due to timestamp collisions in sort
        expected_evset = EventSet.from_dataframe(
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
        # instance core operator
        operator = DropIndexOperator(
            self.input_node, index_to_drop=["store_id"], keep=False
        )
        # instance operator implementation
        operator_impl = DropIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(node=self.input_evset)["node"]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_evset)

    def test_str_index(self):
        evset = EventSet.from_dataframe(
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

        expected_output = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 2, 3],
                    "b": ["B1", "B1", "B2", "B2"],
                    "a": [1, 2, 3, 4],
                    "d": ["D1", "D2", "D3", "D4"],
                    "c": ["C1", "C2", "C1", "C2"],
                }
            ),
            index_names=["c"],
        )

        # Run op
        op = DropIndexOperator(node=node, index_to_drop=["b"], keep=True)
        instance = DropIndexNumpyImplementation(op)
        output = instance.call(node=evset)["node"]
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
