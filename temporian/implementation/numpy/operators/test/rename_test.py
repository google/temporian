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

from temporian.core.operators.rename import RenameOperator
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.rename import (
    RenameNumpyImplementation,
)


class RenameOperatorTest(absltest.TestCase):
    """Rename operator test."""

    def setUp(self):
        self.df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1.0, 0.0],
                ["A", 2.0, np.nan, -2.0, 32.0],
            ],
            columns=["store_id", "timestamp", "sales", "costs", "weather"],
        )

        self.input_evset = EventSet.from_dataframe(
            self.df, index_names=["store_id"]
        )
        self.input_node = self.input_evset.node()

        df = pd.DataFrame(
            [
                ["A", 1.0, "X", -1.0, 0.0],
                ["A", 2.0, "Y", -2.0, 32.0],
            ],
            columns=["store_id", "timestamp", "sales", "costs", "weather"],
        )

        self_input_evset_2 = EventSet.from_dataframe(
            df, index_names=["store_id", "sales"]
        )
        self.input_node_2 = self_input_evset_2.node()

    def test_rename_single_feature_with_str(self) -> None:
        """Test renaming single feature with str."""
        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
            ],
            columns=["timestamp", "sales"],
        )

        self.input_evset = EventSet.from_dataframe(df)
        self.input_node = self.input_evset.node()

        new_df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
            ],
            columns=["timestamp", "costs"],
        )

        expected_evset = EventSet.from_dataframe(new_df)

        operator = RenameOperator(self.input_node, "costs")

        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_single_feature_with_dict(self) -> None:
        """Test renaming single feature with dict."""
        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
            ],
            columns=["timestamp", "sales"],
        )

        self.input_evset = EventSet.from_dataframe(df)
        self.input_node = self.input_evset.node()

        new_df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
            ],
            columns=["timestamp", "costs"],
        )

        expected_evset = EventSet.from_dataframe(new_df)

        operator = RenameOperator(self.input_node, {"sales": "costs"})

        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_multiple_features(self) -> None:
        """Test renaming multiple features."""
        new_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1.0, 0.0],
                ["A", 2.0, np.nan, -2.0, 32.0],
            ],
            columns=["store_id", "timestamp", "new_sales", "costs", "profit"],
        )

        expected_evset = EventSet.from_dataframe(
            new_df, index_names=["store_id"]
        )

        operator = RenameOperator(
            node=self.input_node,
            features={"sales": "new_sales", "weather": "profit"},
        )
        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_single_index_with_str(self) -> None:
        """Test renaming index."""
        new_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1.0, 0.0],
                ["A", 2.0, np.nan, -2.0, 32.0],
            ],
            columns=["product_id", "timestamp", "sales", "costs", "weather"],
        )

        expected_evset = EventSet.from_dataframe(
            new_df, index_names=["product_id"]
        )

        operator = RenameOperator(
            node=self.input_node,
            index="product_id",
        )
        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_single_index_with_dict(self) -> None:
        """Test renaming index."""
        new_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1.0, 0.0],
                ["A", 2.0, np.nan, -2.0, 32.0],
            ],
            columns=["product_id", "timestamp", "sales", "costs", "weather"],
        )

        expected_evset = EventSet.from_dataframe(
            new_df, index_names=["product_id"]
        )

        operator = RenameOperator(
            node=self.input_node,
            index={"store_id": "product_id"},
        )
        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_multiple_indexes(self) -> None:
        """Test renaming multiple indexes."""

        df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1, 0.0],
                ["A", 2.0, np.nan, -2, 32.0],
            ],
            columns=["store_id", "timestamp", "sales", "costs", "weather"],
        )

        self.input_evset = EventSet.from_dataframe(
            df, index_names=["store_id", "costs"]
        )

        self.input_node = self.input_evset.node()

        new_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1, 0.0],
                ["A", 2.0, np.nan, -2, 32.0],
            ],
            columns=["product_id", "timestamp", "sales", "roi", "weather"],
        )

        expected_evset = EventSet.from_dataframe(
            new_df, index_names=["product_id", "roi"]
        )

        operator = RenameOperator(
            node=self.input_node,
            index={"store_id": "product_id", "costs": "roi"},
        )
        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]

        self.assertEqual(renamed_evset, expected_evset)

    def test_rename_feature_with_empty_str(self) -> None:
        """Test renaming feature with empty string."""
        with self.assertRaises(ValueError):
            RenameOperator(node=self.input_node, features={"sales": ""})

    def test_rename_feature_with_empty_str_without_dict(self) -> None:
        """Test renaming feature with empty string."""
        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
            ],
            columns=["timestamp", "sales"],
        )

        self.input_node = EventSet.from_dataframe(df).node()

        with self.assertRaises(ValueError):
            RenameOperator(self.input_node, "")

    def test_rename_feature_with_non_str_object(self) -> None:
        """Test renaming feature with non string object."""
        with self.assertRaises(ValueError):
            RenameOperator(node=self.input_node, features={"sales": 1})

    def test_rename_feature_with_non_existent_feature(self) -> None:
        """Test renaming feature with non existent feature."""
        with self.assertRaises(KeyError):
            RenameOperator(node=self.input_node, features={"sales_1": "costs"})

    def test_rename_feature_with_duplicated_new_feature_names(self) -> None:
        """Test renaming feature with duplicated new names."""
        with self.assertRaises(ValueError):
            RenameOperator(
                node=self.input_node,
                features={"sales": "new_sales", "costs": "new_sales"},
            )

    def test_rename_index_with_empty_str(self) -> None:
        """Test renaming index with empty string."""
        with self.assertRaises(ValueError):
            RenameOperator(node=self.input_node, index={"sales": ""})

    def test_rename_index_with_empty_str_without_dict(self) -> None:
        """Test renaming index with empty string."""
        df = pd.DataFrame(
            [
                [1.0, "A"],
                [2.0, "A"],
            ],
            columns=["timestamp", "sales"],
        )

        self.input_node = EventSet.from_dataframe(
            df, index_names=["sales"]
        ).node()

        with self.assertRaises(ValueError):
            RenameOperator(self.input_node, index="")

    def test_rename_index_with_non_str_object(self) -> None:
        """Test renaming index with non string object."""
        with self.assertRaises(ValueError):
            RenameOperator(node=self.input_node, index={"sales": 1})

    def test_rename_index_with_non_existent_index(self) -> None:
        """Test renaming index with non existent index."""
        with self.assertRaises(KeyError):
            RenameOperator(node=self.input_node, index={"sales_1": "costs"})

    def test_rename_index_with_duplicated_new_index_names(self) -> None:
        """Test renaming index with duplicated new names."""
        with self.assertRaises(ValueError):
            RenameOperator(
                node=self.input_node,
                index={"store_id": "new_sales", "sales": "new_sales"},
            )

    def test_rename_feature_and_index_with_same_name(self) -> None:
        """Test renaming feature and index with same name."""

        operator = RenameOperator(
            node=self.input_node,
            index={"store_id": "sales"},
        )
        impl = RenameNumpyImplementation(operator)

        with self.assertRaises(ValueError):
            impl.call(node=self.input_evset)["node"]

    def test_rename_feature_and_index_inverting_name(self) -> None:
        """Test renaming feature and index with same name complex case."""
        new_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0, -1.0, 0.0],
                ["A", 2.0, np.nan, -2.0, 32.0],
            ],
            columns=["sales", "timestamp", "store_id", "costs", "weather"],
        )

        expected_evset = EventSet.from_dataframe(new_df, index_names=["sales"])

        operator = RenameOperator(
            node=self.input_node,
            features={"sales": "store_id"},
            index={"store_id": "sales"},
        )
        impl = RenameNumpyImplementation(operator)
        renamed_evset = impl.call(node=self.input_evset)["node"]
        self.assertEqual(renamed_evset, expected_evset)


if __name__ == "__main__":
    absltest.main()
