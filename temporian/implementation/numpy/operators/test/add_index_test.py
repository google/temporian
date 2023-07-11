from typing import List, Any, Tuple
from dataclasses import dataclass

from absl.testing import absltest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from temporian.core.operators.add_index import (
    add_index,
    set_index,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.add_index import (
    AddIndexNumpyImplementation,
    operators_cc,
)
from temporian.io.pandas import from_pandas
from temporian.core.evaluation import run
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)

cc_add_index = operators_cc.add_index_compute_index


@dataclass
class SortedGroup:
    """A single group.

    Groups are intermediate results during the computation of a new index.
    """

    key: Tuple[Any]
    row_idxs: List[int]


def sort_groups(group_keys, row_idxs, group_begin_idx) -> List[SortedGroup]:
    """Sorts groups for deterministic comparison."""

    result = []
    for group_idx, group_key in enumerate(group_keys):
        result.append(
            SortedGroup(
                key=group_key,
                row_idxs=row_idxs[
                    group_begin_idx[group_idx] : group_begin_idx[group_idx + 1]
                ].tolist(),
            )
        )
    result.sort(key=lambda x: x.key)
    return result


class AddIndexNumpyImplementationTest(absltest.TestCase):
    def setUp(self) -> None:
        self.input_evset = from_pandas(
            pd.DataFrame(
                [
                    ["A", 0, 1, 0.4, 10.0],
                    ["A", 0, 1, 0.5, 11.0],
                    ["A", 0, 1, 0.6, 12.0],
                    ["B", 0, 2, 0.1, 13.0],
                    ["B", 0, 2, 0.2, 14.0],
                    ["B", 1, 2, 0.3, 15.0],
                    ["B", 1, 2, 0.4, 16.0],
                    ["B", 1, 3, 0.3, 17.0],
                ],
                columns=[
                    "state_id",
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                ],
            ),
            indexes=["state_id"],
        )

        self.input_node = self.input_evset.node()

    def test_cc_empty(self):
        data = [np.array([], np.int64)]
        group_keys, row_idxs, group_begin_idx = cc_add_index(data)

        self.assertEqual(group_keys, [])
        assert_array_equal(row_idxs, np.array([], np.int64))
        assert_array_equal(group_begin_idx, np.array([0], np.int64))

    def test_cc_integer(self):
        data = [
            np.array([1, 1, 1, 2, 2, 3], np.int64),
            np.array([5, 5, 7, 1, 1, 4], np.int64),
        ]
        group_keys, row_idxs, group_begin_idx = cc_add_index(data)

        print("group_keys:", group_keys, flush=True)
        print("row_idxs:", row_idxs, flush=True)
        print("group_begin_idx:", group_begin_idx, flush=True)

        self.assertEqual(
            sort_groups(group_keys, row_idxs, group_begin_idx),
            [
                SortedGroup((1, 5), [0, 1]),
                SortedGroup((1, 7), [2]),
                SortedGroup((2, 1), [3, 4]),
                SortedGroup((3, 4), [5]),
            ],
        )

    def test_cc_string(self):
        data = [
            np.array(["A", "A", "A", "B", "B", "C"], np.bytes_),
            np.array(["X", "X", "Y", "X", "X", "Z"], np.bytes_),
        ]
        group_keys, row_idxs, group_begin_idx = cc_add_index(data)

        print("group_keys:", group_keys, flush=True)
        print("row_idxs:", row_idxs, flush=True)
        print("group_begin_idx:", group_begin_idx, flush=True)

        self.assertEqual(
            sort_groups(group_keys, row_idxs, group_begin_idx),
            [],
        )

    def test_add_index_single(self) -> None:
        expected_evset = from_pandas(
            pd.DataFrame(
                [
                    ["A", 0, 1, 0.4, 10.0],
                    ["A", 0, 1, 0.5, 11.0],
                    ["A", 0, 1, 0.6, 12.0],
                    ["B", 0, 2, 0.1, 13.0],
                    ["B", 0, 2, 0.2, 14.0],
                    ["B", 1, 2, 0.3, 15.0],
                    ["B", 1, 3, 0.3, 17.0],
                    ["B", 1, 2, 0.4, 16.0],
                ],
                columns=[
                    "state_id",
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                ],
            ),
            indexes=["state_id", "store_id"],
        )
        output = add_index(self.input_node, "store_id")
        operator_impl = AddIndexNumpyImplementation(output.creator)
        output_evset = operator_impl.call(input=self.input_evset)["output"]

        assertEqualEventSet(self, output_evset, expected_evset)

    def test_add_index_multiple(self) -> None:
        expected_evset = from_pandas(
            pd.DataFrame(
                [
                    ["A", 0, 1, 0.4, 10.0],
                    ["A", 0, 1, 0.5, 11.0],
                    ["A", 0, 1, 0.6, 12.0],
                    ["B", 0, 2, 0.1, 13.0],
                    ["B", 0, 2, 0.2, 14.0],
                    ["B", 1, 2, 0.3, 15.0],
                    ["B", 1, 3, 0.3, 17.0],
                    ["B", 1, 2, 0.4, 16.0],
                ],
                columns=[
                    "state_id",
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                ],
            ),
            indexes=["state_id", "store_id", "item_id"],
        )
        output = add_index(self.input_node, ["store_id", "item_id"])
        # instance operator implementation
        operator_impl = AddIndexNumpyImplementation(output.creator)

        # call operator
        output_evset = operator_impl.call(input=self.input_evset)["output"]

        # validate output
        assertEqualEventSet(self, output_evset, expected_evset)

    def test_set_index_single(self) -> None:
        expected_evset = from_pandas(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0, "B"],
                    [0, 2, 0.2, 14.0, "B"],
                    [0, 1, 0.4, 10.0, "A"],
                    [0, 1, 0.5, 11.0, "A"],
                    [0, 1, 0.6, 12.0, "A"],
                    [1, 2, 0.3, 15.0, "B"],
                    [1, 3, 0.3, 17.0, "B"],
                    [1, 2, 0.4, 16.0, "B"],
                ],
                columns=[
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                    "state_id",
                ],
            ),
            indexes=["store_id"],
        )
        output = set_index(self.input_node, ["store_id"])
        output_evset = run(
            output, {self.input_node: self.input_evset}, check_execution=True
        )

        assertEqualEventSet(self, output_evset, expected_evset)

    def test_set_index_multiple(self) -> None:
        expected_evset = from_pandas(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0, "B"],
                    [0, 2, 0.2, 14.0, "B"],
                    [0, 1, 0.4, 10.0, "A"],
                    [0, 1, 0.5, 11.0, "A"],
                    [0, 1, 0.6, 12.0, "A"],
                    [1, 2, 0.3, 15.0, "B"],
                    [1, 3, 0.3, 17.0, "B"],
                    [1, 2, 0.4, 16.0, "B"],
                ],
                columns=[
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                    "state_id",
                ],
            ),
            indexes=["store_id", "item_id"],
        )
        output = set_index(
            self.input_node,
            ["store_id", "item_id"],
        )
        output_evset = run(
            output, {self.input_node: self.input_evset}, check_execution=True
        )

        assertEqualEventSet(self, output_evset, expected_evset)

    def test_set_index_multiple_change_order(self) -> None:
        common = {"features": {"a": [], "b": [], "c": []}, "timestamps": []}

        evset_abc = event_set(**common, indexes=["a", "b", "c"])
        evset_acb = event_set(**common, indexes=["a", "c", "b"])
        evset_cba = event_set(**common, indexes=["c", "b", "a"])
        evset_cab = event_set(**common, indexes=["c", "a", "b"])

        def my_run(src_evset, new_index, expected_evset):
            output = set_index(src_evset.node(), new_index)
            output_evset = run(
                output, {src_evset.node(): src_evset}, check_execution=True
            )
            assertEqualEventSet(self, output_evset, expected_evset)

        my_run(evset_abc, ["a", "b", "c"], evset_abc)
        my_run(evset_abc, ["a", "c", "b"], evset_acb)
        my_run(evset_abc, ["c", "b", "a"], evset_cba)
        my_run(evset_abc, ["c", "a", "b"], evset_cab)
        my_run(evset_cba, ["a", "b", "c"], evset_abc)


if __name__ == "__main__":
    absltest.main()
