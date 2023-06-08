from absl.testing import absltest

import pandas as pd

from temporian.core.operators.add_index import (
    add_index,
    set_index,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.add_index import (
    AddIndexNumpyImplementation,
)
from temporian.io.pandas import from_pandas
from temporian.core.evaluation import evaluate
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)


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
            index_names=["state_id"],
        )

        self.input_node = self.input_evset.node()

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
            index_names=["state_id", "store_id"],
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
            index_names=["state_id", "store_id", "item_id"],
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
            index_names=["store_id"],
        )
        output = set_index(self.input_node, ["store_id"])
        output_evset = evaluate(
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
            index_names=["store_id", "item_id"],
        )
        output = set_index(
            self.input_node,
            ["store_id", "item_id"],
        )
        output_evset = evaluate(
            output, {self.input_node: self.input_evset}, check_execution=True
        )

        assertEqualEventSet(self, output_evset, expected_evset)

    def test_set_index_multiple_change_order(self) -> None:
        common = {"features": {"a": [], "b": [], "c": []}, "timestamps": []}

        evset_abc = event_set(**common, index_features=["a", "b", "c"])
        evset_acb = event_set(**common, index_features=["a", "c", "b"])
        evset_cba = event_set(**common, index_features=["c", "b", "a"])
        evset_cab = event_set(**common, index_features=["c", "a", "b"])

        def run(src_evset, new_index, expected_evset):
            output = set_index(src_evset.node(), new_index)
            output_evset = evaluate(
                output, {src_evset.node(): src_evset}, check_execution=True
            )
            assertEqualEventSet(self, output_evset, expected_evset)

        run(evset_abc, ["a", "b", "c"], evset_abc)
        run(evset_abc, ["a", "c", "b"], evset_acb)
        run(evset_abc, ["c", "b", "a"], evset_cba)
        run(evset_abc, ["c", "a", "b"], evset_cab)
        run(evset_cba, ["a", "b", "c"], evset_abc)


if __name__ == "__main__":
    absltest.main()
