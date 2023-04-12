from absl.testing import absltest

import pandas as pd

from temporian.core.operators.set_index import SetIndexOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.set_index import (
    SetIndexNumpyImplementation,
)


class SetIndexNumpyImplementationTest(absltest.TestCase):
    def setUp(self) -> None:
        # input NumPy event
        self.numpy_input_evt = NumpyEvent.from_dataframe(
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
        # input event
        self.input_evt = self.numpy_input_evt.schema()

    def test_append_single(self) -> None:
        # output NumPy event
        expected_numpy_output_evt = NumpyEvent.from_dataframe(
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
        operator = SetIndexOperator(
            self.input_evt, feature_names="store_id", append=True
        )
        # instance operator implementation
        operator_impl = SetIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(event=self.numpy_input_evt)[
            "event"
        ]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_numpy_output_evt)

    def test_append_multiple(self) -> None:
        # output NumPy event
        expected_numpy_output_evt = NumpyEvent.from_dataframe(
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
        operator = SetIndexOperator(
            self.input_evt,
            feature_names=["store_id", "item_id"],
            append=True,
        )
        # instance operator implementation
        operator_impl = SetIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(event=self.numpy_input_evt)[
            "event"
        ]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_numpy_output_evt)

    def test_set_single(self) -> None:
        # output NumPy event
        expected_numpy_output_evt = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [0, 1, 0.4, 10.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                    [1, 2, 0.3, 15.0],
                    [1, 3, 0.3, 17.0],
                    [1, 2, 0.4, 16.0],
                ],
                columns=[
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                ],
            ),
            index_names=["store_id"],
        )
        operator = SetIndexOperator(
            self.input_evt, feature_names=["store_id"], append=False
        )
        # instance operator implementation
        operator_impl = SetIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(event=self.numpy_input_evt)[
            "event"
        ]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_numpy_output_evt)

    def test_set_multiple(self) -> None:
        # output NumPy event
        expected_numpy_output_evt = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2, 0.1, 13.0],
                    [0, 2, 0.2, 14.0],
                    [0, 1, 0.4, 10.0],
                    [0, 1, 0.5, 11.0],
                    [0, 1, 0.6, 12.0],
                    [1, 2, 0.3, 15.0],
                    [1, 2, 0.4, 16.0],
                    [1, 3, 0.3, 17.0],
                ],
                columns=[
                    "store_id",
                    "item_id",
                    "timestamp",
                    "sales",
                ],
            ),
            index_names=["store_id", "item_id"],
        )
        operator = SetIndexOperator(
            self.input_evt,
            feature_names=["store_id", "item_id"],
            append=False,
        )
        # instance operator implementation
        operator_impl = SetIndexNumpyImplementation(operator)

        # call operator
        op_numpy_output_evt = operator_impl.call(event=self.numpy_input_evt)[
            "event"
        ]

        # validate output
        self.assertEqual(op_numpy_output_evt, expected_numpy_output_evt)


if __name__ == "__main__":
    absltest.main()
