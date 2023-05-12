"""Utilities for unit testing."""
from typing import List, Mapping, Optional

import pandas as pd

from temporian.core.data import node as node_lib
from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature
from temporian.core.data.node import Node
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core import operator_lib
from temporian.implementation.numpy.data.event_set import EventSet

# The name of the operator is defined by the number of inputs and outputs.
# For example "OpI1O2" has 1 input and 2 outputs.


def create_input_node(name: Optional[str] = None):
    return node_lib.input_node(
        features=[
            Feature("f1", DType.FLOAT64),
            Feature("f2", DType.FLOAT64),
        ],
        index_levels=[("x", DType.INT64), ("y", DType.STRING)],
        name=name,
    )


def create_input_event_set(name: Optional[str] = None) -> EventSet:
    return EventSet.from_dataframe(
        pd.DataFrame(
            {
                "timestamp": [0, 2, 4, 6],
                "x": [10, 20, 30, 40],
                "y": ["a", "b", "c", "d"],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            }
        ),
        index_names=["x", "y"],
        name=name,
    )


class OpI1O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O1",
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, input: Node):
        super().__init__()

        self.add_input("input", input)
        self.add_output(
            "output",
            Node(
                features=[
                    Feature(
                        "f3",
                        DType.FLOAT64,
                        sampling=input.sampling,
                        creator=self,
                    ),
                    Feature(
                        "f4",
                        DType.INT64,
                        sampling=input.sampling,
                        creator=self,
                    ),
                ],
                sampling=Sampling(
                    index_levels=[], creator=self, is_unix_timestamp=False
                ),
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O1)


class OpI1O1NotCreator(base.Operator):
    """Unlike OpI1O1, OpI1O1NotCreator only passes the features/sampling."""

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O1NotCreator",
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, input: Node):
        super().__init__()
        self.add_input("input", input)
        self.add_output(
            "output",
            Node(
                features=[f for f in input.features],
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O1NotCreator)


class OpI2O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI2O1",
            inputs=[
                pb.OperatorDef.Input(key="input_1"),
                pb.OperatorDef.Input(key="input_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, input_1: Node, input_2: Node):
        super().__init__()
        self.add_input("input_1", input_1)
        self.add_input("input_2", input_2)
        self.add_output(
            "output",
            Node(
                features=[
                    Feature(
                        "f5",
                        DType.BOOLEAN,
                        sampling=input_1.sampling,
                        creator=self,
                    ),
                    Feature(
                        "f6",
                        DType.STRING,
                        sampling=input_1.sampling,
                        creator=self,
                    ),
                ],
                sampling=input_1.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI2O1)


class OpI1O2(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O2",
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[
                pb.OperatorDef.Output(key="output_1"),
                pb.OperatorDef.Output(key="output_2"),
            ],
        )

    def __init__(self, input: Node):
        super().__init__()
        self.add_input("input", input)
        self.add_output(
            "output_1",
            Node(
                features=[
                    Feature(
                        "f1",
                        DType.INT32,
                        sampling=input.sampling,
                        creator=self,
                    )
                ],
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.add_output(
            "output_2",
            Node(
                features=[
                    Feature(
                        "f1",
                        DType.FLOAT32,
                        sampling=input.sampling,
                        creator=self,
                    ),
                ],
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O2)


class OpWithAttributes(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpWithAttributes",
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[
                pb.OperatorDef.Output(key="output"),
            ],
            attributes=[
                pb.OperatorDef.Attribute(
                    key="attr_int",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="attr_str",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="attr_list",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="attr_float",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                ),
                pb.OperatorDef.Attribute(
                    key="attr_bool",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
                pb.OperatorDef.Attribute(
                    key="attr_map",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                ),
            ],
        )

    def __init__(
        self,
        input: Node,
        attr_int: int,
        attr_str: str,
        attr_list: List[str],
        attr_float: float,
        attr_bool: bool,
        attr_map: Mapping[str, str],
    ):
        super().__init__()
        self.add_attribute("attr_int", attr_int)
        self.add_attribute("attr_str", attr_str)
        self.add_attribute("attr_list", attr_list)
        self.add_attribute("attr_float", attr_float)
        self.add_attribute("attr_bool", attr_bool)
        self.add_attribute("attr_map", attr_map)
        self.add_input("input", input)
        self.add_output(
            "output",
            Node(
                features=[],
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpWithAttributes)
