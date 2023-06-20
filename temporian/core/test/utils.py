"""Utilities for unit testing."""
from typing import Dict, List, Optional

from temporian.core.data.dtypes.dtype import DType
from temporian.core.data.node import (
    Node,
    input_node,
    create_node_with_new_reference,
    create_node_new_features_new_sampling,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core import operator_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set

# The name of the operator is defined by the number of inputs and outputs.
# For example "OpI1O2" has 1 input and 2 outputs.


def create_source_node(name: Optional[str] = None):
    return input_node(
        features=[
            ("f1", DType.FLOAT64),
            ("f2", DType.FLOAT64),
        ],
        indexes=[("x", DType.INT64), ("y", DType.STRING)],
        name=name,
    )


def create_input_event_set(name: Optional[str] = None) -> EventSet:
    return event_set(
        timestamps=[0, 2, 4, 6],
        features={
            "x": [10, 20, 30, 40],
            "y": ["a", "b", "c", "d"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [5.0, 6.0, 7.0, 8.0],
        },
        indexes=["x", "y"],
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
            create_node_new_features_new_sampling(
                features=[
                    ("f3", DType.FLOAT64),
                    ("f4", DType.INT64),
                ],
                indexes=[],
                is_unix_timestamp=False,
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
            create_node_with_new_reference(
                schema=input.schema,
                features=input.feature_nodes,
                sampling=input.sampling_node,
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
            create_node_new_features_existing_sampling(
                features=[
                    ("f5", DType.BOOLEAN),
                    ("f6", DType.STRING),
                ],
                sampling_node=input_1,
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
            create_node_new_features_existing_sampling(
                features=[("f1", DType.INT32)],
                sampling_node=input,
                creator=self,
            ),
        )
        self.add_output(
            "output_2",
            create_node_new_features_existing_sampling(
                features=[("f1", DType.FLOAT32)],
                sampling_node=input,
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
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
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
                pb.OperatorDef.Attribute(
                    key="attr_list_dtypes",
                    type=pb.OperatorDef.Attribute.Type.LIST_DTYPE,
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
        attr_map: Dict[str, str],
        attr_list_dtypes: List[DType],
    ):
        super().__init__()
        self.add_attribute("attr_int", attr_int)
        self.add_attribute("attr_str", attr_str)
        self.add_attribute("attr_list", attr_list)
        self.add_attribute("attr_float", attr_float)
        self.add_attribute("attr_bool", attr_bool)
        self.add_attribute("attr_map", attr_map)
        self.add_attribute("attr_list_dtypes", attr_list_dtypes)

        self.add_input("input", input)
        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[],
                sampling_node=input,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpWithAttributes)
