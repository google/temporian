"""Reads csv files in Beam."""

import apache_beam as beam
from temporian.core.data.node import Node
from typing import Dict, List
from temporian.beam.io import BeamEventSet
from temporian.core.evaluation import build_schedule
from temporian.beam import implementation_lib
from temporian.beam import operators as _  # Implementations
import sys


def run_multi_io(
    inputs: Dict[Node, beam.PCollection[BeamEventSet]], outputs: List[Node]
) -> Dict[Node, beam.PCollection[BeamEventSet]]:
    """
    TODO: Support more containers for outputs and returned values.
    """

    schedule = build_schedule(
        inputs=set(inputs.keys()), outputs=set(outputs), verbose=2
    )

    data = {**inputs}

    num_operators = len(schedule.ordered_operators)
    for operator_idx, operator in enumerate(schedule.ordered_operators):
        operator_def = operator.definition()

        print("=============================", file=sys.stderr)
        print(
            f"{operator_idx+1} / {num_operators}: Run {operator}",
            file=sys.stderr,
        )

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_node]
            for input_key, input_node in operator.inputs.items()
        }

        # Get Beam implementation
        implementation_cls = implementation_lib.get_implementation_class(
            operator_def.key
        )
        implementation = implementation_cls(operator)

        # Add implementation to Beam pipeline
        operator_outputs = implementation(**operator_inputs)

        # Collect outputs
        for output_key, output_node in operator.outputs.items():
            data[output_node] = operator_outputs[output_key]

    return {output: data[output] for output in outputs}


@beam.ptransform_fn
def run(
    pipe: beam.PCollection[BeamEventSet], input: Node, output: Node
) -> beam.PCollection[BeamEventSet]:
    output_pipe = run_multi_io(inputs={input: pipe}, outputs=[output])
    return output_pipe[output]
