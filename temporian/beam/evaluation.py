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

"""Run a graph in Beam."""

import sys
from typing import Dict, List

import apache_beam as beam

from temporian.core.data.node import EventSetNode
from temporian.core.evaluation import build_schedule
from temporian.beam import implementation_lib
from temporian.beam import operators as _  # Implementations
from temporian.beam.typing import BeamEventSet


@beam.ptransform_fn
def run(
    pipe: BeamEventSet,
    input: EventSetNode,
    output: EventSetNode,
    verbose: int = 0,
) -> BeamEventSet:
    """Runs a single-input, single-output Temporian graph in Beam.

    Usage example:

    ```python
    import temporian as tp
    import temporian.beam as tpb

    # Create a graph.
    input_node = tp.input_node([("a", tp.str_), ("b", tp.float32)])
    output_node = input_node["b"].moving_sum(4)

    with beam.Pipeline() as pipeline:
        (pipeline
        | "Read input" >> tpb.from_csv(input_path, input_node.schema)
        | "Process data" >> tpb.run(input=input_node, output=output_node)
        | "Save result" >> tpb.to_csv(output_path, output_node.schema)
        )
        pipeline.run()
    ```

    If you graph contains more than one input or output nodes, use
    `run_multi_io` instead.

    Args:
        pipe: A Beam PCollection containing the input event set. Use
            `tpb.from_csv` to read data from csv files, or use
            `tpb.to_event_set` to import an event set from a dictionary of
            key/values such as the output of Beam IO connectors
            (https://beam.apache.org/documentation/io/connectors/).
        input: Input node of a Temporian graph.
        output: Output node of a Temporian graph.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.

    Returns:
        A Beam PCollection containing the output event set.
    """

    output_pipe = run_multi_io(
        inputs={input: pipe}, outputs=[output], verbose=verbose
    )
    return output_pipe[output]


def run_multi_io(
    inputs: Dict[EventSetNode, BeamEventSet],
    outputs: List[EventSetNode],
    verbose: int = 0,
) -> Dict[EventSetNode, BeamEventSet]:
    """Runs a multi-input, multi-output Temporian graph in Beam.

    Usage example:

    ```python
    import temporian as tp
    import temporian.beam as tpb

    # Create a graph.
    input_node_1 = tp.input_node([("a", tp.float32)])
    input_node_2 = tp.input_node([("b", tp.float32)])
    output_node_1 = input_node_1.moving_sum(4)
    output_node_2 = input_node_2.moving_sum(4)

    with beam.Pipeline() as p:
        input_beam_1 = p | tpb.from_csv(input_path_1, input_node_1.schema)
        input_beam_2 = p | tpb.from_csv(input_path_2, input_node_2.schema)

        outputs = tpb.run_multi_io(
            inputs={
                input_node_1: input_beam_1,
                input_node_2: input_beam_2,
            },
            outputs=[output_node_1, output_node_2],
        )
        outputs[output_node_1] | tpb.to_csv(
            output_path_1, output_node_1.schema, shard_name_template=""
        )
        outputs[output_node_2] | tpb.to_csv(
            output_path_2, output_node_2.schema, shard_name_template=""
        )
        pipeline.run()
    ```

    If you graph contains a single input and output node, use `run` instead.

    Args:
        inputs: EventSetNode indexed dictionary of input Beam event-sets for all the
            inputs of the Temporian graph.
        outputs: List of output nodes to compute.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.

    Returns:
        A output node indexed dictionary of output beam event-sets. Each item
        in `outputs` becomes one item in the returned dictionary.
    """

    schedule = build_schedule(
        inputs=set(inputs.keys()), outputs=set(outputs), verbose=verbose
    )

    data = {**inputs}

    num_steps = len(schedule.steps)
    for step_idx, step in enumerate(schedule.steps):
        operator_def = step.op.definition

        if verbose > 0:
            print("=============================", file=sys.stderr)
            print(
                f"{step_idx+1} / {num_steps}: Run {step.op}",
                file=sys.stderr,
            )

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_node]
            for input_key, input_node in step.op.inputs.items()
        }

        # Get Beam implementation
        implementation_cls = implementation_lib.get_implementation_class(
            operator_def.key
        )
        implementation = implementation_cls(step.op)

        # Add implementation to Beam pipeline
        operator_outputs = implementation(**operator_inputs)

        # Collect outputs
        for output_key, output_node in step.op.outputs.items():
            data[output_node] = operator_outputs[output_key]

    return {output: data[output] for output in outputs}
