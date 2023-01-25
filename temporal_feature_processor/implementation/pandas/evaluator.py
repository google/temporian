from typing import Dict, List

from temporal_feature_processor.core.operators import base
from temporal_feature_processor.core.data import event
from temporal_feature_processor.implementation.pandas.data import event as pandas_event
from temporal_feature_processor.implementation.pandas.operators import core_mapping


def evaluate_schedule(
    data: Dict[event.Event,
               pandas_event.PandasEvent], schedule: List[base.Operator]
) -> Dict[event.Event, pandas_event.PandasEvent]:
  outputs = []
  for operator in schedule:

    # get implementation
    implementation = core_mapping.OPERATOR_IMPLEMENTATIONS[operator.definition(
    ).key]()  # TODO: add operator attributes when instancing implementation

    # construct operator inputs
    operator_inputs = {
        input_key: data[input_event]
        for input_key, input_event in operator.inputs().items()
    }

    # compute output
    operator_outputs = implementation(**operator_inputs)
    outputs.append(operator_outputs)

    # materialize data in output events
    for output_key, output_event in operator.outputs().items():
      data[output_event] = operator_outputs[output_key]

  return outputs
