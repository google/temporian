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


from typing import Dict, Tuple, Iterable
import numpy as np

from temporian.core.operators.propagate import (
    Propagate as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    extract_from_iterable,
)
from temporian.beam.typing import BeamEventSet, FeatureItem, BeamIndexKey
import apache_beam as beam
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)
from temporian.implementation.numpy.data import dtype_normalization


class PropagateBeamImplementation(BeamOperatorImplementation):
    def call(
        self, input: BeamEventSet, sampling: BeamEventSet
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        index_mapping = self.operator.index_mapping
        output_schema = self.operator.outputs["output"].schema
        output_np_dtypes = [
            dtype_normalization.tp_dtype_to_np_dtype(feature.dtype)
            for feature in output_schema.features
        ]
        has_no_features = len(output_schema.features) == 0

        def sampling_index_to_input_index(
            item: FeatureItem,
        ) -> Tuple[BeamIndexKey, BeamIndexKey]:
            indexes, _ = item
            input_indexes = tuple([indexes[i] for i in index_mapping])
            return input_indexes, indexes

        assert len(sampling) > 0
        groupped_sampling = sampling[
            0
        ] | f"Group sampling {self.operator}" >> beam.Map(
            sampling_index_to_input_index
        )

        def duplicate_features(
            groups: Tuple[
                BeamIndexKey,
                Tuple[Iterable[FeatureItemValue], Iterable[BeamIndexKey]],
            ],
            output_feature_idx: int,
        ):
            _, (inputs, sampling_indexes) = groups
            single_input = extract_from_iterable(inputs)
            if single_input is None:
                empty_timestamps = np.array([], dtype=np.float64)
                for sampling_index in sampling_indexes:
                    if has_no_features:
                        empty_values = None
                    else:
                        empty_values = np.zeros(
                            shape=0,
                            dtype=output_np_dtypes[output_feature_idx],
                        )
                    yield sampling_index, (empty_timestamps, empty_values)
            else:
                input_timestamps, input_values = single_input
                for sampling_index in sampling_indexes:
                    yield sampling_index, (input_timestamps, input_values)

        def apply(feature_idx, item):
            return (
                (
                    item,
                    groupped_sampling,
                )
                | f"Join feature #{feature_idx} and sampling {self.operator}"
                >> beam.CoGroupByKey()
                | f"Emit feature #{feature_idx}  {self.operator}"
                >> beam.FlatMap(duplicate_features, feature_idx)
            )

        output = tuple(apply(idx, item) for idx, item in enumerate(input))
        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, PropagateBeamImplementation
)
