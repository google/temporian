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


"""Implementation for the Combine operator."""


from typing import Dict, List
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.combine import Combine
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class CombineNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Combine) -> None:
        assert isinstance(operator, Combine)
        super().__init__(operator)

    def __call__(self, **inputs: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Combine)
        output_schema = self.output_schema("output")
        output_evset = EventSet(data={}, schema=output_schema)

        input_evsets = list(inputs.values())
        first_input = input_evsets[0]
        first_features = first_input.schema.feature_names()

        # Concatenate timestamps and features for each index, and sort
        for index_key, index_data in first_input.data.items():
            all_timestamps: List[np.ndarray] = []
            all_feats: Dict[str, List[np.ndarray]] = {
                name: [] for name in first_features
            }

            # Get timestamps and features from all input EventSets
            for evset in input_evsets:
                index_data = evset.get_index_value(index_key)
                evset_feats = evset.schema.feature_names()
                all_timestamps.append(index_data.timestamps)
                # Access feature data by name, since position in IndexData may vary
                for feat in first_features:
                    feat_idx = evset_feats.index(feat)
                    all_feats[feat].append(index_data.features[feat_idx])

            # Concatenate and sort timestamps
            timestamps: np.ndarray = np.concatenate(all_timestamps)
            order = np.argsort(timestamps, kind="mergesort")
            timestamps = timestamps[order]

            # Concatenate features and sort based on timestamps
            features: List[np.ndarray] = []
            for feature in first_features:
                feat = np.concatenate(all_feats[feature])
                features.append(feat[order])

            # Fill IndexData
            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=features,
                    timestamps=timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Combine, CombineNumpyImplementation
)
