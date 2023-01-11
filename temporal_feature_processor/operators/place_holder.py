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

"""Place holder operator."""

from typing import List

from temporal_feature_processor import core_pb2 as pb
from temporal_feature_processor import operator_lib
from temporal_feature_processor import sampling as sampling_lib
from temporal_feature_processor.data import event as event_lib
from temporal_feature_processor.data import feature as feature_lib
from temporal_feature_processor.operators import base


class PlaceHolder(base.Operator):
    """Place holder operator."""

    def __init__(self, features: List[feature_lib.Feature], index: List[str]):
        super().__init__()

        sampling = sampling_lib.Sampling(index=index)

        for feature in features:
            if feature.sampling() is not None:
                raise ValueError("Cannot create a placeholder on existing features.")
            feature.set_sampling(sampling)

        self.add_output(
            "output",
            event_lib.Event(
                features=features,
                sampling=sampling,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="PLACE_HOLDER",
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(PlaceHolder)


def place_holder(
    features: List[feature_lib.Feature], index: List[str]
) -> event_lib.Event:
    return PlaceHolder(features=features, index=index).outputs()["output"]
