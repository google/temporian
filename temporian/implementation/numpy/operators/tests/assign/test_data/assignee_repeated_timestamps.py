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

"""NumpyAssignOperator - assignee repeated timestamps test.

Tests the correct functionality of the NumpyAssignOperator when the assignee
has repeated timestamps.

"""

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    names=["user_id"],
    data={
        (151591562,): np.array(
            [
                "2022-02-05",
                "2022-02-05",  # 2022-02-05 is repeated here should not be a problem
                "2022-02-07",
            ],
            dtype="datetime64",
        ),
        (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
    },
)

sampling_2 = NumpySampling(
    names=["user_id"],
    data={
        (151591562,): np.array(
            ["2022-02-05", "2022-02-07", "2022-02-09"],
            dtype="datetime64",
        ),
        (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
    },
)

INPUT_1 = NumpyEvent(
    data={
        (151591562,): [
            NumpyFeature(
                name="sales",
                data=np.array([10.0, 20.0, 30.0]),
            ),
        ],
        (191562515,): [
            NumpyFeature(
                name="sales",
                data=np.array([40.0]),
            ),
        ],
    },
    sampling=sampling_1,
)

INPUT_2 = NumpyEvent(
    data={
        (151591562,): [
            NumpyFeature(
                name="costs",
                data=np.array([-10.0, -20.0, 0.0]),
            ),
        ],
        (191562515,): [
            NumpyFeature(
                name="costs",
                data=np.array([0.0]),
            ),
        ],
    },
    sampling=sampling_2,
)

OUTPUT = NumpyEvent(
    data={
        (151591562,): [
            NumpyFeature(
                name="sales",
                data=np.array([10.0, 20.0, 30.0]),
            ),
            NumpyFeature(
                name="costs",
                data=np.array([-10.0, -10.0, -20.0]),  # -10.0 is repeated here
                # because the timestamp is repeated
            ),
        ],
        (191562515,): [
            NumpyFeature(
                name="sales",
                data=np.array([40.0]),
            ),
            NumpyFeature(
                name="costs",
                data=np.array([0.0]),
            ),
        ],
    },
    sampling=sampling_1,
)
