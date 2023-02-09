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

"""PandasAssignOperator - repeated timestamps test.

Tests that there cannot exist repeated timestamps on the assigned event for any
index value. Repeated timestamps are allowed on the assignee event.
"""

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    names=["user_id"],
    data={
        (151591562,): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
    },
)

sampling_2 = NumpySampling(
    names=["user_id"],
    data={
        (151591562,): np.array(
            ["2022-02-08", "2022-02-09", "2022-02-10"],
            dtype="datetime64",
        ),
        (191562515,): np.array(["2022-02-08"], dtype="datetime64"),
    },
)
