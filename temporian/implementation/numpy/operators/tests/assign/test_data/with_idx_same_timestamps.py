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

"""NumpyAssignOperator - with index, same timestamps test.

Tests the correct output when the right event has same timestamps than the left
event, for any index value. Both input events are indexed.
"""
import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    index=["product_id"],
    data={
        (666964,): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        (372306,): np.array(["2022-02-06"], dtype="datetime64"),
    },
)

sampling_2 = NumpySampling(
    index=["product_id"],
    data={
        (666964,): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        (372306,): np.array(["2022-02-06"], dtype="datetime64"),
    },
)

INPUT_1 = NumpyEvent(
    data={
        (666964,): [
            NumpyFeature(
                name="sales",
                data=np.array([0.0, 100.0, 200.0]),
            ),
        ],
        (372306,): [
            NumpyFeature(
                name="sales",
                data=np.array([1160.0]),
            ),
        ],
    },
    sampling=sampling_1,
)

INPUT_2 = NumpyEvent(
    data={
        (666964,): [
            NumpyFeature(
                name="costs",
                data=np.array([0.0, 250.0, 500.0]),
            ),
        ],
        (372306,): [
            NumpyFeature(
                name="costs",
                data=np.array([508.0]),
            ),
        ],
    },
    sampling=sampling_2,
)

OUTPUT = NumpyEvent(
    data={
        (666964,): [
            NumpyFeature(
                name="sales",
                data=np.array([0.0, 100.0, 200.0]),
            ),
            NumpyFeature(
                name="costs",
                data=np.array([0.0, 250.0, 500.0]),
            ),
        ],
        (372306,): [
            NumpyFeature(
                name="sales",
                data=np.array([1160.0]),
            ),
            NumpyFeature(
                name="costs",
                data=np.array([508.0]),
            ),
        ],
    },
    sampling=sampling_1,
)
