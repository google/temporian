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

"""NumpyAssignOperator - different index test.

Tests that the left and right events cannot have different indexes.
"""
import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    names=["store_id"],
    data={
        ("A",): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        ("B",): np.array(["2022-02-05", "2022-02-06"], dtype="datetime64"),
    },
)

sampling_2 = NumpySampling(
    names=["product_id"],
    data={
        (1,): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        (2,): np.array(["2022-02-05", "2022-02-06"], dtype="datetime64"),
    },
)

INPUT_1 = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="sales",
                data=np.array([14, 15, 16]),
            ),
        ],
        ("B",): [
            NumpyFeature(
                name="sales",
                data=np.array([10, 11]),
            ),
        ],
    },
    sampling=sampling_1,
)

INPUT_2 = NumpyEvent(
    data={
        (1,): [
            NumpyFeature(
                name="costs",
                data=np.array([-14, -15, -16]),
            ),
        ],
        (2,): [
            NumpyFeature(
                name="costs",
                data=np.array([-10, -11]),
            ),
        ],
    },
    sampling=sampling_2,
)
