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

"""NumpyAssignOperator - complete timestamps.

Tests a correct output in a complete timestamps scenario. both samplings will have different timestamps and
in different order.
"""
import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    names=["product_id"],
    data={
        (666964,): np.array(
            [
                "2022-02-01",
                "2022-02-02",
                "2022-02-03",
                "2022-02-04",
                "2022-02-05",
            ],
            dtype="datetime64",
        ),
    },
)

sampling_2 = NumpySampling(
    names=["product_id"],
    data={
        (666964,): np.array(
            [  # missing timestamps from sampling_1 (2022-02-01, 2022-02-03, 2022-02-04)
                "2022-01-31",  # not in sampling_1
                "2022-02-02",  # its on sampling_1 but in different index
                "2022-02-05",  # its on sampling_1 but in different index
                "2022-02-07",  # not in sampling_1
            ],
            dtype="datetime64",
        ),
    },
)

INPUT_1 = NumpyEvent(
    data={
        (666964,): [
            NumpyFeature(
                name="sales",
                sampling=sampling_1,
                data=np.array([10, 11, 12, 13, 14]),
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
                sampling=sampling_2,
                data=np.array([1, 2, 3, 4]),
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
                sampling=sampling_1,
                data=np.array([10, 11, 12, 13, 14]),
            ),
            NumpyFeature(
                name="costs",
                sampling=sampling_1,
                data=np.array(
                    [np.nan, 2, np.nan, np.nan, 3]
                ),  # np.nan on missing timestamps
                # 1 and 2 value on different index
                # missing 3 value as its timestamp is not in sampling_1
            ),
        ],
    },
    sampling=sampling_1,
)
