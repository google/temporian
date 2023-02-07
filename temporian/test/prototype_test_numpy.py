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
import numpy as np
from absl import logging
from absl.testing import absltest

from temporian.core import evaluator
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.select import select
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class PrototypeTest(absltest.TestCase):
    def setUp(self) -> None:
        # index name
        index_names = ["store_id", "product_id"]

        # sampling
        sampling = NumpySampling(
            names=index_names,
            data={
                ("A", 1): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                ("A", 2): np.array(["2022-02-06"], dtype="datetime64"),
                ("B", 2): np.array(
                    ["2022-02-06", "2022-02-07"], dtype="datetime64"
                ),
                ("B", 3): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
            },
        )
        # input event - contains two features, "sales" and "costs"
        self.event = NumpyEvent(
            data={
                ("A", 1): [
                    NumpyFeature(
                        name="sales",
                        index=index_names,
                        data=np.array([14, 15, 16]),
                    ),
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([11, 12, 13]),
                    ),
                ],
                ("A", 2): [
                    NumpyFeature(
                        name="sales",
                        index=index_names,
                        data=np.array([10]),
                    ),
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([9]),
                    ),
                ],
                ("B", 2): [
                    NumpyFeature(
                        name="sales",
                        index=index_names,
                        data=np.array([7, 8]),
                    ),
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([5, 6]),
                    ),
                ],
                ("B", 3): [
                    NumpyFeature(
                        name="sales",
                        index=index_names,
                        data=np.array([3, 4]),
                    ),
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([1, 2]),
                    ),
                ],
            },
            sampling=sampling,
        )
        # expected output event. Select "costs" feature only
        self.expected_output_event = NumpyEvent(
            data={
                ("A", 1): [
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([11, 12, 13]),
                    ),
                ],
                ("A", 2): [
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([9]),
                    ),
                ],
                ("B", 2): [
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([5, 6]),
                    ),
                ],
                ("B", 3): [
                    NumpyFeature(
                        name="costs",
                        index=index_names,
                        data=np.array([1, 2]),
                    ),
                ],
            },
            sampling=sampling,
        )

    def test_prototoype(self) -> None:
        event = Event(
            [Feature("sales", int), Feature("costs", int)],
            sampling=Sampling(["store_id", "product_id"]),
        )
        output_event = event["costs"]

        output_event_numpy = evaluator.evaluate(
            output_event,
            input_data={
                # assignee event specified from disk
                event: self.event,
            },
            backend="numpy",
        )
        materialized_output = output_event_numpy[output_event]
        logging.info(f"{materialized_output=}")
        logging.info("-" * 200)
        logging.info(f"{self.expected_output_event=}")


if __name__ == "__main__":
    absltest.main()
