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
print("Imports")
import pandas as pd

# from scalene import scalene_profiler
from temporian.implementation.numpy.data.event import NumpyEvent

# Even if not used, ensure that all the necessary code is loaded.
import temporian as tp
import numpy as np
import random

N = 100000
# store ids
TRYOLABS_SHOP = 42
GOOGLE_SHOP = 101
# product ids
MATE_ID = 1
BOOK_ID = 2
PIXEL_ID = 3


def main():
    print("Large memory usage")
    # a = np.random.random(1_000_000_000)
    # a = [random.random() for x in range(1_000_000)]

    print("Running benchmark")

    # scalene_profiler.start()

    event_1_data = NumpyEvent.from_dataframe(
        pd.DataFrame(
            data=[
                *(
                    [
                        [TRYOLABS_SHOP, MATE_ID, 0.0, 14],
                        [TRYOLABS_SHOP, MATE_ID, 1.0, 15],
                        [TRYOLABS_SHOP, MATE_ID, 2.0, 16],
                        [TRYOLABS_SHOP, BOOK_ID, 1.0, 10],
                        [GOOGLE_SHOP, BOOK_ID, 1.0, 7],
                        [GOOGLE_SHOP, BOOK_ID, 2.0, 8],
                        [GOOGLE_SHOP, PIXEL_ID, 0.0, 3],
                        [GOOGLE_SHOP, PIXEL_ID, 1.0, 4],
                    ]
                    * N
                )
            ],
            columns=["store_id", "product_id", "timestamp", "sales"],
        ),
        index_names=["store_id", "product_id"],
    )

    event_2_data = NumpyEvent.from_dataframe(
        pd.DataFrame(
            data=[
                *(
                    [
                        [TRYOLABS_SHOP, MATE_ID, 0.0, -14],
                        [TRYOLABS_SHOP, MATE_ID, 1.0, -15],
                        [TRYOLABS_SHOP, MATE_ID, 2.0, -16],
                        [TRYOLABS_SHOP, BOOK_ID, 1.0, -10],
                        [GOOGLE_SHOP, BOOK_ID, 1.0, -7],
                        [GOOGLE_SHOP, BOOK_ID, 2.0, -8],
                        [GOOGLE_SHOP, PIXEL_ID, 0.0, -3],
                        [GOOGLE_SHOP, PIXEL_ID, 1.0, -4],
                    ]
                    * N
                )
            ],
            columns=["store_id", "product_id", "timestamp", "costs"],
        ),
        index_names=["store_id", "product_id"],
    )

    # TODO: Remove the following line when "from_dataframe" support creating
    # event data with shared sampling. Note that "event_1_data" and
    # "event_2_data" should have the same sampling in this tests.
    event_1_data.sampling = event_2_data.sampling

    event_1 = event_1_data.schema()
    event_2 = event_2_data.schema()
    event_2._sampling = event_1._sampling

    expected_output_event = NumpyEvent.from_dataframe(
        pd.DataFrame(
            data=[
                *(
                    [
                        [TRYOLABS_SHOP, MATE_ID, 0.0, 14, -14, 0],
                        [TRYOLABS_SHOP, MATE_ID, 1.0, 15, -15, 14],
                        [TRYOLABS_SHOP, MATE_ID, 2.0, 16, -16, 15],
                        [TRYOLABS_SHOP, BOOK_ID, 1.0, 10, -10, 0],
                        [GOOGLE_SHOP, BOOK_ID, 1.0, 7, -7, 0],
                        [GOOGLE_SHOP, BOOK_ID, 2.0, 8, -8, 7],
                        [GOOGLE_SHOP, PIXEL_ID, 0.0, 3, -3, 0],
                        [GOOGLE_SHOP, PIXEL_ID, 1.0, 4, -4, 3],
                    ]
                    * N
                )
            ],
            columns=[
                "store_id",
                "product_id",
                "timestamp",
                "sales",
                "costs",
                "lag[1s]_sales",
            ],
        ),
        index_names=["store_id", "product_id"],
    )

    a = tp.assign(event_1, event_2)
    # create and assign sum feature
    # TODO: Restore when arithmetic operator is fixed.
    # b = tp.assign(a, event_1 + event_2)
    c = tp.lag(event_1, duration=1)
    d = tp.assign(a, tp.sample(c, a))
    output_event = d

    output_event_numpy = tp.evaluate(
        output_event,
        input_data={
            event_1: event_1_data,
            event_2: event_2_data,
        },
        # TODO: The assign operator has some issues with dtypes. Re-enable
        # checking when solved.
        check_execution=True,
    )

    # Turn profiling off
    # scalene_profiler.stop()


if __name__ == "__main__":
    main()
