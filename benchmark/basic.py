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
"""
Basic benchmarking script for temporian.

The script creates two events, applies some operators, and evaluates the graph.
"""

import pandas as pd
import temporian as tp
from temporian.implementation.numpy.data.event import NumpyEvent


# control size of benchmark
# N = 10000 takes about 20s
# TODO: receive N as an argument to enable passing via bazel run ... -- --N=...
N = 10000

# store ids
TRYOLABS_SHOP = 42
GOOGLE_SHOP = 101

# product ids
MATE_ID = 1
BOOK_ID = 2
PIXEL_ID = 3


@profile
def main():
    print(f"Running basic benchmark with N={N}...")

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

    event_1_data.sampling = event_2_data.sampling

    event_1 = event_1_data.schema()
    event_2 = event_2_data.schema()
    event_2._sampling = event_1._sampling

    a = tp.assign(event_1, event_2)
    c = tp.sma(a, window_length=1000)
    d = tp.assign(a, tp.sample(c, a))

    tp.evaluate(
        d,
        input_data={
            event_1: event_1_data,
            event_2: event_2_data,
        },
        check_execution=False,
    )


if __name__ == "__main__":
    main()
