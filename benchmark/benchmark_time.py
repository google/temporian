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
Benchmark Python API.
"""

import time
from typing import List, NamedTuple, Union

import numpy as np
import temporian as tp
import pandas as pd

from temporian.implementation.numpy.data.event import NumpyEvent

# TODO(gbm): Add flag to control which benchmark to run.


def _build_toy_dataset(
    n: int, data_prefix="", data2_is_categorical_integer=False
) -> NumpyEvent:
    """Builds a toy dataset with two features.

    Args:
        n: Number of timestamps.
        data_prefix: Optional prefix in the feature names.
        data2_is_categorical_integer: If true, the second feature is
            categorical. If false (default), the second feature is numerical.

    Returns:
        A NumpyEvent containing the toy dataset.
    """

    np.random.seed(0)
    index_values = list(range(int(10)))
    timestamps = np.sort(np.random.randn(n) * n)
    index_1 = np.random.choice(index_values, n)
    index_2 = np.random.choice(index_values, n)
    data_1 = np.random.randn(n)
    if data2_is_categorical_integer:
        data_2 = np.random.choice(list(range(int(10))), n)
    else:
        data_2 = np.random.randn(n)

    return NumpyEvent.from_dataframe(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "index_1": index_1,
                "index_2": index_2,
                data_prefix + "data_1": data_1,
                data_prefix + "data_2": data_2,
            }
        ),
        index_names=["index_1", "index_2"],
    )


def benchmark_simple_moving_average(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n)

        event = ds.schema()
        output = tp.simple_moving_average(event, window_length=10)

        runner.benchmark(
            f"simple_moving_average:{n}",
            lambda: tp.evaluate(output, input_data={event: ds}),
        )


def benchmark_select_and_glue(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n)

        event = ds.schema()
        output = tp.glue(event["data_1"], event["data_2"])

        runner.benchmark(
            f"select_and_glue:{n}",
            lambda: tp.evaluate(output, input_data={event: ds}),
        )


def benchmark_calendar_day_of_month(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        np.random.seed(0)
        timestamps = np.sort(np.random.randn(n) * 1_700_000_000).astype(
            "datetime64[s]"
        )
        ds = NumpyEvent.from_dataframe(pd.DataFrame({"timestamp": timestamps}))

        event = ds.schema()
        output = tp.calendar_day_of_month(event)

        runner.benchmark(
            f"calendar_day_of_month:{n}",
            lambda: tp.evaluate(output, input_data={event: ds}),
        )


def benchmark_sample(runner):
    runner.add_separator()
    for m in [100, 10_000, 1_000_000]:
        for n in [100, 10_000, 1_000_000]:
            ds_1 = _build_toy_dataset(m, "ds_1")
            ds_2 = _build_toy_dataset(n, "ds_2")

            event_1 = ds_1.schema()
            event_2 = ds_2.schema()
            output = tp.sample(event_1, event_2)

            runner.benchmark(
                f"sample:e{m}_s{n}",
                lambda: tp.evaluate(
                    output, input_data={event_1: ds_1, event_2: ds_2}
                ),
            )


def benchmark_propagate(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n, data2_is_categorical_integer=True)
        event = ds.schema()
        output = tp.propagate(event["data_1"], event["data_2"])

        runner.benchmark(
            f"propagate:{n}",
            lambda: tp.evaluate(output, input_data={event: ds}),
        )


def benchmark_from_dataframe(runner):
    runner.add_separator()
    # TODO: Add num_timestamps = 100_000 and 1_000_000 when from_dataframe is
    # fast enough.
    for num_timestamps in [10_000]:
        for num_indexes in [0, 1, 3, 5]:
            for num_index_values in [20]:
                for index_is_string in [False, True]:
                    np.random.seed(0)

                    dt = {"timestamp": np.sort(np.random.rand(num_timestamps))}
                    index_names = []

                    inde_values = list(range(num_index_values))
                    for index_idx in range(num_indexes):
                        index_name = f"index_{index_idx}"
                        index_values = np.random.choice(
                            inde_values, num_timestamps
                        )
                        if index_is_string:
                            index_values = [f"v_{x}" for x in index_values]
                        dt[index_name] = index_values
                        index_names.append(index_name)

                    dt["feature_int64"] = np.random.randint(
                        0, 1000, num_timestamps, np.int64
                    )
                    dt["feature_str"] = np.random.rand(num_timestamps)
                    df = pd.DataFrame(dt)

                    benchmark_name = (
                        f"from_dataframe:s:{num_timestamps}_"
                        f"numidx:{num_indexes}_"
                        f"numidxval:{num_index_values}_"
                        f"idx:{'str' if index_is_string else 'int'}"
                    )

                    runner.benchmark(
                        benchmark_name,
                        lambda: NumpyEvent.from_dataframe(df, index_names),
                    )


def main():
    print("Running benchmark")
    runner = Runner()
    benchmark_from_dataframe(runner)
    benchmark_simple_moving_average(runner)
    benchmark_select_and_glue(runner)
    benchmark_calendar_day_of_month(runner)
    benchmark_sample(runner)
    benchmark_propagate(runner)

    print("All results (again)")
    runner.print_results()


class BenchmarkResult(NamedTuple):
    name: str
    wall_time_seconds: float
    cpu_time_seconds: float

    def __str__(self):
        return (
            f"{self.name:30}    "
            f"{self.wall_time_seconds:10.5f}    "
            f"{self.cpu_time_seconds:10.5f}"
        )


class Runner:
    def __init__(self):
        # Name and tunning time (in seconds) of each benchmarks.
        self._results: List[Union[BenchmarkResult, None]] = []

    def benchmark(self, name, callback, warmup_repetitions=1, repetitions=5):
        print(f"Running {name}")
        # Warmup
        for _ in range(warmup_repetitions):
            callback()

        begin_wall_time = time.perf_counter()
        begin_cpu_time = time.process_time()

        for _ in range(repetitions):
            callback()

        end_wall_time = time.perf_counter()
        end_cpu_time = time.process_time()

        result = BenchmarkResult(
            name=name,
            wall_time_seconds=(end_wall_time - begin_wall_time) / repetitions,
            cpu_time_seconds=(end_cpu_time - begin_cpu_time) / repetitions,
        )
        self._results.append(result)

    def add_separator(self):
        self._results.append(None)

    def print_results(self):
        sep_length = 64
        print("=" * sep_length)
        print(f"{'Name':30}    {'Wall time (s)':10}    {'CPU time (s)':10}")
        print("=" * sep_length)
        for idx, result in enumerate(self._results):
            if result is None:
                if idx != 0:
                    print("-" * sep_length, flush=True)
            else:
                print(result)
        print("=" * sep_length, flush=True)


if __name__ == "__main__":
    main()
