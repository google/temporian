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

from temporal_feature_processor.implementation.pandas import evaluator as pandas_evaluator
from temporal_feature_processor.implementation.pandas.data import event as pandas_event


def raise_(exception: Exception):
  raise exception


BACKENDS = {
    "cpp": {
        "event": lambda: raise_(NotImplementedError()),
        "evaluate_schedule_fn": lambda data, schedule: raise_(
            NotImplementedError()
        ),
        "read_csv_fn": lambda path: raise_(NotImplementedError()),
    },
    "pandas": {
        "event": pandas_event.PandasEvent,
        "evaluate_schedule_fn": pandas_evaluator.evaluate_schedule,
        "read_csv_fn": pandas_event.pandas_event_from_csv,
    },
}
