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


# Ensure that Beam is available
import logging

try:
    import apache_beam as _
except ImportError as e:
    logging.fatal(
        """Cannot import Apache Beam library.

Solutions:

1. Install Apache Beam. You can install Beam with the following command: pip install apache-beam

2. Remove the temporian.beam import. If you remove the temporian.beam import, you can still run Temporian graphs in process with `import temporian as tp` and `tp.evaluate`.
"""
    )

from temporian.beam import io as _io

read_csv_raw = _io.read_csv_raw
read_csv = _io.read_csv
write_csv = _io.write_csv
to_event_set = _io.to_event_set

from temporian.beam import evaluation as _evaluation

# TODO: Expose a function-like API.
run = _evaluation.run
run_multi_io = _evaluation.run_multi_io
