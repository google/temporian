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

import datetime
import math
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.parquet import from_parquet, to_parquet
from temporian.test.utils import assertEqualDFRandomRowOrder


class ParquetEventSet(absltest.TestCase):
    def test_correct(self) -> None:
        es = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
        )

        f = NamedTemporaryFile(delete=False)
        to_parquet(es, f.name)
        result = from_parquet(f.name, indexes=["product_id"])
        self.assertEqual(es, result)


if __name__ == "__main__":
    absltest.main()
