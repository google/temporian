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

from absl import logging
from absl.testing import absltest

from temporian.core import serialize
from temporian.core import processor
from temporian.core.data import dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core.test import utils


class SerializeTest(absltest.TestCase):
    def test_serialize(self):
        o1 = utils.OpO1()
        o2 = utils.OpI1O1(o1.outputs()["output"])
        o3 = utils.OpO1()
        o4 = utils.OpI2O1(o2.outputs()["output"], o3.outputs()["output"])
        o5 = utils.OpI1O2(o4.outputs()["output"])

        original = processor.infer_processor(
            [o1.outputs()["output"], o3.outputs()["output"]],
            [o5.outputs()["output_1"], o4.outputs()["output"]],
        )
        logging.info("original:\n%s", original)

        proto = serialize.serialize(original)
        logging.info("proto:\n%s", proto)

        restored = serialize.unserialize(proto)
        logging.info("restored:\n%s", restored)


if __name__ == "__main__":
    absltest.main()
