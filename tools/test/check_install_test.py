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

from absl.testing import absltest
from tools.check_install import check_install


class CheckInstallTest(absltest.TestCase):
    def test_check_install(self):
        eval_output = str(check_install())
        self.assertTrue("indexes:" in eval_output)
        self.assertTrue("features:" in eval_output)
        self.assertTrue("events:" in eval_output)


if __name__ == "__main__":
    absltest.main()
