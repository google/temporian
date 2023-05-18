#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Moves the bazel generated files needed for packaging the wheel to the source
# tree.
# When adding new C++-extensions or protos, include those below.

function temporian::move_generated_files() {
  # If run by "bazel run", $(pwd) is the .runfiles dir that contains all the
  # data dependencies.
  GENFILES=$(pwd)

  PYBIND_SO="temporian/implementation/numpy_cc/operators/operators_cc.so"
  cp -f "${BUILD_WORKSPACE_DIRECTORY}/bazel-bin/${PYBIND_SO}" \
    "${BUILD_WORKSPACE_DIRECTORY}/${PYBIND_SO}"

  FILES="
    temporian/proto/core_pb2.py
  "
  for FILE in ${FILES}; do
    cp -f ${GENFILES}/${FILE} ${BUILD_WORKSPACE_DIRECTORY}/${FILE}
  done
}

temporian::move_generated_files