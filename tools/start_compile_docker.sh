#!/bin/bash

# Starts the manylinux2014 docker container that includes a bazel installation.

DOCKER=gcr.io/tfx-oss-public/manylinux2014-bazel:bazel-5.3.0
TEMPORIAN_DIR=temporian
sudo docker run -it -v ${PWD}/..:/working_dir -w /working_dir/${TEMPORIAN_DIR} ${DOCKER} $@