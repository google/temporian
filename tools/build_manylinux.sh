#!/bin/bash

# When run in a manylinux2014 container, builds temporian and fixes the binaries
# for compatibility.
# Usage: PYTHON_VERSION=<version> tools/build_manylinux.sh
# where <version> is 38, 39, 310, or 311.

set -x
set -e

function temporian::setup_environment() {
    if [[ -z "${PYTHON_VERSION}" ]]; then
        echo "Must set PYTHON_VERSION env to 38|39|310|311"; exit 1;
    fi
    # Bazel will use PYTHON_BIN_PATH to determine the right python library.
    if [[ "${PYTHON_VERSION}" == 38 ]]; then
        PYTHON_DIR=/opt/python/cp38-cp38
        elif [[ "${PYTHON_VERSION}" == 39 ]]; then
        PYTHON_DIR=/opt/python/cp39-cp39
        elif [[ "${PYTHON_VERSION}" == 310 ]]; then
        PYTHON_DIR=/opt/python/cp310-cp310
        elif [[ "${PYTHON_VERSION}" == 311 ]]; then
        PYTHON_DIR=/opt/python/cp311-cp311
    else
        echo "Must set PYTHON_VERSION env to 38|39|310|311"; exit 1;
    fi
    export PIP_BIN="${PYTHON_DIR}"/bin/pip
    export PYTHON_BIN_PATH="${PYTHON_DIR}"/bin/python
    echo "PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
    export WHEEL_BIN="${PYTHON_DIR}"/bin/wheel
    ${PIP_BIN} install --upgrade pip
    ${PIP_BIN} install poetry auditwheel==5.2.0
}

function temporian::build_wheel() {
    export PATH="${PYTHON_DIR}/bin:$PATH"
    "${PYTHON_BIN_PATH}" -m poetry build
}

function temporian::stamp_wheels() {
    for WHEEL_PATH in $PWD/dist/*.whl; do
        WHEEL_DIR=$(dirname "${WHEEL_PATH}")
        TMP_DIR="$(mktemp -d)"
        auditwheel repair --plat manylinux2014_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"
        rm -f "${WHEEL_PATH}"
    done
}

temporian::setup_environment
temporian::build_wheel
temporian::stamp_wheels

# TODO: Add automated tests.