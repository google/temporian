#!/bin/bash

# print commands as they're executed, exit on error
set -ve

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

scalene --reduced-profile benchmark/scripts/basic.py