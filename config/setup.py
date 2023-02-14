# Copyright 2023 Google LLC.
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

from setuptools import setup

setup(
    name="temporian",
    version="0.0.0",
    description=(
        "Temporian is a library to pre-process temporal signals before their"
        " use as input features with off-the-shelf tabular machine learning"
        " libraries (e.g., TensorFlow Decision Forests)."
    ),
    long_description="",
    url="https://github.com/google/temporian",
    project_urls={
        "Bug Tracker": "https://github.com/google/temporian/issues",
    },
    # Update with members of the Google + Tryolabs collaboration.
    author=(
        "Mathieu Guillame-Bert, Ian Spektor, Guillermo Etchebarne, Diego"
        " Marvid, Richard Stotz"
    ),
    author_email=(
        "gbm@google.com, ispektor@tryolabs.com, getchebarne@tryolabs.com,"
        " dmarvid@tryolabs.com, richardstotz@google.com"
    ),
    packages=[],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    license="Apache 2.0",
)
