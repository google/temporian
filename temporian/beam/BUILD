package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "beam",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/logging
        ":evaluation",
        ":io",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        "//temporian/core:evaluation",
        "//temporian/beam:io",
        ":implementation_lib",
        "//temporian/beam/operators",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "implementation_lib",
    srcs = ["implementation_lib.py"],
    srcs_version = "PY3",
)
