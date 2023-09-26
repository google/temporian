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
        ":evaluation",
        "//temporian/beam/io:csv",
        "//temporian/beam/io:dict",
        "//temporian/beam/io:tensorflow",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        "//temporian/core:evaluation",
        ":implementation_lib",
        "//temporian/beam/operators",
        "//temporian/core/data:node",
        ":typing",
    ],
)

py_library(
    name = "implementation_lib",
    srcs = ["implementation_lib.py"],
    srcs_version = "PY3",
)

py_library(
    name = "typing",
    srcs = ["typing.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/core:typing",
    ],
)
