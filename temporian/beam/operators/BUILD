package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:io",
        "//temporian/core/operators:base",
    ],
)

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":add_index",
        ":select",
        "//temporian/beam/operators/window:moving_sum",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        "//temporian/beam:io",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:select",
        "//temporian/beam:implementation_lib",
    ],
)

py_library(
    name = "add_index",
    srcs = ["add_index.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/beam:implementation_lib",
        "//temporian/beam:io",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:add_index",
    ],
)
