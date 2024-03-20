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
        # already_there/apache_beam
        "//temporian/core/operators:base",
        "//temporian/beam:typing",
    ],
)

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":add_index",
        ":select",
        ":rename",
        ":prefix",
        ":leak",
        ":glue",
        ":filter",
        ":timestamps",
        ":filter_empty_index",
        ":cast",
        ":unique_timestamps",
        ":resample",
        "//temporian/beam/operators/window:moving_count",
        "//temporian/beam/operators/window:moving_max",
        "//temporian/beam/operators/window:moving_min",
        "//temporian/beam/operators/window:moving_standard_deviation",
        "//temporian/beam/operators/window:moving_sum",
        "//temporian/beam/operators/window:simple_moving_average",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:select",
    ],
)

py_library(
    name = "add_index",
    srcs = ["add_index.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/beam:typing",
        "//temporian/beam:implementation_lib",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:add_index",
    ],
)

py_library(
    name = "rename",
    srcs = ["rename.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:rename",
    ],
)


py_library(
    name = "prefix",
    srcs = ["prefix.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:prefix",
    ],
)


py_library(
    name = "leak",
    srcs = ["leak.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:leak",
    ],
)

py_library(
    name = "glue",
    srcs = ["glue.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:glue",
    ],
)


py_library(
    name = "filter",
    srcs = ["filter.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:filter",
    ],
)


py_library(
    name = "timestamps",
    srcs = ["timestamps.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:timestamps",
    ],
)


py_library(
    name = "filter_empty_index",
    srcs = ["filter_empty_index.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:filter_empty_index",
    ],
)


py_library(
    name = "cast",
    srcs = ["cast.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:cast",
    ],
)


py_library(
    name = "unique_timestamps",
    srcs = ["unique_timestamps.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:unique_timestamps",
    ],
)


py_library(
    name = "resample",
    srcs = ["resample.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:implementation_lib",
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:resample",
    ],
)
