package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        ":backends",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:base",
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "backends",
    srcs = ["backends.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/pandas:evaluator",
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "operator_lib",
    srcs = ["operator_lib.py"],
    srcs_version = "PY3",
)

py_library(
    name = "processor",
    srcs = ["processor.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
    ],
)

py_library(
    name = "serialize",
    srcs = ["serialize.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
    ],
)
