package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":graph",
        ":schedule",
        ":typing",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/core/operators:leak",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "operator_lib",
    srcs = ["operator_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:base",
    ],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
    ],
)

py_library(
    name = "graph",
    srcs = ["graph.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":typing",
        "//temporian/core:compilation",
        "//temporian/core:graph",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "compilation",
    srcs = ["compilation.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "event_set_ops",
    srcs = ["event_set_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:duration",
    ],
)

py_library(
    name = "typing",
    srcs = ["typing.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:event_set_ops",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:event_set",
    ],
)
