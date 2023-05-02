package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "arithmetic",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/arithmetic",
        "//temporian/core/operators/arithmetic:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "add",
    srcs = ["add.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:add",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "subtract",
    srcs = ["subtract.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:subtract",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "multiply",
    srcs = ["multiply.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:multiply",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "divide",
    srcs = ["divide.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:divide",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "equal",
    srcs = ["equal.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:equal",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "floordiv",
    srcs = ["floordiv.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:floordiv",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)
