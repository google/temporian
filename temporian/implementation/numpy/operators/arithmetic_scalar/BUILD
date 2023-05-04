package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "arithmetic_scalar",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/arithmetic_scalar",
        "//temporian/core/operators/arithmetic_scalar:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "add_scalar",
    srcs = ["add_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:add_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "equal_scalar",
    srcs = ["equal_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:equal_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "subtract_scalar",
    srcs = ["subtract_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:subtract_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "multiply_scalar",
    srcs = ["multiply_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:multiply_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "divide_scalar",
    srcs = ["divide_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:divide_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "floordiv_scalar",
    srcs = ["floordiv_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic_scalar:floordiv_scalar",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)
