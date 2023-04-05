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
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "addition",
    srcs = ["addition.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:addition",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "subtraction",
    srcs = ["subtraction.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:subtraction",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "multiplication",
    srcs = ["multiplication.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:multiplication",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "division",
    srcs = ["division.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/arithmetic:division",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)
