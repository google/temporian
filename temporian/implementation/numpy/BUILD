package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "numpy",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        ":implementation_lib",
        "//temporian/core/data:event",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_library(
    name = "implementation_lib",
    srcs = ["implementation_lib.py"],
    srcs_version = "PY3",
)
