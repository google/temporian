package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "pandas",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/operators:base",
        "//temporian/implementation/pandas/data:event",
        "//temporian/implementation/pandas/operators:core_mapping",
    ],
)
