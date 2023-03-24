package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "window",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/window:base",
        "//temporian/implementation/numpy/data:event",
    ],
)
