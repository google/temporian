package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "moving_sum",
    srcs = ["moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        "//temporian/beam:typing",
        "//temporian/beam:implementation_lib",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators:base",
    ],
)
