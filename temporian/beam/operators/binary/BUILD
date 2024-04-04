package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "binary",
    srcs = ["binary.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/binary:base",
        "//temporian/implementation/numpy/operators/binary:base",
    ],
)
