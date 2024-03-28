package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "scalar",
    srcs = ["scalar.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/scalar:base",
        "//temporian/implementation/numpy/operators/scalar:base",
    ],
)
