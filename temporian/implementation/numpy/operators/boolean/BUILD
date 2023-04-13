package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "boolean",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/boolean:base_feature",
        "//temporian/core/operators/boolean:base_scalar",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "equal",
    srcs = ["equal.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/boolean:equal_feature",
        "//temporian/core/operators/boolean:equal_scalar",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)
