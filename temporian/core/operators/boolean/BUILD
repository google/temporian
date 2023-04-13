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
    deps = [
        ":base_feature",
        ":base_scalar",
        ":equal_scalar",
    ],
)

py_library(
    name = "base_scalar",
    srcs = ["base_scalar.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "base_feature",
    srcs = ["base_feature.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "equal_scalar",
    srcs = ["equal_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base_scalar",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "equal_feature",
    srcs = ["equal_feature.py"],
    srcs_version = "PY3",
    deps = [
        ":base_feature",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
    ],
)
