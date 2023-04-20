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
    deps = [
        ":add",
        ":base",
        ":divide",
        ":equal",
        ":floordiv",
        ":multiply",
        ":negate",
        ":subtract",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "add",
    srcs = ["add.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "equal",
    srcs = ["equal.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
    ],
)

py_library(
    name = "subtract",
    srcs = ["subtract.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "multiply",
    srcs = ["multiply.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "divide",
    srcs = ["divide.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "floordiv",
    srcs = ["floordiv.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "negate",
    srcs = ["negate.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
    ],
)
