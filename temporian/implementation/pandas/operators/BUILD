package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "assign",
    srcs = ["assign.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/implementation/pandas:utils",
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "sum",
    srcs = ["sum.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:sum",
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "core_mapping",
    srcs = ["core_mapping.py"],
    srcs_version = "PY3",
    deps = [
        ":assign",
        ":select",
        ":sum",
        "//temporian/implementation/pandas/operators/window:simple_moving_average",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/implementation/pandas/data:event",
    ],
)
