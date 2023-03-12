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
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/core/operators:assign"
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event",
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
        ":simple_moving_average"
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/pandas/data:event",
    ],
)

py_library(
    name = "sum",
    srcs = ["sum.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:sum",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # numpy dep,
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/core/data:duration",
        "//temporian/core/operators:simple_moving_average",
    ],
)
