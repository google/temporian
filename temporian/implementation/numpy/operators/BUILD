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
    ],
)

py_library(
    name = "lag",
    srcs = ["lag.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:duration",
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
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
        ":arithmetic",
        ":assign",
        ":lag",
        ":select",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:arithmetic",
        "//temporian/implementation/numpy/data:event",
    ],
)
