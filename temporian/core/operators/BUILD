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
    deps = [
        ":assign",
        ":place_holder",
        ":simple_moving_average",
    ],
)

py_library(
    name = "assign",
    srcs = ["assign.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "place_holder",
    srcs = ["place_holder.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)
