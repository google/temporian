package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "numpy",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":implementation_lib",
        "//temporian/core:schedule",
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators",
    ],
)

py_library(
    name = "implementation_lib",
    srcs = ["implementation_lib.py"],
    srcs_version = "PY3",
)
