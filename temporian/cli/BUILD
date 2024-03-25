package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binary
# =========

py_binary(
    name = "main",
    srcs = ["main.py"],
    srcs_version = "PY3",
    deps=["//temporian",":view",],
)


# Library
# =========

py_library(
    name = "view",
    srcs = ["view.py"],
    srcs_version = "PY3",
    deps=["//temporian",],
)
