package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "dtypes",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":api_symbols",
    ],
)

py_library(
    name = "api_symbols",
    srcs = ["api_symbols.py"],
    srcs_version = "PY3",
    deps = [
        ":dtype",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    srcs_version = "PY3",
)
