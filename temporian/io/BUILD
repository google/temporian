package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "io",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":csv",
        ":pandas",
    ],
)

py_library(
    name = "csv",
    srcs = ["csv.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/pandas
        ":pandas",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "pandas",
    srcs = ["pandas.py"],
    srcs_version = "PY3",
    deps = [
        # force/pandas
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)
