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
        ":tensorflow",
    ],
)

py_library(
    name = "csv",
    srcs = ["csv.py"],
    srcs_version = "PY3",
    deps = [
        # force/pandas
        ":pandas",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "pandas",
    srcs = ["pandas.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # force/pandas
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # force/tensorflow
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)
