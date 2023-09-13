package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "dict",
    srcs = ["dict.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:dtype_normalization",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/beam:typing",
        "//temporian/io:format",
    ],
)

py_library(
    name = "csv",
    srcs = ["csv.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/beam:typing",
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:dtype_normalization",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/beam:typing",
        "//temporian/io:format",
    ],
)
