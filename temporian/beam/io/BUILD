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
        ":np_array_coder",
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
        "//temporian/beam/io:dict",
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
        "//temporian/io:tensorflow",
        "//temporian/beam/io:dict",
        "//temporian/implementation/numpy/data:dtype_normalization",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/beam:typing",
        "//temporian/io:format",
    ],
)

py_library(
    name = "np_array_coder",
    srcs = ["np_array_coder.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        # already_there/numpy
    ],
)
