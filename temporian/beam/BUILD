package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Libraries
# =========

py_library(
    name="beam",
    srcs=["__init__.py"],
    srcs_version="PY3",
    deps=[":io", ":evaluation"],
)

py_library(
    name="io",
    srcs=["io.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
    ],
)

py_library(
    name="evaluation",
    srcs=["evaluation.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/core/data:node",
        "//temporian/beam/operators",
        ":implementation_lib",
    ],
)

py_library(
    name="implementation_lib",
    srcs=["implementation_lib.py"],
    srcs_version="PY3",
)
