package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Libraries
# =========

py_library(
    name="io",
    srcs=["__init__.py"],
    srcs_version="PY3",
    deps=[":csv", ":pandas", ":numpy"],
)

py_library(
    name="csv",
    srcs=["csv.py"],
    srcs_version="PY3",
    deps=[
        # force/pandas
        ":pandas",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/utils:typecheck",
    ],
)

py_library(
    name="parquet",
    srcs=["parquet.py"],
    srcs_version="PY3",
    deps=[
        # force/pandas
        ":pandas",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/utils:typecheck",
    ],
)


py_library(
    name="pandas",
    srcs=["pandas.py"],
    srcs_version="PY3",
    deps=[
        # already_there/numpy
        # force/pandas
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_library(
    name="polars",
    srcs=["polars.py"],
    srcs_version="PY3",
    deps=[
        # already_there/numpy
        # force/polars
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_library(
    name="numpy",
    srcs=["numpy.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_library(
    name="tensorflow",
    srcs=["tensorflow.py"],
    srcs_version="PY3",
    deps=[
        # force/tensorflow
        # already_there/numpy
        "//temporian/core/data:dtype",
        ":format",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:dtype_normalization",
        "//temporian/core/operators:drop_index",
    ],
)

py_library(
    name="format",
    srcs=["format.py"],
    srcs_version="PY3",
    deps=[],
)
