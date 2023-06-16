package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Tests
# =====

py_test(
    name="io_test",
    srcs=["io_test.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/beam:io",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
    ],
    data=["//temporian/test/test_data"],
)

py_test(
    name="evaluation_test",
    srcs=["evaluation_test.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/beam:io",
        "//temporian/beam:evaluation",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
        "//temporian/implementation/numpy/operators/test:test_util",
        ":utils",
    ],
    data=["//temporian/test/test_data"],
)

# Libraries
# =========

py_library(
    name="utils",
    srcs=["utils.py"],
    srcs_version="PY3",
    deps=[],
)
