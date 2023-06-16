package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/beam",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
    ],
    data = ["//temporian/test/test_data"],
)
