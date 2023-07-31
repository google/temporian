package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/beam:io",
        "//temporian/core/data:node",
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators:select",
        "//temporian/core/operators/window:moving_sum",
    ],
)

# Libraries
# =========

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl
        # already_there/apache_beam
        "//temporian/beam:io",
        "//temporian/beam:evaluation",
        "//temporian/io:csv",
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/test:test_util",
    ],
)
