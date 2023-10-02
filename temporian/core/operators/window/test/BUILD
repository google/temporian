package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_cumsum",
    srcs = ["test_cumsum.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_moving_sum",
    srcs = ["test_moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)
