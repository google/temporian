package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "propagate_test",
    srcs = ["propagate_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:propagate",
    ],
)

py_test(
    name = "join_test",
    srcs = ["join_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:join",
    ],
)

py_test(
    name = "fft_test",
    srcs = ["fft_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:fast_fourier_transform",
    ],
)

py_test(
    name = "test_map",
    srcs = ["test_map.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "until_next_test",
    srcs = ["until_next_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:until_next",
    ],
)
