package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

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
    name = "test_propagate",
    srcs = ["test_propagate.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_rename",
    srcs = ["test_rename.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_select",
    srcs = ["test_select.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_select_index_values",
    srcs = ["test_select_index_values.py"],
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

py_test(
    name = "tick_calendar_test",
    srcs = ["tick_calendar_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:tick_calendar",
    ],
)

py_test(
    name = "test_arithmetic_multi_index",
    srcs = ["test_arithmetic_multi_index.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)
