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
    name = "test_add_index",
    srcs = ["test_add_index.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_filter",
    srcs = ["test_filter.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_glue",
    srcs = ["test_glue.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_lag",
    srcs = ["test_lag.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_leak",
    srcs = ["test_leak.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
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
    name = "test_prefix",
    srcs = ["test_prefix.py"],
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
    name = "test_resample",
    srcs = ["test_resample.py"],
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
    name = "test_until_next",
    srcs = ["test_until_next.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/test:utils",
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
    name = "test_arithmetic",
    srcs = ["test_arithmetic.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_where",
    srcs = ["test_where.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_arithmetic_scalar",
    srcs = ["test_arithmetic_scalar.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_logical",
    srcs = ["test_logical.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_relational",
    srcs = ["test_relational.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_relational_scalar",
    srcs = ["test_relational_scalar.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_unary",
    srcs = ["test_unary.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_filter_moving_count",
    srcs = ["test_filter_moving_count.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_tick_calendar",
    srcs = ["test_tick_calendar.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_enumerate",
    srcs = ["test_enumerate.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_cast",
    srcs = ["test_cast.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_combine",
    srcs = ["test_combine.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "test_timestamps",
    srcs = ["test_timestamps.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)
