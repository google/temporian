package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core:serialization",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
    ],
)

# Tests
# =====
py_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:pandas",
        "//temporian/implementation/numpy/operators/binary",
        ":utils",
    ],
)

py_test(
    name = "relational_test",
    srcs = ["relational_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators/binary",
        ":utils",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators/binary",
        ":utils",
    ],
)

py_test(
    name = "arithmetic_scalar_test",
    srcs = ["arithmetic_scalar_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators/scalar",
        ":utils",
    ],
)

py_test(
    name = "relational_scalar_test",
    srcs = ["relational_scalar_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators/scalar",
        ":utils",
    ],
)

py_test(
    name = "cast_test",
    srcs = ["cast_test.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core:evaluation",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:cast",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators:cast",
    ],
)

py_test(
    name = "unary_test",
    srcs = ["unary_test.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/io:pandas",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators:unary",
    ],
)

py_test(
    name = "begin_test",
    srcs = ["begin_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:begin",
        "//temporian/implementation/numpy/operators:begin",
    ],
)

py_test(
    name = "end_test",
    srcs = ["end_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:end",
        "//temporian/implementation/numpy/operators:end",
    ],
)

py_test(
    name = "timestamps_test",
    srcs = ["timestamps_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:timestamps",
        "//temporian/implementation/numpy/operators:timestamps",
    ],
)

py_test(
    name = "enumerate_test",
    srcs = ["enumerate_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:enumerate",
        "//temporian/implementation/numpy/operators:enumerate",
    ],
)

py_test(
    name = "combine_test",
    srcs = ["combine_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:combine",
        "//temporian/implementation/numpy/operators:combine",
    ],
)

py_test(
    name = "fft_test",
    srcs = ["fft_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:fast_fourier_transform",
        "//temporian/implementation/numpy/operators:fast_fourier_transform",
    ],
)

py_test(
    name = "filter_moving_count_test",
    srcs = ["filter_moving_count_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:filter_moving_count",
        "//temporian/implementation/numpy/operators:filter_moving_count",
    ],
)

py_test(
    name = "tick_calendar_test",
    srcs = ["tick_calendar_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/test:utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:tick_calendar",
        "//temporian/implementation/numpy/operators:tick_calendar",
    ],
)
