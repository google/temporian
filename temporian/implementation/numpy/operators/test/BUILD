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
    name = "unique_timestamps_test",
    srcs = ["unique_timestamps_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        # already_there/numpy
        "//temporian/io:pandas",
        "//temporian/core/operators:unique_timestamps",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators:unique_timestamps",
    ],
)

py_test(
    name = "since_last_test",
    srcs = ["since_last_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        # already_there/numpy
        "//temporian/io:pandas",
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:since_last",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/operators:since_last",
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
    name = "tick_test",
    srcs = ["tick_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:tick",
        "//temporian/implementation/numpy/operators:tick",
    ],
)

py_test(
    name = "join_test",
    srcs = ["join_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:join",
        "//temporian/implementation/numpy/operators:join",
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