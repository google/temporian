package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # already_there/absl/logging
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/core:serialization",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/logging
        # already_there/absl/testing:absltest
        # already_there/bokeh
        # already_there/matplotlib
        "//temporian",
    ],
)

py_test(
    name = "api_beam_test",
    srcs = ["api_beam_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/logging
        # already_there/absl/testing:absltest
        # already_there/bokeh
        # already_there/matplotlib
        "//temporian",
        "//temporian/beam",
    ],
)

py_test(
    name = "doc_test",
    srcs = ["doc_test.py"],
    data = glob(["**/*.md"]),
    srcs_version = "PY3",
    deps = [
        "//temporian",
        # already_there/absl/testing:absltest
    ],
)

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        "//temporian",
        ":utils",
    ],
)
