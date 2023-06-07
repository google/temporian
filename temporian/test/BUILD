package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

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
    name = "docstring_test",
    srcs = ["docstring_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian",
    ],
)


py_test(
    name = "prototype_test_numpy",
    srcs = ["prototype_test_numpy.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        "//temporian",
    ],
    data = ["//temporian/test/test_data"],
)

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/pandas
        "//temporian",
    ],
    data = ["//temporian/test/test_data"],
)

py_test(
    name = "user_guide_test",
    srcs = ["user_guide_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian",
    ],
)
