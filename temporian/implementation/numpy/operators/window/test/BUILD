package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian/implementation/numpy/operators/window:simple_moving_average",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:dtype",
        "//temporian/core/operators:simple_moving_average",
    ],
)
