package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian/implementation/pandas/operators/window:simple_moving_average",
        "//temporian/implementation/pandas/operators/window/tests/simple_moving_average/data:diff_sampling",
        "//temporian/implementation/pandas/operators/window/tests/simple_moving_average/data:many_events_per_day",
        "//temporian/implementation/pandas/operators/window/tests/simple_moving_average/data:many_features",
        "//temporian/implementation/pandas/operators/window/tests/simple_moving_average/data:no_index",
        "//temporian/implementation/pandas/operators/window/tests/simple_moving_average/data:same_sampling",
    ],
)
