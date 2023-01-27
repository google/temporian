
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
        "//temporal_feature_processor/implementation/pandas/operators/window:simple_moving_average",
        "//temporal_feature_processor/implementation/pandas/operators/window/tests/simple_moving_average/data:diff_sampling",
        "//temporal_feature_processor/implementation/pandas/operators/window/tests/simple_moving_average/data:many_events_per_day",
        "//temporal_feature_processor/implementation/pandas/operators/window/tests/simple_moving_average/data:many_features",
        "//temporal_feature_processor/implementation/pandas/operators/window/tests/simple_moving_average/data:no_index",
        "//temporal_feature_processor/implementation/pandas/operators/window/tests/simple_moving_average/data:same_sampling",
    ],
)
