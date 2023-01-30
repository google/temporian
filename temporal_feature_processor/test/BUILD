
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
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporal_feature_processor",
    ],
)

py_test(
    name = "prototype_test",
    srcs = ["prototype_test.py"],
    data = ["test_data/prototype"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporal_feature_processor/core:evaluator",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/core/operators:assign",
        "//temporal_feature_processor/implementation/pandas/data:event",
    ],
)
