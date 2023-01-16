
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "operator_test",
    srcs = ["operator_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/core/operators:base",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)
