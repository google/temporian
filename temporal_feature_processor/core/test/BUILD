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

py_test(
    name = "processor_test",
    srcs = ["processor_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        "//temporal_feature_processor/core:processor",
        "//temporal_feature_processor/core/data:dtype",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/core/operators:base",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)
