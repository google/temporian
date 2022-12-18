
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "temporal_feature_processor",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    tags = ["temporal_feature_processor"],
    deps = [":core"],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    tags = ["temporal_feature_processor"],
    deps = [
        # absl/logging dep,
    ],
)

# Tests
# =====

py_test(
    name = "temporal_feature_processor_test",
    srcs = ["temporal_feature_processor_test.py"],
    srcs_version = "PY3",
    tags = ["temporal_feature_processor"],
    deps = [
        ":temporal_feature_processor",
        # absl/logging dep,
        # absl/testing:absltest dep,
    ],
)
