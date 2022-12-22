
load("//temporal_feature_processor/utils:compile.bzl", "all_proto_library")

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
    deps = [":core"],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":core_py_proto",
        # absl/logging dep,
    ],
)

# Proto
# ========

all_proto_library(
    name = "core_proto",
    srcs = ["core.proto"],
    deps = [],
)

# Tests
# =====

py_test(
    name = "temporal_feature_processor_test",
    srcs = ["temporal_feature_processor_test.py"],
    srcs_version = "PY3",
    deps = [
        ":temporal_feature_processor",
        # absl/logging dep,
        # absl/testing:absltest dep,
    ],
)
