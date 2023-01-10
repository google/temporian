
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
    deps = [
        ":core",
        ":dtype",
        ":evaluator",
        ":event",
        ":feature",
        ":operator",
        ":operator_lib",
        ":processor",
        "//temporal_feature_processor/operators",
    ],
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

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        ":event",
        # pandas dep,
    ],
)

py_library(
    name = "operator_lib",
    srcs = ["operator_lib.py"],
    srcs_version = "PY3",
)

py_library(
    name = "processor",
    srcs = ["processor.py"],
    srcs_version = "PY3",
)

py_library(
    name = "operator",
    srcs = ["operator.py"],
    srcs_version = "PY3",
    deps = [":event"],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY3",
    deps = [":sampling"],
)

py_library(
    name = "event",
    srcs = ["event.py"],
    srcs_version = "PY3",
    deps = [
        ":feature",
        ":sampling",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    srcs_version = "PY3",
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
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
        # pandas dep,
    ],
)
