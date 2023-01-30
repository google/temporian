
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        ":backends",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/core/operators:base",
        "//temporal_feature_processor/implementation/pandas/data:event",
    ],
)

py_library(
    name = "backends",
    srcs = ["backends.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/implementation/pandas:evaluator",
        "//temporal_feature_processor/implementation/pandas/data:event",
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
    deps = [
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/core/operators:base",
    ],
)
