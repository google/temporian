
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# All test data
# =============

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

# Individual test data
# =====================

py_library(
    name = "no_index",
    srcs = ["no_index.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/data:sampling",
    ],
)

py_library(
    name = "same_sampling",
    srcs = ["same_sampling.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/data:sampling",
    ],
)

py_library(
    name = "diff_sampling",
    srcs = ["diff_sampling.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/data:sampling",
    ],
)

py_library(
    name = "many_events_per_day",
    srcs = ["many_events_per_day.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/data:sampling",
    ],
)

py_library(
    name = "many_features",
    srcs = ["many_features.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/data:sampling",
    ],
)
