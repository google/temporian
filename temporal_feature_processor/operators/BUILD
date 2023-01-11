package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":place_holder",
        ":simple_moving_average",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/data:event",
        "//temporal_feature_processor/implementation/pandas/operators:base",
    ],
)

py_library(
    name = "place_holder",
    srcs = ["place_holder.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporal_feature_processor:core_py_proto",
        "//temporal_feature_processor:operator_lib",
        "//temporal_feature_processor:sampling",
        "//temporal_feature_processor/data:event",
        "//temporal_feature_processor/data:feature",
        "//temporal_feature_processor/implementation/pandas/operators:base",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporal_feature_processor:core_py_proto",
        "//temporal_feature_processor:operator_lib",
        "//temporal_feature_processor/data:event",
        "//temporal_feature_processor/data:feature",
        "//temporal_feature_processor/implementation/pandas/operators:base",
    ],
)
