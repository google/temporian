
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
        ":place_holder",
        ":simple_moving_average",
    ],
)

py_library(
    name = "place_holder",
    srcs = ["place_holder.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor:core_py_proto",
        "//temporal_feature_processor:event",
        "//temporal_feature_processor:feature",
        "//temporal_feature_processor:operator",
        "//temporal_feature_processor:operator_lib",
        "//temporal_feature_processor:sampling",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor:core_py_proto",
        "//temporal_feature_processor:event",
        "//temporal_feature_processor:feature",
        "//temporal_feature_processor:operator",
        "//temporal_feature_processor:operator_lib",
    ],
)
