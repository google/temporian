
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
        ":assign",
        ":place_holder",
        ":simple_moving_average",
    ],
)

py_library(
    name = "assign",
    srcs = ["assign.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporal_feature_processor/core:operator_lib",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/implementation/pandas/operators:assign",
        "//temporal_feature_processor/implementation/pandas/operators:base",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/implementation/pandas/operators:base",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)

py_library(
    name = "place_holder",
    srcs = ["place_holder.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporal_feature_processor/core:operator_lib",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/implementation/pandas/operators:base",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporal_feature_processor/core:operator_lib",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/implementation/pandas/operators:base",
        "//temporal_feature_processor/implementation/pandas/operators/window:simple_moving_average",
        "//temporal_feature_processor/proto:core_py_proto",
    ],
)
