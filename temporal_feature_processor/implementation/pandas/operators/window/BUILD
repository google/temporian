package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "window",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":simple_moving_average",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/implementation/pandas/operators:base",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/implementation/pandas:sampling",
        "//temporal_feature_processor/implementation/pandas/data:event",
    ],
)
