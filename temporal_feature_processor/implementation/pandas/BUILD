package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "pandas",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/operators:base",
        "//temporal_feature_processor/implementation/pandas/data:event",
        "//temporal_feature_processor/implementation/pandas/operators:core_mapping",
    ],
)
