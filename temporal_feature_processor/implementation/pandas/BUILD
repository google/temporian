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
    deps = [
        ":sampling",
        "//temporal_feature_processor/implementation/pandas/operators",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
)
