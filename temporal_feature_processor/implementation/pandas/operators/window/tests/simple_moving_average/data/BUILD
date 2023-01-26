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
    ],
)
