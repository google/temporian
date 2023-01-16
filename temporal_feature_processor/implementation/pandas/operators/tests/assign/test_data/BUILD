package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# All test data
# =============

py_library(
    name = "test_data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":different_index",
        ":repeated_timestamps",
        ":with_idx_more_timestamps",
        ":with_idx_same_timestamps",
    ],
)

# Indiviudual test data
# =====================

py_library(
    name = "different_index",
    srcs = ["different_index.py"],
    srcs_version = "PY3",
)

py_library(
    name = "repeated_timestamps",
    srcs = ["repeated_timestamps.py"],
    srcs_version = "PY3",
)

py_library(
    name = "with_idx_more_timestamps",
    srcs = ["with_idx_more_timestamps.py"],
    srcs_version = "PY3",
)

py_library(
    name = "with_idx_same_timestamps",
    srcs = ["with_idx_same_timestamps.py"],
    srcs_version = "PY3",
)
