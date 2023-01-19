package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "event",
    srcs = ["event.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
    ],
)
