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
    deps = [
        ":dtype",
        ":event",
        ":feature",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    srcs_version = "PY3",
)

py_library(
    name = "event",
    srcs = ["event.py"],
    srcs_version = "PY3",
    deps = [
        ":feature",
        "//temporal_feature_processor:sampling",
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor:sampling",
    ],
)
