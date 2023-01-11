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
        ":base",
        "//temporal_feature_processor/implementation/pandas/operators/window",
    ],
)

py_library(
    name = "assign",
    srcs = ["assign.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/implementation/pandas/data:event",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/implementation/pandas/data:event",
    ],
)
