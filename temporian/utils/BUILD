load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Bzl Libraries
# =============

bzl_library(
    name = "compile_bzl",
    srcs = ["compile.bzl"],
)

# Libraries
# =========

py_library(
    name = "string",
    srcs = ["string.py"],
    srcs_version = "PY3",
)

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
)

py_library(
    name = "typecheck",
    srcs = ["typecheck.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:event_set_ops",
        "//temporian/core/data:node",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "golden",
    srcs = ["golden.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl
    ],
)
