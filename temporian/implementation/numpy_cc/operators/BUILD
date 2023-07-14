load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pybind_library(
    name = "common",
    hdrs = ["common.h"],
)

pybind_library(
    name = "since_last",
    srcs = ["since_last.cc"],
    hdrs = ["since_last.h"],
    deps = [":common"],
)

pybind_library(
    name = "window",
    srcs = ["window.cc"],
    hdrs = ["window.h"],
    deps = [":common"],
)

pybind_library(
    name = "resample",
    srcs = ["resample.cc"],
    hdrs = ["resample.h"],
    deps = [":common"],
)

pybind_library(
    name = "join",
    srcs = ["join.cc"],
    hdrs = ["join.h"],
    deps = [":common"],
)

pybind_library(
    name = "add_index",
    srcs = ["add_index.cc"],
    hdrs = ["add_index.h"],
    deps = [":common"],
)

pybind_extension(
    name = "operators_cc",
    srcs = ["pyinit.cc"],
    deps = [
        ":add_index",
        ":join",
        ":resample",
        ":since_last",
        ":window",
    ],
)

py_library(
    name = "operators_cc",
    data = [":operators_cc.so"],
)
