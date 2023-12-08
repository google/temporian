load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pybind_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)

pybind_library(
    name = "year",
    srcs = ["year.cc"],
    hdrs = ["year.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "month",
    srcs = ["month.cc"],
    hdrs = ["month.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)
