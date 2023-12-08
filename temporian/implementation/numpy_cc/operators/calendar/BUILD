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

pybind_library(
    name = "day_of_month",
    srcs = ["day_of_month.cc"],
    hdrs = ["day_of_month.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "hour",
    srcs = ["hour.cc"],
    hdrs = ["hour.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "minute",
    srcs = ["minute.cc"],
    hdrs = ["minute.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "second",
    srcs = ["second.cc"],
    hdrs = ["second.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "day_of_year",
    srcs = ["day_of_year.cc"],
    hdrs = ["day_of_year.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "day_of_week",
    srcs = ["day_of_week.cc"],
    hdrs = ["day_of_week.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)

pybind_library(
    name = "isoweek",
    srcs = ["isoweek.cc"],
    hdrs = ["isoweek.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//temporian/implementation/numpy_cc/operators:common",
        ":common"
    ],
)