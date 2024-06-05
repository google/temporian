load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_library_test")

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

cc_library(
    name = "custom_heap",
    srcs = [],
    hdrs = ["custom_heap.h"],
    deps = [":common"],
)

pybind_library(
    name = "window",
    srcs = ["window.cc"],
    hdrs = ["window.h"],
    deps = [":common", "custom_heap"],
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

pybind_library(
    name = "tick_calendar",
    srcs = ["tick_calendar.cc"],
    hdrs = ["tick_calendar.h"],
    deps = [":common", ":tick_calendar_utils"],
)

pybind_library(
    name = "until_next",
    srcs = ["until_next.cc"],
    hdrs = ["until_next.h"],
    deps = [":common"],
)

pybind_library(
    name = "filter_moving_count",
    srcs = ["filter_moving_count.cc"],
    hdrs = ["filter_moving_count.h"],
    deps = [":common"],
)


pybind_library(
    name = "calendar",
    srcs = ["calendar.cc"],
    hdrs = ["calendar.h"],
    deps = [
        "@com_google_absl//absl/time",
        ":common",
        ":tick_calendar_utils",
    ],
)

pybind_extension(
    name = "operators_cc",
    srcs = ["pyinit.cc"],
    deps = [
        ":add_index",
        ":filter_moving_count",
        ":join",
        ":resample",
        ":since_last",
        ":tick_calendar",
        ":until_next",
        ":window",
        ":calendar",
    ],
)

py_library(
    name = "operators_cc",
    data = [":operators_cc.so"],
)

cc_library(
    name = "tick_calendar_utils",
    srcs = ["tick_calendar_utils.cc"],
    hdrs = ["tick_calendar_utils.h"],
    deps = ["@com_google_absl//absl/time", ":common"],
)

cc_test(
    name = "tick_calendar_utils_test",
    srcs = ["tick_calendar_utils_test.cc"],
    deps = [
        ":tick_calendar_utils",
        "@com_google_googletest//:gtest_main"
        ],
)
