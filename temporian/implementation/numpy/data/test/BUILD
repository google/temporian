package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "df_to_event_test",
    srcs = ["df_to_event_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_test(
    name = "event_to_df_test",
    srcs = ["event_to_df_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "event_test",
    srcs = ["event_test.py"],
    srcs_version = "PY3",
    deps = ["//temporian/implementation/numpy/data:event_set"],
)

py_test(
    name = "plotter_test",
    srcs = ["plotter_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:plotter",
    ],
)
