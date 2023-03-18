package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "test_df_to_event",
    srcs = ["test_df_to_event.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_test(
    name = "test_event_to_df",
    srcs = ["test_event_to_df.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_test(
    name = "test_sampling",
    srcs = ["test_sampling.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:sampling",
    ],
)
