package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "window",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # numpy dep,
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/core/data:duration",
        "//temporian/core/operators:simple_moving_average",
    ],
)

# Tests
# =====
py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        ":simple_moving_average",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:dtype",
        "//temporian/core/operators:simple_moving_average",
    ],
)
