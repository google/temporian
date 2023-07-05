package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "check_install",
    srcs = ["check_install.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # already_there/pandas
        "//temporian",
    ],
)

# Tests
# =========

py_test(
    name = "build_cleaner_test",
    srcs = [
        "build_cleaner.py",
        "build_cleaner_test.py",
    ],
    srcs_version = "PY3",
)
