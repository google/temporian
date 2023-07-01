package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Libraries
# =========

py_library(
    name="check_install",
    srcs=["check_install.py"],
    srcs_version="PY3",
    deps=[
        "//temporian",
    ],
)


# Tests
# =====

py_test(
    name="build_cleaner_test",
    srcs=["build_cleaner_test.py", "build_cleaner.py"],
    srcs_version="PY3",
)
