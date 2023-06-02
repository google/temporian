package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========
py_library(
    name = "read_event_set",
    srcs = ["read_event_set.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/pandas
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_library(
    name = "save_event_set",
    srcs = ["save_event_set.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event_set",
    ],
)
