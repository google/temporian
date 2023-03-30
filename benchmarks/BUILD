package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "complete",
    srcs = ["complete.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian",
    ],
)

# FAILS: ImportError: cannot import name 'sample' from 'temporian.implementation.numpy_cc.operators' (unknown location)
# Don't seem to be able to specify :complete as a build dependency, only as data
# If I add :complete or //temporian to `deps`:
# ERROR: /Users/ian/temporian/BUILD:14:8: in deps attribute of sh_test rule //:bench: py_library rule '//temporian:temporian' is misplaced here (expected sh_library)
# If I add them to `data` I get same ImportError as below
# sh_test(
#     name = "bench",
#     srcs = ["run.sh"],
#     data = [
#         "complete",
#         "//temporian",
#     ],
#     # deps = [
#     #     # "//temporian",
#     # ],
# )

# FAILS: ImportError: cannot import name 'sample' from 'temporian.implementation.numpy_cc.operators' (unknown location)
# Seems to be running on a different architecture where the C++ extension is not built
# genrule(
#     name = "run",
#     testonly = True,
#     # srcs = ["//benchmarks:run.sh"],
#     outs = ["a"],
#     cmd = "python benchmarks/complete.py",
#     tools = [
#         ":complete",
#         "//temporian",
#     ],
# )

# RUNS, but I need to wrap the scalene command around it :)
# py_test(
#     name = "complete",
#     srcs = ["complete.py"],
#     srcs_version = "PY3",
#     deps = [
#         "//temporian",
#     ],
# )

# HACKY: Run scalene from inside a python file, so that I can specify python deps while running it in the test architecture
# Also, I'm not seeing memory reporting in the dashboard
py_test(
    name = "run",
    srcs = ["run.py"],
    srcs_version = "PY3",
    deps = [
        ":complete",
    ],
)
