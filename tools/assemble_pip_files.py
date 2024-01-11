"""Assemble the files to create a pip package from Bazel's output."""

import glob
import os
from pathlib import Path
import platform
import shutil as s


SRC_BIN = "bazel-bin/temporian"
DST_PK = "tmp_package"
INIT_FILENAME = "__init__.py"

if platform.system() == "Windows":
    DST_EXTENSION = "pyd"
else:
    DST_EXTENSION = "so"


def rec_glob_copy(src_dir: str, dst_dir: str, pattern: str):
    """Copies the files matching a pattern from the src to the dst directory."""

    # TODO: Use "root_dir=src_dir" argument when >=python3.10
    os.makedirs(dst_dir, exist_ok=True)
    for fall in glob.glob(f"{src_dir}/{pattern}", recursive=True):
        frel = os.path.relpath(fall, src_dir)
        dst = f"{dst_dir}/{frel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s.copy(f"{src_dir}/{frel}", dst)


def main():
    # Remove and recreate the package directory
    if os.path.exists(DST_PK):
        try:
            s.rmtree(DST_PK)
        except Exception:
            print(
                "Fail to remove the existing dir with rmtree. Use rmdir"
                " instead."
            )
            os.system(f"rmdir /S /Q {DST_PK}")
    os.makedirs(DST_PK)

    # Individual files
    for f in [
        "README.md",
        "config/setup.py",
        "config/MANIFEST.in",
        "LICENSE",
    ]:
        s.copy(f, DST_PK)

    # Source files
    rec_glob_copy("temporian", f"{DST_PK}/temporian", "**/*.py")

    # Compiled binary
    s.copy(
        f"{SRC_BIN}/implementation/numpy_cc/operators/operators_cc.so",
        f"{DST_PK}/temporian/implementation/numpy_cc/operators/operators_cc.{DST_EXTENSION}",
    )

    # Generated protobuffer accessors
    s.copy(
        f"{SRC_BIN}/proto/core_pb2.py",
        f"{DST_PK}/temporian/proto/core_pb2.py",
    )

    # Create the missing __init__.py files
    for path, _, files in os.walk(f"{DST_PK}/temporian"):
        if INIT_FILENAME not in files:
            Path(f"{path}/{INIT_FILENAME}").touch()


if __name__ == "__main__":
    main()
