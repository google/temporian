"""
Generates a docs/contributing.md on the fly from the CONTRIBUTING.md and fixes some of the links.

Source: https://github.com/tryolabs/norfair/blob/master/docs/gen_index.py
"""

import re

import mkdocs_gen_files

# read CONTRIBUTING on the root of the repo
with open("CONTRIBUTING.md", "r", encoding="utf-8") as f:
    content = f.read()

# fix links to relative paths under temporian/, benchmark/, docs/ and tools/
for dir in ["temporian", "benchmark", "docs", "tools"]:
    content = re.sub(
        rf"\({dir}\/([\w\/-]+)",
        rf"(https://github.com/google/temporian/blob/main/{dir}/\1",
        content,
    )

# write the file
with mkdocs_gen_files.open("contributing.md", "w") as fd:  #
    print(content, file=fd)
