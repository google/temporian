"""
Generates a docs/index.md on the fly from the README.md and fixes some of the links.

Source: https://github.com/tryolabs/norfair/blob/master/docs/gen_index.py
"""

import re

import mkdocs_gen_files

# read README on the root of the repo
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# fix `docs/src/` links
content = re.sub(r"docs\/src\/([\w\/-]+)\.py", r"./\1", content)

# remove "docs/src" from gifs and images
content = re.sub(r"\]\(/?docs/src/", r"](", content)

# remove "docs/src" from src fields in html
content = re.sub(r"src=\"/?docs/src/", 'src="', content)

# fix `temporian/__/__.py` links (by adding `reference/` and removing `.py`)
# so that they work in mkdocs.
content = re.sub(
    r"temporian\/([\w\/-]+)\.py", r"reference/temporian/\1", content
)

# Remove entire "## Documentation" and "## Contributing" sections
content = re.sub(r"## Documentation.*## ", "## ", content, flags=re.DOTALL)
content = re.sub(r"## Contributing.*## ", "## ", content, flags=re.DOTALL)

# write the index
with mkdocs_gen_files.open("index.md", "w") as fd:  #
    print(content, file=fd)
