"""
Generate the code reference pages.

This script traverses the markdown files under docs/src/reference and for each:
- If the file is not empty it is ignored and will appear in the docs as-is.
- If the file has no extension, it is interpreted as a placeholder for the
    top-level symbol with its same name and its reference page is generated in
    its same path.
    E.g., if an empty file named `docs/src/reference/temporianio/to_csv.md`
    exists, the reference page for `temporian.to_csv` will be generated in
    reference/temporian/io/to_csv/ in the docs.

Related: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

REFERENCE = Path("docs/src/reference")

# Sort files by depth and name for alphabetical order and subdirs at the bottom
paths = sorted(
    REFERENCE.rglob("*.md"), key=lambda p: str(len(p.parts)) + str(p)
)

for path in paths:
    path = Path(path)
    ref_path = path.relative_to(REFERENCE)
    nav_path = ref_path.with_suffix("")
    full_ref_path = "reference" / ref_path

    nav[list(nav_path.parts)] = ref_path.as_posix()

    # If file is empty we assume it's a top-level symbol and auto
    # generate the mkdocstring identifier for it
    if path.stat().st_size == 0:
        with mkdocs_gen_files.open(full_ref_path, "w") as fd:
            ident = "temporian." + nav_path.name
            fd.write(f"::: {ident}")


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
