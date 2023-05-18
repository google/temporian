"""
Generate the code reference pages.

Source: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

SRC_PATH = Path("temporian")

skipped_paths = []

for path in sorted(SRC_PATH.rglob("*.py")):
    if path.parent.name not in ["test", "tests"]:
        module_path = path.relative_to(SRC_PATH.parent).with_suffix("")
        doc_path = path.relative_to(SRC_PATH.parent).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = list(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(parts)
            print("::: " + identifier, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)
    else:
        skipped_paths.append(str(path))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
