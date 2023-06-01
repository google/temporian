"""
Generate the code reference pages.

Source: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

SRC_PATH = Path("temporian")

paths = set()
non_parsable_imports = []

with open("temporian/__init__.py", "r", encoding="utf8") as f:
    lines = f.read().splitlines()

for line in lines:
    words = line.split(" ")

    # It is an import statement
    if words[0] == "from":
        # Remove trailing "as <name>" if it exists
        if words[-2] == "as":
            # If symbol was renamed to a private name, skip it
            if words[-1].startswith("_"):
                continue

            words = words[:-2]

        # It is a single-symbol import like "from <module> import <symbol>"
        if words[-2] == "import":
            module_path = Path(words[1].replace(".", "/"))

            # Check if the import is a dir module
            module_path_with_suffix = module_path / words[-1]
            if module_path_with_suffix.exists():
                module_path = module_path_with_suffix

            # Check if the import is a file module
            module_path_with_suffix = module_path / (words[-1] + ".py")
            if module_path_with_suffix.exists():
                module_path = module_path_with_suffix.with_suffix("")

            # If it's not a module import it is a normal symbol import
            # (function, class, etc.) so we add its whole module to the docs

            paths.add(module_path)

        else:
            non_parsable_imports.append(line)

if non_parsable_imports:
    raise RuntimeError(
        "`gen_ref_pages` failed to parse the following import statements in"
        f" the top-level __init__.py file: {non_parsable_imports}. Import"
        " statements in the top-level module must import a single symbol each,"
        " in the form `from <module> import <symbol>` or `from <module> import"
        " <symbol> as <name>`."
    )

for path in sorted(paths):
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

with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
