"""
Generate the code reference pages.

Source: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path
from typing import Set, Tuple

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

SRC_PATH = Path("temporian")

# Stores symbol and path of each public API member
members: Set[Tuple[str, Path]] = set()

non_parsable_imports = []

with open("temporian/__init__.py", "r", encoding="utf8") as f:
    lines = f.read().splitlines()

for line in lines:
    words = line.split(" ")

    # It is an import statement
    if words[0] == "from":
        # Remove trailing "as <name>" if it exists and save symbol's name
        symbol = None
        if words[-2] == "as":
            # If symbol was renamed to a private name, skip it
            if words[-1].startswith("_"):
                continue

            symbol = words[-1]
            words = words[:-2]

        # `words` is now in the form "from module.submodule import symbol"
        if words[-2] == "import":
            name = words[-1]

            # TODO: handle wildcard imports
            if name == "*":
                continue

            # If symbol wasn't renamed, use its imported name
            if symbol is None:
                symbol = name

            path = Path(words[1].replace(".", "/")) / name

            members.add((symbol, path))

        # It is a multi-symbol import statement, error will be raised below
        else:
            non_parsable_imports.append(line)

print(members)

if non_parsable_imports:
    raise RuntimeError(
        "`gen_ref_pages` failed to parse the following import statements in"
        f" the top-level __init__.py file: {non_parsable_imports}. Import"
        " statements in the top-level module must import a single symbol each,"
        " in the form `from <module> import <symbol>` or `from <module> import"
        " <symbol> as <name>`."
    )

for symbol, path in sorted(members):
    module_path = path.relative_to(SRC_PATH.parent).with_suffix("")
    doc_path = Path(SRC_PATH, symbol).with_suffix(".md")

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    full_doc_path = Path("reference", doc_path)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print("# tp." + symbol, file=fd)
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
