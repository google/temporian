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

# We need to be able to parse other files to allow wildcard imports
# Storing pair of (prefix, path) to parse in a stack
files_to_parse = [(None, SRC_PATH / "__init__.py")]

while files_to_parse:
    prefix, file = files_to_parse.pop()

    with open(file, "r", encoding="utf8") as f:
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

                # We only allow wildcard imports from modules explicitly named
                # api_symbols to prevent unwanted names in the public API
                if name == "*":
                    module_path = Path(words[1].replace(".", "/")).with_suffix(
                        ".py"
                    )
                    if module_path.stem == "api_symbols":
                        new_prefix = (
                            (prefix + ".") if prefix else ""
                        ) + module_path.parent.name
                        files_to_parse.append((new_prefix, module_path))
                        continue

                    non_parsable_imports.append(line)
                    continue

                # If symbol wasn't renamed, use its imported name
                if symbol is None:
                    symbol = name

                path = Path(words[1].replace(".", "/")) / name

                if prefix:
                    symbol = prefix + "." + symbol

                members.add((symbol, path))

            # It is a multi-symbol import statement, error will be raised below
            else:
                non_parsable_imports.append(line)

if non_parsable_imports:
    raise RuntimeError(
        "`gen_ref_pages` failed to parse the following import statements in"
        f" the top-level __init__.py file: {non_parsable_imports}. Import"
        " statements in the top-level module must import a single symbol each,"
        " in the form `from <module> import <symbol>`, `from <module> import"
        " <symbol> as <name>`, or `from <module> import *`."
    )

nav["temporian"] = "index.md"

for symbol, path in sorted(members):
    symbol_path = Path(symbol.replace(".", "/"))
    symbol_name = symbol_path.name
    src_path = SRC_PATH / symbol_name

    doc_path = SRC_PATH / symbol_path
    parts = list(doc_path.parts)
    doc_path = doc_path.with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(list(src_path.parts))
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
