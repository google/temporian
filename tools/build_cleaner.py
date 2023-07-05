"""Find the missing and extra dependencies in Bazel Python build rules.

This program does not modify BUILD files. Instead, it lists the dependencies to
add and remove.

How to use the build cleaner.

    # Run the following command.
    python tools/build_cleaner.py

If case of a false positive, you might have to update the configuration fields
BUILT_IN_MODULES, THIRD_PARTY_MODULES, or EXTRA_SOURCE_TO_RULE.

Limitations
    - The build cleaner only infer the dependencies of py_library rules. For
        other rules (e.g. py proto, cc wrapper), the mapping should be
        specified manually in EXTRA_SOURCE_TO_RULE.
    - In some cases, import of module and import of symbol cannot be
        differentiated without parsing the python source code. In this case,
        build cleaner does not try to parse the python source code and rely on a
        permissive heuristic (possible false negative). See
        "list_possible_source_of_import" for details.
"""

from collections import defaultdict
import ast
from dataclasses import dataclass
import re
import os
from typing import List, Any, Tuple, Optional, Dict

import colorama
from colorama import Fore

# Bazel rules to scan for dependencies.
ALLOWED_BUILD_RULES = {
    "py_library",
    "py_binary",
}

# Python modules that do not require checking / dependencies.
BUILT_IN_MODULES = {
    "typing",
    "os",
    "abc",
    "datetime",
    "math",
    "csv",
    "argparse",
    "collections",
    "enum",
    "__future__",
    "sys",
    "dataclasses",
    "time",
    "io",
    "google.protobuf.text_format",
}

# Third party modules that need an "# already_there/" rule.
#
# While we assume that third party modules are already installed, we need to
# list then in comments in the rule "deps" section.
THIRD_PARTY_MODULES = {
    "numpy",
    "pandas",
    "apache_beam",
    "tempfile",
    "logging",
}

# Mapping from source file to rule.
#
# Use EXTRA_SOURCE_TO_RULE to specify mapping that cannot be detected by
# the build cleaner (e.g. proto, cc code).
EXTRA_SOURCE_TO_RULE = {
    ("temporian/proto/core_pb2.py", "//temporian/proto:core_py_proto"),
    (
        "temporian/implementation/numpy_cc/operators/operators_cc.py",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ),
}


# Rule prefix for third party module dependencies.
THIRD_PARTY_RULE_PREFIX = "# already_there/"

# Filename of BUILD files.
BUILD_FILENAMES = ["BUILD", "BUILD.bazel"]


@dataclass
class BuildRule:
    """A Bazel build rule."""

    name: str
    srcs: List[str]
    deps: List[str]
    rule: str


@dataclass
class DepsDelta:
    """Operations to apply on a build rule."""

    adds: List[Tuple[str, ...]]
    subs: List[Tuple[str, ...]]
    issues: List[str]


SourceToRule = Dict[Tuple[str, ...], List[Tuple[str, ...]]]


def get_keyword(call: ast.Call, key: str) -> Any:
    """Extracts the value of a keyword from a call."""

    for keyword in call.keywords:
        if keyword.arg == key:
            assert isinstance(keyword.value, ast.Constant)
            return keyword.value.value
    raise ValueError(f"Cannot find {key} in {ast.dump(call, indent=4)}")


def get_keyword_list_or_empty(call: ast.Call, key: str) -> List[Any]:
    """Extracts the value of a keyword from a call.

    The value is expected to be a list.
    Returns [] if the keyword does not exist.
    """

    for keyword in call.keywords:
        if keyword.arg == key:
            assert isinstance(keyword.value, ast.List)
            ret = []
            for v in keyword.value.elts:
                assert isinstance(v, ast.Constant)
                ret.append(v.value)
            return ret
    return []


def source_imports(path: str) -> List[str]:
    """Lists the imports in a .py file."""

    file_ast = ast.parse(open(path, encoding="utf-8").read())

    imports = []
    for item in file_ast.body:
        if isinstance(item, ast.Import):
            for name in item.names:
                imports.append(name.name)
        if isinstance(item, ast.ImportFrom):
            for name in item.names:
                assert item.module is not None
                imports.append(item.module + "." + name.name)
    return imports


def list_build_rules(path: str) -> List[BuildRule]:
    """List the build rules in a BUILD file."""

    build_content = open(path, encoding="utf-8").read()

    build_content = re.sub(
        THIRD_PARTY_RULE_PREFIX + r"(\S+)",
        r'"//\1",',
        build_content,
    )
    file_ast = ast.parse(build_content)

    rules: List[BuildRule] = []
    for item in file_ast.body:
        if not isinstance(item, ast.Expr):
            continue
        if not isinstance(item.value, ast.Call):
            continue
        if not isinstance(item.value.func, ast.Name):
            continue
        if item.value.func.id not in ALLOWED_BUILD_RULES:
            continue

        name = get_keyword(item.value, "name")
        deps = get_keyword_list_or_empty(item.value, "deps")
        srcs = get_keyword_list_or_empty(item.value, "srcs")
        assert isinstance(name, str)

        rules.append(
            BuildRule(name=name, deps=deps, srcs=srcs, rule=item.value.func.id)
        )

    return rules


def expand_dep(dep: str, rule_dir: str) -> Tuple[str, ...]:
    """Expands completely a bazel dep."""

    if dep[0] == ":":
        dep = rule_dir + dep
    elif dep[:2] == "//":
        dep = dep[2:]
    else:
        raise ValueError(f"Wrong dep format: {dep}")
    items = dep.split("/")
    final_split = items[-1].split(":")
    if len(final_split) == 1:
        items.append(items[-1])
    elif len(final_split) == 2:
        items.pop()
        items.extend(final_split)
    else:
        assert False
    return tuple(items)


def list_possible_source_of_import(
    imp_items: Tuple[str, ...]
) -> List[Tuple[str, ...]]:
    """List the possible source file of a given import."""

    assert len(imp_items) >= 1

    if imp_items[0] in THIRD_PARTY_MODULES:
        return [(imp_items[0], "__init__.py")]

    # Example of situations with import a.b.c (3)
    srcs = []

    # a/b.py
    if len(imp_items) >= 2:
        srcs.append(imp_items[:-2] + (imp_items[-2] + ".py",))

    # a/b/c.py
    srcs.append(imp_items[:-1] + (imp_items[-1] + ".py",))

    # a/b/c/__init__.py
    srcs.append(imp_items + ("__init__.py",))

    # a/b/__init__.py
    srcs.append(imp_items[:-1] + ("__init__.py",))

    return srcs


def extract_dirname_from_path(path: str) -> List[str]:
    """Decomposes a path into individual dirname.

    For example "a/b/c" becomes ["a", "b", "c"].
    """

    dirnames = []
    while True:
        path, dirname = os.path.split(path)
        if dirname and dirname != ".":
            dirnames.append(dirname)
        else:
            if path and path != ".":
                dirnames.append(path)
            break
    dirnames.reverse()
    return dirnames


def find_all_build_files(dir: str) -> List[Tuple[str, str]]:
    """List all the BUILD files.

    Returns:
        The list of (directory, filename) of all BUILD files.
    """

    build_file_dirs = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file in BUILD_FILENAMES:
                root = root.strip("./")
                build_file_dirs.append((root, file))
    return build_file_dirs


def find_source_to_rules(
    build_file_dirs: List[Tuple[str, str]]
) -> SourceToRule:
    """Mapping from source file to build rules."""

    source_to_rules = defaultdict(lambda: [])
    for build_file_dir, build_file_name in build_file_dirs:
        rule_base = tuple(extract_dirname_from_path(build_file_dir))
        file_rules = list_build_rules(
            os.path.join(build_file_dir, build_file_name)
        )
        for rule in file_rules:
            for src in rule.srcs:
                source_to_rules[rule_base + (src,)].append(
                    rule_base + (rule.name,)
                )
    return source_to_rules


def compute_delta(
    deps: List[str],
    imports: List[str],
    rule_dir: str,
    source_to_rules: SourceToRule,
) -> Optional[DepsDelta]:
    """Computes the operation on the deps to support all the imports.

    Args:
        deps: Dependencies of the rule.
        imports: Imports of the rule.
        rule_dir: Path of the rule relative to the repo root.
        source_to_rules: Mapping from all available source files to rules.
    """

    issues = []
    adds = set()
    subs = set()

    # The current dependencies of the rule.
    expanded_deps = set([expand_dep(dep, rule_dir) for dep in deps])

    # Dependencies effectively used by this rule.
    used_deps = set()

    # used_dependencies = [False] * len(expanded_deps)

    for imp in imports:
        imp_items = tuple(imp.split("."))

        # This import does not need a build rule.
        if imp_items[0] in BUILT_IN_MODULES or imp in BUILT_IN_MODULES:
            continue

        # The source files that might solve this import.
        possible_srcs = list_possible_source_of_import(imp_items)
        matching_possible_src = None
        for possible_src in possible_srcs:
            if possible_src in source_to_rules:
                matching_possible_src = possible_src
                break

        if matching_possible_src is None:
            issues.append(
                f'Cannot infer dependency for "{imp}". Possible source files:'
                f" {possible_srcs}."
            )
            continue

        possible_deps = source_to_rules[matching_possible_src]

        if len(possible_deps) > 1:
            issues.append(f'Multiple possible rules for "{imp}"')

        if possible_deps[0] not in expanded_deps:
            adds.add(possible_deps[0])
        else:
            used_deps.add(possible_deps[0])

    for dep in expanded_deps:
        if dep in used_deps:
            continue
        subs.add(dep)

    if adds or subs or issues:
        return DepsDelta(adds=list(adds), subs=list(subs), issues=issues)
    else:
        return None


def to_user_rule(normalized_rule: Tuple[str, ...]) -> str:
    """Converts a tuple rule to a user rule.

    Example: ("a", "b", "c") => "//a/b:c".
    """

    if (
        len(normalized_rule) == 2
        and normalized_rule[0] == normalized_rule[1]
        and normalized_rule[0] in THIRD_PARTY_MODULES
    ):
        return THIRD_PARTY_RULE_PREFIX + normalized_rule[0] + ","

    if len(normalized_rule) >= 2 and normalized_rule[-1] == normalized_rule[-2]:
        return '"//' + "/".join(normalized_rule[:-1]) + '",'
    else:
        return (
            '"//'
            + "/".join(normalized_rule[:-1])
            + ":"
            + normalized_rule[-1]
            + '",'
        )


def clean_build_rule(
    rule: BuildRule, rule_dir: str, source_to_rules: SourceToRule
):
    """Clean a BUILD file."""

    imports = []
    for src in rule.srcs:
        src_path = os.path.join(rule_dir, src)
        imports.extend(source_imports(src_path))

    return compute_delta(rule.deps, imports, rule_dir, source_to_rules)


def clean_repository(dir: str):
    print("List BUILD files")
    build_file_dirs = find_all_build_files(dir)
    print(f"Found {len(build_file_dirs)} build files")

    print("List rules")
    source_to_rules = find_source_to_rules(build_file_dirs)
    for name in THIRD_PARTY_MODULES:
        source_to_rules[(name, "__init__.py")].append((name, name))
    for src, rule in EXTRA_SOURCE_TO_RULE:
        source_to_rules[tuple(extract_dirname_from_path(src))].append(
            expand_dep(rule, "")
        )
    print(f"Found {len(source_to_rules)} source files")

    num_adds = 0
    num_subs = 0
    num_issues = 0

    for build_file_dir, build_file_name in build_file_dirs:
        build_file_path = os.path.join(build_file_dir, build_file_name)
        rules = list_build_rules(build_file_path)
        in_is_shown = False
        for rule in rules:
            delta = clean_build_rule(rule, build_file_dir, source_to_rules)
            if delta is not None:
                num_adds += len(delta.adds)
                num_subs += len(delta.subs)
                num_issues += len(delta.issues)

                if not in_is_shown:
                    print(f"In {Fore.CYAN}{build_file_path}{Fore.RESET}")
                    in_is_shown = True

                print(
                    "  Rule"
                    f" {Fore.CYAN}//{build_file_dir}:{rule.name}{Fore.RESET}"
                )
                if delta.issues:
                    print("    Issues:")
                    for issue in delta.issues:
                        print(f"      {Fore.MAGENTA}{issue}{Fore.RESET}")
                if delta.adds:
                    print("    Add:")
                    for add in delta.adds:
                        print(
                            f"      {Fore.GREEN}{to_user_rule(add)}{Fore.RESET}"
                        )
                if delta.subs:
                    print("    Remove:")
                    for sub in delta.subs:
                        print(
                            f"      {Fore.RED}{to_user_rule(sub)}{Fore.RESET}"
                        )
    print(f"{num_adds} additions, {num_subs} removals, {num_issues} issues")


def main():
    colorama.init()
    clean_repository(".")


if __name__ == "__main__":
    main()
