"""Clean dependencies in Bazel Python rules.

How to use the build cleaner.

    # Run the following command to get the list of deps to add and remove.
    # build_cleaner.py does not modify any file..
    tools/build_cleaner.py

If case of a false positive.
    - Update "DEP_LESS_IMPORTS" if a build-in Python module is not detected.
    - Update "THIRD_PARTY_DEPS" if a third party module is not detected.

Limitations
    - The build cleaner assumes that each python source file has a rule with the
        same name. For Example "a/b/c.py" is available in the rule "//a/b:c".
    - In some cases, import of module and import of symbol cannot be
        differentiated without parsing the python source code. In this case,
        build cleaner does not try to parse the python source code and rely on a
        permissive heuristic (possible false negative).
"""

import ast
from dataclasses import dataclass
import re
import os
import sys
from typing import List, Any, Tuple, Optional

import colorama
from colorama import Fore

# Bazel rules to scan for dependencies.
ALLOWED_BUILD_RULES = {"py_library"}

# Modules that do not require a bazel dependency.
DEP_LESS_IMPORTS = set(
    list(sys.modules.keys())
    + [
        "datetime",
        "math",
        "csv",
    ]
)

# Third party modules that need a "# already_there/" rule.
THIRD_PARTY_DEPS = {
    "numpy",
    "pandas",
    "apache_beam",
    "tempfile",
    "logging",
}

# Rule prefix for third party module dependencies.
THIRD_PARTY_RULE_PREFIX = "# already_there/"

# Filename of BUILD files.
BUILD_FILENAME = "BUILD"

# Import for Temporian protos.
# i.e. import temporian.proto.core_pb2
PROTO_IMPORT = ("temporian", "proto", "core_pb2")

# The dependency rule for the Temporian protos.
# i.e. //temporian/proto:core_py_proto
PROTO_RULE = ("temporian", "proto", "core_py_proto")


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


def expand_dep_rule(dep: str, rule_dir: str) -> Tuple[str, ...]:
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


def list_possible_deps_of_import(
    imp_items: Tuple[str, ...]
) -> List[Tuple[str, ...]]:
    """List the possible dependencies of a given import."""

    assert isinstance(imp_items, tuple)
    assert len(imp_items) >= 1

    # This is a proto.
    if imp_items == PROTO_IMPORT:
        return [PROTO_RULE]

    # This is a third party dependency where a single dependency to the library
    # is necessary.
    if imp_items[0] in THIRD_PARTY_DEPS:
        return [(imp_items[0], imp_items[0])]

    # Example of situations with import a.b.c (3)
    deps = []

    # 	//a/b:b (3)
    # 	b is a main package, c is a symbol
    if len(imp_items) >= 2:
        deps.append(imp_items[:-1] + (imp_items[-2],))

    # 	//a/b/c:c (4)
    # 	c is a main package
    deps.append(imp_items + (imp_items[-1],))

    # 	//a:b (2)
    # 	b is a non-main package, c is a symbol
    if len(imp_items) >= 2:
        deps.append(imp_items[:-1])

    # 	//a/b:c (3)
    # 	c is a non-main package
    deps.append(imp_items)

    return deps


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


def find_all_build_files(dir: str) -> List[str]:
    """List all the BUILD files."""

    build_file_dirs = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file == BUILD_FILENAME:
                root = root.strip("./")
                build_file_dirs.append(root)
    return build_file_dirs


def find_all_rules(build_file_dirs: List[str]) -> List[Tuple[str, ...]]:
    """List all the rules."""

    rules = []
    for build_file_dir in build_file_dirs:
        rule_base = tuple(extract_dirname_from_path(build_file_dir))
        file_rules = list_build_rules(
            os.path.join(build_file_dir, BUILD_FILENAME)
        )
        for file_rule in file_rules:
            rules.append(rule_base + (file_rule.name,))
    return rules


def compute_delta(
    deps: List[str],
    imports: List[str],
    rule_dir: str,
    all_rules: List[Tuple[str, ...]],
) -> Optional[DepsDelta]:
    """Computes the operation on the deps to support all the imports."""

    issues = []
    adds = set()
    subs = set()

    expanded_deps = [expand_dep_rule(dep, rule_dir) for dep in deps]
    used_dependencies = [False] * len(expanded_deps)

    all_rule_set = set(all_rules)

    for imp in imports:
        imp_items = tuple(imp.split("."))
        if imp_items[0] in DEP_LESS_IMPORTS:
            continue
        acceptable_deps = list_possible_deps_of_import(imp_items)
        possible_deps = list(set(acceptable_deps).intersection(all_rule_set))

        match_dep_idx = -1
        for dep_idx, dep in enumerate(expanded_deps):
            if dep in possible_deps:
                match_dep_idx = dep_idx
                break

        if match_dep_idx == -1:
            if len(possible_deps) == 0:
                issues.append(
                    f'Cannot infer dependency for "{".".join(imp_items)}"'
                )
            else:
                adds.add(possible_deps[0])
        else:
            used_dependencies[match_dep_idx] = True

    for dep_idx, is_used in enumerate(used_dependencies):
        if is_used:
            continue
        subs.add(expanded_deps[dep_idx])

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
        and normalized_rule[0] in THIRD_PARTY_DEPS
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
    rule: BuildRule, rule_dir: str, all_rules: List[Tuple[str, ...]]
):
    """Clean a BUILD file."""

    imports = []
    for src in rule.srcs:
        src_path = os.path.join(rule_dir, src)
        imports.extend(source_imports(src_path))

    return compute_delta(rule.deps, imports, rule_dir, all_rules)


def clean_repository(dir: str):
    print("List BUILD files")
    build_file_dirs = find_all_build_files(dir)
    print(f"Found {len(build_file_dirs)} build files")

    print("List rules")
    all_rules = find_all_rules(build_file_dirs)
    for rule in THIRD_PARTY_DEPS:
        all_rules.append((rule, rule))
    all_rules.append(PROTO_RULE)
    print(f"Found {len(all_rules)} rules")

    num_adds = 0
    num_subs = 0
    num_issues = 0

    for build_file_dir in build_file_dirs:
        build_file_path = os.path.join(build_file_dir, BUILD_FILENAME)
        rules = list_build_rules(build_file_path)
        in_is_shown = False
        for rule in rules:
            delta = clean_build_rule(rule, build_file_dir, all_rules)
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
