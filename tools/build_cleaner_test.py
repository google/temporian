import os

from absl.testing import absltest

from tools import build_cleaner as bc


class BuildCleaner(absltest.TestCase):
    def test_expand_dep(self):
        self.assertEqual(bc.expand_dep(":c", "a/b"), ("a", "b", "c"))
        self.assertEqual(bc.expand_dep("//a/b:c", "c/d"), ("a", "b", "c"))
        self.assertEqual(bc.expand_dep("//a/b", "c/d"), ("a", "b", "b"))

    def test_list_possible_source_of_import(self):
        self.assertEqual(
            bc.list_possible_source_of_import(("a", "b")),
            [
                ("a.py",),
                ("a", "b.py"),
                ("a", "b", "__init__.py"),
                ("a", "__init__.py"),
            ],
        )

    def test_extract_dirname_from_path(self):
        self.assertEqual(
            bc.extract_dirname_from_path(os.path.join("a", "b", "c")),
            ["a", "b", "c"],
        )

    def test_to_user_rule(self):
        self.assertEqual(bc.to_user_rule(("a", "a")), '"//a",')
        self.assertEqual(bc.to_user_rule(("a", "b")), '"//a:b",')
        self.assertEqual(
            bc.to_user_rule(("numpy", "numpy")), "# already_there/numpy,"
        )

    def test_compute_delta(self):
        delta = bc.compute_delta(
            deps=["//a:b", "//a:c"],
            imports=["a.b", "a.d"],
            rule_dir="a",
            source_to_rules={
                ("a", "b.py"): [("a", "b")],
                ("a", "d.py"): [("a", "d")],
                ("a", "e.py"): [("a", "e")],
            },
        )
        self.assertEqual(
            delta,
            bc.DepsDelta(
                adds=[("a", "d")],
                subs=[("a", "c")],
                issues=[],
            ),
        )


if __name__ == "__main__":
    absltest.main()
