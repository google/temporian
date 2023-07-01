import os

from absl.testing import absltest

from tools import build_cleaner as bc


class BuildCleaner(absltest.TestCase):
    def test_expand_dep_rule(self):
        self.assertEqual(bc.expand_dep_rule(":c", "a/b"), ("a", "b", "c"))
        self.assertEqual(bc.expand_dep_rule("//a/b:c", "c/d"), ("a", "b", "c"))
        self.assertEqual(bc.expand_dep_rule("//a/b", "c/d"), ("a", "b", "b"))

    def test_list_possible_deps_of_import(self):
        self.assertEqual(
            bc.list_possible_deps_of_import(("a", "b")),
            [
                ("a", "a"),
                ("a", "b", "b"),
                ("a",),
                ("a", "b"),
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


if __name__ == "__main__":
    absltest.main()
