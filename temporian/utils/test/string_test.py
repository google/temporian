from absl.testing import absltest

from temporian.utils import string


class FormatTest(absltest.TestCase):
    def test_indent(self):
        self.assertEqual(string.indent(""), "  ")
        self.assertEqual(string.indent("hello"), "  hello")
        self.assertEqual(string.indent("hello\n"), "  hello\n")
        self.assertEqual(string.indent("hello\nworld"), "  hello\n  world")
        self.assertEqual(string.indent("hello\nworld\n"), "  hello\n  world\n")


if __name__ == "__main__":
    absltest.main()
