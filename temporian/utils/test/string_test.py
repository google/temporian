from absl.testing import absltest

from temporian.utils import string


class FormatTest(absltest.TestCase):
    def test_indent(self):
        self.assertEqual(string.indent(""), "    ")
        self.assertEqual(string.indent("hello"), "    hello")
        self.assertEqual(string.indent("hello\n"), "    hello\n")
        self.assertEqual(string.indent("hello\nworld"), "    hello\n    world")
        self.assertEqual(
            string.indent("hello\nworld\n"), "    hello\n    world\n"
        )

    def test_pretty_num_bytes(self):
        self.assertEqual(string.pretty_num_bytes(0), "0 B")
        self.assertEqual(string.pretty_num_bytes(500), "500 B")
        self.assertEqual(string.pretty_num_bytes(501), "0.5 kB")
        self.assertEqual(string.pretty_num_bytes(int(1e3)), "1.0 kB")
        self.assertEqual(string.pretty_num_bytes(int(1e6)), "1.0 MB")
        self.assertEqual(string.pretty_num_bytes(int(1e9)), "1.0 GB")
        self.assertEqual(string.pretty_num_bytes(int(1e12)), "1000.0 GB")


if __name__ == "__main__":
    absltest.main()
