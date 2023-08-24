"""Utility to compare generated and golden unit test data."""

import os
from absl import flags
import logging


def test_data() -> str:
    return os.path.join(flags.FLAGS.test_srcdir, "temporian")


def check_string(test, value: str, golden_path: str):
    effective_golden_path = os.path.join(test_data(), golden_path)
    golden_data = open(effective_golden_path).read()

    if value != golden_data:
        value_path = "/tmp/golden_test_value.html"
        logging.info("Save effetive value of golden test in %s", value_path)
        logging.info(
            "Update the golden file with: cp %s %s", value_path, golden_path
        )
        with open(value_path, "w") as f:
            f.write(value)

    test.assertEqual(value, golden_data)
