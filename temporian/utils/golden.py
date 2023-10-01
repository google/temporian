"""Utility to compare generated and golden unit test data."""

import logging

from temporian.test.utils import get_test_data_path


def check_string(test, value: str, golden_path: str):
    effective_golden_path = get_test_data_path(golden_path)
    golden_data = open(effective_golden_path).read()

    if value != golden_data:
        value_path = "/tmp/golden_test_value.html"
        logging.info("Save effective value of golden test in %s", value_path)
        logging.info(
            "Update the golden file with: cp %s %s", value_path, golden_path
        )
        with open(value_path, "w") as f:
            f.write(value)

    test.assertEqual(value, golden_data)
