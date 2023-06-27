#!/usr/bin/env python

""" Minimal script to test that temporian is installed and running well.

Usage example:
    pip install -U temporian
    ./tools/test_install.py
"""

import numpy as np
import pandas as pd
import temporian as tp


def check_install():
    # Generate a synthetic dataset
    timestamps = np.arange(0, 100, 0.1)
    source_evset = tp.from_pandas(
        pd.DataFrame({"timestamp": timestamps, "signal": np.sin(timestamps)})
    )
    source_node = source_evset.node()
    sma = tp.simple_moving_average(
        source_node["signal"], tp.duration.seconds(30)
    )

    return tp.run(sma, {source_node: source_evset})


if __name__ == "__main__":
    print(check_install())
    print("Temporian executed OK.")
