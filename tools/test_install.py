#!/usr/bin/env python

""" Minimal script to test that temporian is installed and running well.

Usage example:
    pip install temporian
    ./tools/test_install.py
"""

import numpy as np
import pandas as pd
import temporian as tp

# Generate a synthetic dataset
timestamps = np.arange(0, 100, 0.1)
source_data = tp.EventSet.from_dataframe(
    pd.DataFrame({"timestamp": timestamps, "signal": np.sin(timestamps)})
)
source_node = source_data.node()
sma = tp.simple_moving_average(source_node["signal"], tp.duration.seconds(30))

result_data = tp.evaluate(sma, {source_node: source_data})
print(result_data)
print("Temporian executed OK.")
