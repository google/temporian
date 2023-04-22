"""
The M5 Competition, held in 2020, was part of the prestigious Makridakis
Forecasting Competitions. The goal of this competition was to accurately
forecast the sales of 3,000 individual items across 10 Walmart stores for the
next 28 days. The winning strategy involved transforming the complex
multi-variate time series dataset into a more manageable tabular format, which
was then used to train a collection of Gradient Boosted Trees models.

See https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview.

In this example, we will explore how to use the Temporian library to replicate
this transformation from time series into a tabular dataset. By following the
step-by-step guide provided, you will learn how to leverage the library's
powerful functionalities to replicate transformations of your time series data.
With Temporian, converting time series data into a tabular format has never been
easier.
"""

# NOTE: This example can be used to test the API.
# TODO: Turn into a proper tutorial.

import os
from datetime import datetime, timezone, timedelta
import urllib.request
import tempfile

import zipfile
import pandas as pd
import numpy as np

import temporian as tp

# Directory used to download the raw M5 dataset and to export the tabular
# dataset.
work_directory = os.path.join(tempfile.gettempdir(), "m5")
os.makedirs(work_directory, exist_ok=True)
print("Work directory (contains the output artefacts):", work_directory)

# Download the M5 dataset
raw_data_zip = os.path.join(work_directory, "raw.zip")
if not os.path.exists(raw_data_zip):
    print("Download M5 dataset in", raw_data_zip)
    # Note: This url is a copy of the M5 dataset we did.
    # TODO: Find a way to download the M5 dataset from the original location.
    url = "https://docs.google.com/uc?export=download&id=1NYHXmgrcXg50zR4CVWPPntHx9vvU5jbM&confirm=t"
    urllib.request.urlretrieve(url, raw_data_zip)

# Extract the M5 dataset
raw_data_dir = os.path.join(work_directory, "raw")
if not os.path.exists(raw_data_dir):
    print("Extract M5 dataset in", raw_data_dir)
    with zipfile.ZipFile(raw_data_zip, "r") as zip_ref:
        zip_ref.extractall(raw_data_dir)


# Load raw dataset
print("Load raw dataset")
print("================")

# During development, set nrows to a small value (e.g. nrows=100) to
# only load a sample of the dataset.
#
# TODO: Set to None when fast enough.
nrows = 100

raw_path = lambda x: os.path.join(raw_data_dir, x)
sales_raw = pd.read_csv(raw_path("sales_train_evaluation.csv"), nrows=nrows)
sell_prices_raw = pd.read_csv(raw_path("sell_prices.csv"), nrows=nrows)
calendar_raw = pd.read_csv(raw_path("calendar.csv"))

print("sales_raw:")
sales_raw.info()
print("sell_prices_raw:")
sell_prices_raw.info()
print("calendar_raw:")
calendar_raw.info()

# Melt dataset
#
# The different timesteps of sale data ("sales_raw") are stored in different
# columns. Instead, we want for each timesteps to be a different row.
#
# For example:
#   The record "id,item_id,dept_id,d_1,d_2,d_3...d_n" will be converted into
#   n records "id,item_id,dept_id,day,sales".
#
print("Melt dataset")
print("============")

sales_raw = pd.melt(
    sales_raw,
    var_name="day",
    value_name="sales",
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
)
sales_raw["day"] = sales_raw["day"].apply(lambda x: int(x[2:]))

print("sales_raw:\n", sales_raw)

# Compute timestamps
#
# In the M5 raw dataset, timestamps are stored in three different ways in the
# three different tables. To make our life easy, we will convert all the
# timestamps into python datetimes objects.

print("Compute timestamps")
print("==================")

# In "sales_raw", timestamps are expressed in number of days since
# 29/1/2011. Let's convert them into python datetimes.
origin_date = datetime(2011, 1, 29)  # , tzinfo=timezone.utc
sales_raw["timestamp"] = sales_raw["day"].apply(
    lambda x: (origin_date + timedelta(days=x - 1))
)

# In "calendar_raw", timestamps are expressed as "2011-01-29" like strings.
# Let's convert them into python datetimes.

# TODO: Check if date conversion is correct, regarding UTC.
calendar_raw["timestamp"] = calendar_raw["date"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d")
)

# In "sell_prices_raw", timestamps as expressed with a special index "wm_yr_wk"
# defined in "calendar_raw". Let's convert them into python datetimes.
wm_yr_wk_to_date = calendar_raw[calendar_raw["weekday"] == "Saturday"][
    ["timestamp", "wm_yr_wk"]
]
map_wm_yr_wk_to_date = {}
for _, row in wm_yr_wk_to_date.iterrows():
    map_wm_yr_wk_to_date[row["wm_yr_wk"]] = row["timestamp"]

sell_prices_raw["timestamp"] = sell_prices_raw["wm_yr_wk"].apply(
    lambda x: map_wm_yr_wk_to_date[x]
)

print("sales_raw:\n", sales_raw)
print("sell_prices_raw:\n", sell_prices_raw)
print("calendar_raw:\n", calendar_raw)

# Remove unused data + cast
#
# We can remove and release the memory for all the old timestamp data.
print("Remove unused data + cast")
print("=========================")

del sales_raw["id"]
del sales_raw["day"]
del sell_prices_raw["wm_yr_wk"]
del calendar_raw["date"]
del calendar_raw["wm_yr_wk"]
del calendar_raw["d"]
del calendar_raw["weekday"]
del calendar_raw["wday"]
del calendar_raw["month"]
del calendar_raw["year"]

# TODO: Use int16 when available.
sales_raw["sales"] = sales_raw["sales"].astype(np.int32)
# TODO: Use boolean when available.
calendar_raw["snap_CA"] = calendar_raw["snap_CA"].astype(np.int32)
calendar_raw["snap_TX"] = calendar_raw["snap_TX"].astype(np.int32)
calendar_raw["snap_WI"] = calendar_raw["snap_WI"].astype(np.int32)

# TODO: Make the case in Temporian. Float32 is required for the SMA operator.
sales_raw["sales"] = sales_raw["sales"].astype(np.float32)

# Convert to Temporian Events
#
# We can finally convert the dataset to Temporian format.

print("Convert to Temporian Events")
print("===========================")

sales_data = tp.EventData.from_dataframe(
    sales_raw,
    index_names=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
)

calendar_data = tp.EventData.from_dataframe(calendar_raw)

sell_prices_data = tp.EventData.from_dataframe(
    sell_prices_raw,
    index_names=["store_id", "item_id"],
)

print("sales_data:\n", sales_data)
print("sell_prices_data:\n", sell_prices_data)
print("calendar_data:\n", calendar_data)

# Now that our data is in the Temporian format. We can delete the Pandas
# dataframe objects.

print("Release Pandas dataframes")
print("===========================")

del sales_raw
del sell_prices_raw
del calendar_raw

# Plot raw data
print("Plot raw data")
print("============")

plot_options = {
    # We only plot the 1st year of data to make the plot more readable.
    "min_time": datetime(2015, 1, 1),
    "max_time": datetime(2016, 1, 1),
}

sales_data.plot(**plot_options).savefig(
    os.path.join(work_directory, "raw_sales.png")
)
calendar_data.plot(**plot_options).savefig(
    os.path.join(work_directory, "raw_calendar.png")
)
sell_prices_data.plot(**plot_options).savefig(
    os.path.join(work_directory, "raw_sell_prices.png")
)

# Convert time series to tabular dataset
print("Convert time series to tabular dataset")
print("======================================")

# We define the computation graph
sales = sales_data.schema()
calendar = calendar_data.schema()
sell_prices = sell_prices_data.schema()

augmented_sales = tp.glue(
    # Moving average of sales
    tp.prefix("sma_7.", tp.simple_moving_average(sales, tp.duration.days(7))),
    tp.prefix("sma_28.", tp.simple_moving_average(sales, tp.duration.days(28))),
    # Sum of sales
    tp.prefix("sum_7.", tp.moving_sum(sales, tp.duration.days(7))),
    tp.prefix("sum_28.", tp.moving_sum(sales, tp.duration.days(28))),
)

# Lagged sales
lagged_sales = []
for lag in [1, 2]:
    lagged_sales.append(
        tp.sample(
            tp.prefix(f"lag_{lag}.", tp.lag(sales, tp.duration.days(lag))),
            sales,
        )
    )
lagged_sales = tp.glue(*lagged_sales)


calendar_events = tp.glue(
    tp.calendar_day_of_week(sales),
    tp.calendar_day_of_month(sales),
    tp.calendar_month(sales),
)

label_sales = []
for lag in [1, 2, 3]:
    label_sales.append(
        tp.sample(
            tp.prefix(f"leak_{lag}.", tp.leak(sales, tp.duration.days(lag))),
            sales,
        )
    )
label_sales = tp.glue(*label_sales)

# Sum of the sales, sampled each day, per departement.
sales_per_dept = tp.drop_index(sales, "item_id")
sampling_once_per_day = tp.unique_timestamps(sales_per_dept["item_id"])
sum_dayly_sales_per_dept = tp.prefix(
    "per_dept.sum_28_",
    tp.moving_sum(
        sales_per_dept["sales"],
        tp.duration.days(28),
        sampling_once_per_day,
    ),
)

# For each item, add the sum of departement sales for this specific item.
sales_aggregated_item_level = tp.sample(
    tp.propagate(sum_dayly_sales_per_dept, sales), sales
)

# TODO: Last calendar events (need since_last, filter)
# TODO: Skip early time (start + a few days)
# TODO: Skip before first sales (need filter)
# TODO: Comparators + boolean operators

tabular_dataset = tp.glue(
    sales,  # Raw sales
    sales_aggregated_item_level,
    lagged_sales,
    label_sales,
    augmented_sales,
    calendar_events,
)

# We run the computation
tabular_dataset_data = tp.evaluate(
    tabular_dataset,
    {
        sales: sales_data,
        calendar: calendar_data,
        sell_prices: sell_prices_data,
    },
)

# Plot results
print("Plot results")
print("============")

tabular_dataset_data.plot(**plot_options).savefig(
    os.path.join(work_directory, "tabular_dataset.png")
)

# Export to csv file
print("Export to csv file")
print("==================")

tabular_dataset_data.to_dataframe().to_csv(
    os.path.join(work_directory, "tabular_dataset.csv"), index=False
)
