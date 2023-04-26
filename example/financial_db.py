# %% [markdown]
# # Financial Database
# In this example we retrieve and prepare the data from the [Financial
# relational dataset](https://relational.fit.cvut.cz/dataset/Financial)
# using Temporian.
#
# This dataset contains 682 loans (606 successful and 76 not successful)
# along with their information and transactions.
# The standard task is to predict the loan outcome for
# finished loans (A vs B in loan.status)
# at the time of the loan start (defined by loan.date).
#
# Since the dataset is provided in a relational database, the package
# sqlalchemy needs to be installed, and the connection to the DB is performed
# using the public credentials provided in the link above.

# %% [markdown]
# ## Dependencies

# %%
# !pip install "sqlalchemy<2.0" pymysql

# %%
import pandas as pd

import temporian as tp
from pathlib import Path
from time import time
from sqlalchemy import create_engine

# %% [markdown]
# ## Load dataframes

# %%
tmp_dir = Path("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

path_df_loan = tmp_dir / "loan.csv"
path_df_trx = tmp_dir / "trans.csv"

if not path_df_loan.exists() or not path_df_trx.exists():
    print("Connecting to DB...")
    conn_str = (
        "mysql+pymysql://guest:relational@relational.fit.cvut.cz:3306/financial"
    )
    engine = create_engine(conn_str)
    print("Retrieving table: loan -> df_loan")
    t0 = time()
    df_loan = pd.read_sql("SELECT * FROM loan", engine)
    t1 = time()
    print(f"Took {t1 - t0:.1f} seconds")

    print("Retrieving table: trans -> df_trx")
    df_trx = pd.read_sql("SELECT * FROM trans", engine)
    print(f"Took {time() - t1:.1f} seconds")

    print(f"Caching tables into: {tmp_dir}")
    df_trx.to_csv(path_df_trx)
    df_loan.to_csv(path_df_loan)
else:
    print(f"Loading cached tables from {tmp_dir}")
    df_trx = pd.read_csv(path_df_trx, index_col=[0])
    df_loan = pd.read_csv(path_df_loan, index_col=[0])

# %%
# Check that all accounts have a single loan
assert not df_loan["account_id"].duplicated().any()

# Check we've 682 loans
df_loan

# %%
# Check 1_056_320 transactions
df_trx

# %%
# Convert date columns
df_loan["date"] = pd.to_datetime(df_loan["date"])
df_trx["date"] = pd.to_datetime(df_trx["date"])

# Remove unused columns
df_loan.drop(columns=["loan_id"], inplace=True, errors="ignore")

# Remove trx without loan and get loan dates into transactions
df_trx = df_trx.join(
    df_loan[["account_id", "date"]],
    on=["account_id"],
    how="inner",
    rsuffix="_loan",
)

# Only transactions before the loan can be used for prediction
df_trx["valid"] = df_trx["date_loan"] >= df_trx["date"]

# Delete no longer used
df_trx.drop(columns=["date_loan", "account_id_loan"], inplace=True)

print(f"Valid transactions ({len(df_trx)=}):")
df_trx["valid"].value_counts()

# %% [markdown]
# ## Create Temporian Events

# %%
# Create events from the dataframes
e_loan_data = tp.EventData.from_dataframe(
    df_loan, timestamp_column="date", index_names=["account_id"]
)
e_loan_data

# %%
e_trx_data = tp.EventData.from_dataframe(
    df_trx, timestamp_column="date", index_names=["account_id"]
)
e_trx_data

# %% [markdown]
# ### Check transaction plots

# %%
# Visualize all the data
e_trx_data.plot()

# %% [markdown]
# ### Smooth curves and filter some columns to visualize

# %%
# Get event node
source_trx = e_trx_data.schema()

# Select only some columns for the event
e_trx = source_trx[["amount", "balance", "valid"]]

# Filter only valid data
e_trx = tp.filter(e_trx, e_trx["valid"])

# Count the number of transactions per day and month
day_n_trx = tp.moving_count(e_trx["amount"], tp.duration.days(1.0))
month_n_trx = tp.moving_count(e_trx["amount"], tp.duration.weeks(4.0))

# TODO: Fix
# Augment feature: monthly average amount
# monthly_amount = tp.simple_moving_average(e_trx["amount"], tp.duration.weeks(4.0))

# TODO: Count the number of VYDAJ types per month

result = tp.glue(
    e_trx,
    tp.prefix("daily_count.", day_n_trx),
    tp.prefix("monthly_count.", month_n_trx),
    # tp.prefix("monthly_sum.", monthly_amount),
)

result_data = tp.evaluate(result, {source_trx: e_trx_data})

result_data

# Select only two accounts to plot
# result_data.plot(indexes=[(5,), (10,)])
result_data.plot()

# %%
