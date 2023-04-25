# %% [markdown]
"""
# Financial Database
In this example we retrieve and prepare the data from the [Financial
relational dataset](https://relational.fit.cvut.cz/dataset/Financial)
using Temporian.

This dataset contains 682 loans (606 successful and 76 not successful)
along with their information and transactions.
The standard task is to predict the loan outcome for
finished loans (A vs B in loan.status)
at the time of the loan start (defined by loan.date).

Since the dataset is provided in a relational database, the package
sqlalchemy needs to be installed, and the connection to the DB is performed
using the public credentials provided in the link above.
"""

# %% [markdown]
"""
## Dependencies
"""
# %%
# !pip install "sqlalchemy<2.0" pymysql

# %%
import pandas as pd
import temporian as tp
from pathlib import Path
from time import time
from sqlalchemy import create_engine

# %% [markdown]
"""
## Load dataframes
"""
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
    df_trx = pd.read_csv(path_df_trx)
    df_loan = pd.read_csv(path_df_loan)

# %%
# Check we've 682 loans
df_loan

# %%
# Check 1_056_320 transactions
df_trx

# %%
# TODO:
# - filter only transactions prior to loan date
# - some plots (e.g: loan dates with amounts -added up per day?-)
# - plot transactions for a good and bad loan
