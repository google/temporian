import temporian as tp

import pdb;pdb.set_trace()
# Load sale transactions
sales = tp.from_csv("sales.csv")
sales_df = tp.to_pandas(sales)

# Index sales per store
sales_per_store = sales.add_index("store")

# List work days
days = sales_per_store.tick_calendar(hour=22)
work_days = (days.calendar_day_of_week() <= 5).filter()

work_days.plot(max_num_plots=1)
